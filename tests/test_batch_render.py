"""
Check the correctness of batch_render
python3 tests/test_batch_render.py -s <path/to/dataset> --start_checkpoint <path/to/checkpoint> --num_images <number_of_images_in_batch>
"""

import os
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from random import randint
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, safe_interact
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
from contextlib import contextmanager

from gaussian_renderer import render
from gaussian_renderer.batch_render import batch_render
from gaussian_renderer.reference_render import reference_render

def training(dataset, opt, pipe, checkpoint, num_images):
    np.random.seed(0)
    torch.manual_seed(0)

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        print(f"Restoring from checkpoint: {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iteration = first_iter

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))

    num_batch_cameras = min(num_images, len(viewpoint_indices))
    rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
    print(f"\nUsing {num_batch_cameras} random cameras: {rand_indices}")
    loss_functions = []
    # Same background for all cameras in the batch
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    # gaussians.save_ply(os.path.join(scene.model_path, "gaussians.ply"))
    viewpoint_cams = []
    scalar_losses = []
    vector_losses = []
    render_pkgs = []

    for i, rand_idx in enumerate(rand_indices):
        viewpoint_cam = viewpoint_stack[rand_idx]
        viewpoint_cams.append(viewpoint_cam)

    print("rand_indices:", rand_indices)

    batch_render_pkg = batch_render(viewpoint_cams, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)

    P = gaussians.get_xyz.shape[0]
    max_radii = torch.zeros((P, ), dtype=torch.int32, device="cuda")
    visibility_mask = torch.zeros((P, ), dtype=torch.bool, device="cuda")

    for i, viewpoint_cam in enumerate(viewpoint_cams):
        render_pkg = reference_render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        render_pkgs.append(render_pkg)

        for key in ["render", "depth"]:
            assert torch.allclose(batch_render_pkg[key][i], render_pkg[key], atol=1e-6), f"{key} do not match for camera {i}!"

        max_radii = torch.max(max_radii, render_pkg["radii"])
        visibility_mask[render_pkg["visibility_filter"]] = True

    assert torch.all(batch_render_pkg["max_radii"] == max_radii), "max_radii do not match!"
    assert torch.all(batch_render_pkg["visibility_filter"] == visibility_mask.nonzero()), "visibility_filter do not match!"

    print("All checks passed!")

    import code; code.interact(local=locals())


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--num_images", type=int, default = 1)
    args = parser.parse_args(sys.argv[1:])
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.num_images)

