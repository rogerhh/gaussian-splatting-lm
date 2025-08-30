"""
Check how much memory batch_training_loss uses in rendering
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

from functools import partial
from solver.gaussian_model_state import GaussianModelState
from solver.training_loss import training_loss, scalar_training_loss
from solver.batch_training_loss import batch_training_loss
from solver.solver_functions import LinearSolverFunctions
from solver.conjugate_gradient import cgls_damped

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }

    print("Return at render end")
    return

def print_backward_graph(fn, indent=0, seen=None):
    if fn is None:
        return
    if seen is None:
        seen = set()
    if id(fn) in seen:
        print(" " * indent + f"{repr(fn)} (already visited)")
        return
    seen.add(id(fn))

    print(" " * indent + repr(fn))

    # Try both possibilities
    next_fns = None
    if hasattr(fn, 'next_functions'):
        next_fns = fn.next_functions
    elif hasattr(fn, 'grad_fn') and hasattr(fn.grad_fn, 'next_functions'):
        next_fns = fn.grad_fn.next_functions

    if next_fns is not None:
        for next_fn, _ in next_fns:
            print_backward_graph(next_fn, indent + 4, seen)


def print_gpu_objects():
    import gc

    tensors = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                t = obj if torch.is_tensor(obj) else obj.data
                if t.is_cuda:
                    size_mb = t.numel() * t.element_size() / 1024**2
                    tensors.append((size_mb, type(t), tuple(t.size()), t.dtype, t.device))
        except Exception:
            pass

    # Sort by memory usage (descending)
    tensors.sort(key=lambda x: x[0], reverse=True)

    for size_mb, typ, shape, dtype, device in tensors:
        print(f"{typ.__name__:<20} {shape} {dtype} {device} {size_mb:.2f} MB, id: {id(typ)}")

    print(f"\nTotal: {sum(t[0] for t in tensors):.2f} MB")

    total_tensor_mem = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                t = obj.data
                if t.is_cuda:
                    total_tensor_mem += t.nelement() * t.element_size()
        except Exception:
            pass

    print(f"Sum of CUDA tensor memory (ideal, no fragmentation): {total_tensor_mem/1024**2:.2f} MB")

    # 2. What PyTorch has actually allocated on GPU
    allocated = torch.cuda.memory_allocated()
    reserved  = torch.cuda.memory_reserved()

    print(f"PyTorch allocated memory: {allocated/1024**2:.2f} MB")
    print(f"PyTorch reserved memory: {reserved/1024**2:.2f} MB")

def print_gpu_objects_unique():
    import gc
    seen_ids = set()
    unique_tensors = []

    for obj in gc.get_objects():
        try:
            # Check if object is a tensor
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                t = obj if torch.is_tensor(obj) else obj.data
                tensor_id = id(t)
                if tensor_id not in seen_ids:
                    seen_ids.add(tensor_id)
                    size_mb = t.numel() * t.element_size() / 1024**2
                    unique_tensors.append((size_mb, type(t), tuple(t.size()), t.dtype, t.device, tensor_id))
        except Exception:
            pass

    # Sort by memory usage (descending)
    unique_tensors.sort(key=lambda x: x[0], reverse=True)
    shapes_dict = {}

    # Print results
    total_tensor_mem = 0
    for size_mb, typ, shape, dtype, device, tid in unique_tensors:
        print(f"{typ.__name__:<20} {shape} {dtype} {device} {size_mb:.2f} MB, id: {tid}")
        total_tensor_mem += size_mb

        if shape not in shapes_dict:
            shapes_dict[shape] = 0.0
        shapes_dict[shape] += size_mb

    print(f"Shapes summary (unique shapes):")
    for shape, size_mb in shapes_dict.items():
        print(f"  {shape}: {size_mb:.2f} MB")

    print(f"\nTotal unique CUDA tensor memory (approx.): {total_tensor_mem:.2f} MB")
    print(f"PyTorch allocated memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"PyTorch reserved memory: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

def training(dataset, opt, pipe, checkpoint, num_images):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
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
    stride = 1 # np.random.randint(5, 12)
    # rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
    # with temp_seed(42):
    rand_index_start = np.random.randint(0, len(viewpoint_indices) - num_batch_cameras * stride)
    rand_indices = list(range(rand_index_start, rand_index_start + num_batch_cameras * stride, stride))
    print(f"\nUsing {num_batch_cameras} random cameras: {rand_indices}")
    loss_functions = []
    # Same background for all cameras in the batch
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    # gaussians.save_ply(os.path.join(scene.model_path, "gaussians.ply"))
    viewpoint_cams = []
    scalar_losses = []
    vector_losses = []
    for rand_idx in rand_indices:
        viewpoint_cam = viewpoint_stack[rand_idx]
        viewpoint_cams.append(viewpoint_cam)

    print_gpu_objects_unique()

    import code; code.interact(local=locals(), banner="before batch loss")



    batch_loss = batch_training_loss(iteration=iteration, opt=opt, viewpoint_cams=viewpoint_cams, gaussians=gaussians, pipe=pipe, bg=bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight)

    print_gpu_objects_unique()

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
