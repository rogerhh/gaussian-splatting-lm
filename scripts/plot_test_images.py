#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, safe_interact, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from functools import partial
from scene.gaussian_model import build_scaling_rotation
from solver.gaussian_model_vector import GaussianModelVector
from solver.adam_optimizer import AdamOptimizer
from solver.sophia_optimizer import SophiaOptimizer
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy, construct_Dhat_func
from solver.hellinger_clip import clip_hellinger, debug_hellinger

import re
import glob
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, savedir, prefix):

    ####### Some fixed parameters #########
    train_test_exp = False
    ####### Some fixed parameters #########

    testing_iterations = testing_iterations + list(range(0, opt.iterations + 1, opt.eval_interval))

    datasets = ["train", "bicycle", "playroom", "room", "truck"]
    dataset_paths = {"bicycle": "../datasets/bicycle/",
                     "playroom": "../datasets/playroom/",
                     "room": "../datasets/room/",
                     "train": "../datasets/tandt/train/",
                     "truck": "../datasets/tandt/truck/"}

    methods = ["ADAM", "Ours"]

    method_names = {"ADAM": "mcmc_no_densify",
                    "Ours": "sophia_hellinger_no_densify_update10"}

    for dataset_name in datasets:
        images = {}

        args.source_path = dataset_paths[dataset_name]
        dataset = lp.extract(args)

        first_iter = 0
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)

        for test_camera in scene.getTestCameras():
            image_name = test_camera.image_name.replace(".png", "").replace(".jpg", "")
            gt_image = test_camera.original_image.cuda(dataset.data_device)
            images[image_name] = {}
            images[image_name]["GT"] = torch.clamp(gt_image, 0.0, 1.0)

        for method in methods:
            checkpoint_pattern = f"../icml2026*/eval_selected/{dataset_name}/{method_names[method]}/chkpnt*.pth"
            checkpoint = sorted(glob.glob(checkpoint_pattern))[-1]

            print(f"Loading checkpoint for {dataset_name} with method {method} from {checkpoint}")

            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
            depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

            iter_start = torch.cuda.Event(enable_timing = True)
            iter_end = torch.cuda.Event(enable_timing = True)

            viewpoint_stack = None
            ema_loss_for_log = 0.0
            progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
            first_iter += 1

            bg = background

            with torch.no_grad():
                for test_camera in scene.getTestCameras():
                    test_image = render(test_camera, gaussians, pipe, background)["render"]
                    image_name = test_camera.image_name.replace(".png", "").replace(".jpg", "")

                    images[image_name][method] = torch.clamp(test_image, 0.0, 1.0)

        print(f"Saved initial test view {test_camera.image_name}")
        for test_camera in scene.getTestCameras():
            for method in methods + ["GT"]:
                image_name = test_camera.image_name.replace(".png", "").replace(".jpg", "")
                test_image = images[image_name][method]

                filepath = f"{savedir}_test_images/{method}_{image_name}.png"
                torchvision.utils.save_image(torch.clamp(test_image, 0.0, 1.0), f"{filepath}.png")

        exit()


    exit()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        test_psnrs = {}

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if config['name'] == 'test':
                        image_idx = int(re.findall(r'\d+', viewpoint.image_name)[0])
                        test_psnrs[image_idx] = round(psnr(image, gt_image).mean().item(), 4)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        print(f"[ITER {iteration}] test PSNR: {test_psnrs}")

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--savedir", type=str, default = "./figures/")
    parser.add_argument("--prefix", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)

    args.eval = True
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.savedir, args.prefix)

    # All done
    print("\nTraining complete.")
