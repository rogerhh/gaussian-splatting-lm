import torch
import torch.autograd.forward_ad as fwAD

import glob
import os
from functools import partial
import time

from scene import Scene, GaussianModel
from utils.general_utils import get_expon_lr_func
from solver.training_loss import training_loss
from solver.solver_functions import LinearSolverFunctions
from solver.gaussian_model_state import GaussianModelState
from solver.loss_image_state import MultiLossImageState

def compute_rel_error(a, b):
    if a.abs() < 1e-12 and b.abs() < 1e-12:
        return 0.0
    if (a - b).abs() < 1e-12:
        return 0.0
    return (a - b).abs() / max(a.abs(), b.abs())

torch.manual_seed(42)

func_args_files = glob.glob("func_args-*.pth")

loss_functions = []

dataset = None

for func_arg_file in func_args_files:
    func_args = torch.load(func_arg_file)
    iteration, opt, viewpoint_cam, pipe, bg, dataset = func_args["iteration"], func_args["opt"], func_args["viewpoint_cam"], func_args["pipe"], func_args["bg"], func_args["dataset"]
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    loss_func = partial(training_loss, iteration=iteration, opt=opt, viewpoint_cam=viewpoint_cam, pipe=pipe, bg=bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight)
    loss_functions.append(loss_func)

gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
scene = Scene(dataset, gaussians)
gaussians.training_setup(opt)

cur_state = LinearSolverFunctions(gaussians)
cur_state.set_loss_functions(loss_functions)

u = GaussianModelState.from_gaussians(gaussians)
u_vec = u.as_1d_tensor()

NUM_ITERATIONS = 100

matvec_start = time.time()
for _ in range(NUM_ITERATIONS):
    Ju = cur_state.matvec(u).as_1d_tensor()
matvec_end = time.time()

forward_start = time.time()
for _ in range(NUM_ITERATIONS):
    cur_state.evaluate_loss()
forward_end = time.time()

print(f"Matvec time: {(matvec_end - matvec_start) * 1000 / NUM_ITERATIONS:.6f} milliseconds per iteration")
print(f"Forward time: {(forward_end - forward_start) * 1000 / NUM_ITERATIONS:.6f} milliseconds per iteration")
