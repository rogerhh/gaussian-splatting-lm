import torch
import torch.autograd.forward_ad as fwAD

import glob
import os
from functools import partial

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

v = cur_state.loss.zero_like()
v_vec = v.as_1d_tensor()

num_trials = 1000

for trial in range(num_trials):

    rand_row_idx = torch.randint(0, v_vec.shape[0], (1,)).item()
    v_vec[rand_row_idx] = 1
    v.load_1d_tensor(v_vec)
    print("rand row index = ", rand_row_idx, v.index_to_desc(rand_row_idx))

    Jtv = cur_state.vjp(v).as_1d_tensor()
    v_vec[rand_row_idx] = 0

    P = gaussians.get_xyz.shape[0]
    params_per_gaussian = [3, 3, 45, 3, 4, 1]
    exposure_numel = gaussians._exposure.numel()

    rand_gid = torch.randint(0, P, (1,)).item()
    print("rand_gid = ", rand_gid)

    u = GaussianModelState.zero_like_gaussians(gaussians)
    u_vec = u.as_1d_tensor()

    with torch.no_grad():
        offset = 0
        for width in params_per_gaussian:
            for i in range(width):
                rand_col_idx = offset + rand_gid * width + i
                u_vec[rand_col_idx] = 1
                u.load_1d_tensor(u_vec)
                print("rand col index = ", rand_col_idx, u.index_to_desc(rand_col_idx))
                Ju = cur_state.jvp(u).as_1d_tensor()
                u_vec[rand_col_idx] = 0
                rel_error = compute_rel_error(Jtv[rand_col_idx], Ju[rand_row_idx])
                print(f"fwd {Jtv[rand_col_idx]:.4e} vs J(u)[{rand_row_idx}]: {Ju[rand_row_idx]:.4e}, rel error: {rel_error:.4e}")
                if rel_error > 1e-4:
                    print(f"Significant error detected: {rel_error:.4e}")
                    # Debugging interaction
                    import code; code.interact(local=locals(), banner="Debugging matvec")
            offset += P * width



