import math
import torch
import torch.autograd.forward_ad as fwAD
from solver.gaussian_model_state import GaussianModelState, GaussianModelParamGroupMask, GaussianModelSplatMask
from solver.loss_image_state import MultiBatchLossImageState
from solver.solver_utils import CamProvider
from utils.general_utils import safe_interact, print_gpu_objects_unique
import time
import gc

def print_backwards_graph(gfn, indent=0):
    print(" " * indent, gfn)
    if hasattr(gfn, 'next_functions'):
        for u in gfn.next_functions:
            if u[0] is not None:
                print_backwards_graph(u[0], indent + 2)


class LinearSolverFunctions:

    def __init__(self, loss_func, gaussians, batch_size=-1, param_mask=None, splat_mask=None, damp=None):
        """
        batch_size: The number of cameras that can be run in a single batch, this number should be independent of sample size

        When performing an operation on a subset of cameras, the results will be scaled appropriately to give an unbiased estimate of the full loss/Hessian.

        Damping:
        Damping will be added to the VJP and HV operations but not the JVP operation.

        """

        self.loss_func = loss_func
        self.gaussians = gaussians
        self.batch_size = batch_size
        self.batch_stats = {}
        self.param_mask = param_mask
        self.splat_mask = splat_mask
        self.damp = damp



    def get_initial_solution(self):
        s0 = GaussianModelState.zero_like_gaussians(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)
        return s0

    def evaluate_loss(self, viewpoint_cams, scale, with_batch_stats=False):
        """
        Evaluate the loss functions on the current Gaussian model state.
        scale: A scaling factor to apply to the loss to get an unbiased estimate of the full loss.
        This should be set to the inverse probability of sampling the cameras in viewpoint_cams, generally scale = total_num_cameras / num_sampled_cameras
        """
        batch_stats = {} if with_batch_stats else None
        B = len(viewpoint_cams)
        batch_size = self.batch_size if self.batch_size > 0 else B
        loss_scalar = 0.0
        Ll1_scalar = 0.0
        Ll1depth_scalar = 0.0

        with torch.no_grad():
            for start_idx in range(0, B, batch_size):
                end_idx = min(start_idx + batch_size, B)
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch, batch_stats=batch_stats)
                loss = loss * scale

                loss_scalar += loss.loss_scalar
                Ll1_scalar += loss.Ll1_scalar
                Ll1depth_scalar += loss.Ll1depth_scalar

                if with_batch_stats:
                    for key, value in batch_stats.items():
                        if start_idx == 0:
                            self.batch_stats[key] = [value]
                        else:
                            self.batch_stats[key].append(value)

                del loss
                gc.collect()
                torch.cuda.empty_cache()

        return loss_scalar, Ll1_scalar, Ll1depth_scalar

    @property
    def get_batch_stats(self):
        """
        Return the batch statistics.
        """
        return self.batch_stats

    def jvp(self, v, viewpoint_cams):
        """
        Damping and scaling are not done in JVP
        """

        assert isinstance(v, GaussianModelState), "v must be an instance of GaussianModelState"
        B = len(viewpoint_cams)
        batch_size = self.batch_size if self.batch_size > 0 else B

        for start_idx in range(0, B, batch_size):
            loss_tangents = []
            with torch.no_grad(), fwAD.dual_level(), self.gaussians.make_dual(v):
                end_idx = min(start_idx + batch_size, B)
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss_dual = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)
                loss_primal, loss_tangent = loss_dual.unpack_dual()
                loss_tangents.append(loss_tangent)

                del loss_primal, loss_dual

            loss_tangents = MultiBatchLossImageState(loss_tangents)

            return loss_tangents

    def vjp(self, vs, viewpoint_cams, scale):
        """
        Damping and scaling are not done in VJP
        """

        # Assume vs is a MultiBatchLossImageState with the same batch size and order as the original viewpoint_cams
        assert isinstance(vs, MultiBatchLossImageState), "vs must be an instance of MultiBatchLossImageState"
        B = len(viewpoint_cams)
        batch_size = batch_size if batch_size > 0 else B

        for start_idx in range(0, B, batch_size):
            self.gaussians.zero_grad()

            idx = 0

            with torch.enable_grad():
                end_idx = min(start_idx + batch_size, B)
                vs_batch = vs.batch_losses[idx]

                # Forward pass
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss_batch = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)

                # Backward pass
                loss_batch.backward(vs_batch, retain_graph=False)

                idx += 1

                del loss_batch

        assert not torch.isnan(self.gaussians._xyz.grad).any(), "NaN detected in gaussians._xyz.grad"
        assert not torch.isnan(self.gaussians._features_dc.grad).any(), "NaN detected in gaussians._features_dc.grad"
        assert not torch.isnan(self.gaussians._features_rest.grad).any(), "NaN detected in gaussians._features_rest.grad"
        assert not torch.isnan(self.gaussians._scaling.grad).any(), "NaN detected in gaussians._scaling.grad"
        assert not torch.isnan(self.gaussians._rotation.grad).any(), "NaN detected in gaussians._rotation.grad"
        assert not torch.isnan(self.gaussians._opacity.grad).any(), "NaN detected in gaussians._opacity.grad"

        return GaussianModelState.from_gaussians_grad(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)


    def gradient_and_loss_est(self, viewpoint_cams, scale):
        """
        viewpoint_cams is a subset of all cameras, scale should be set to the inverse probability of sampling those cameras
        """

        self.gaussians.zero_grad()

        B = len(viewpoint_cams)
        batch_size = self.batch_size if self.batch_size > 0 else B
        loss_scalar = 0.0
        Ll1_scalar = 0.0
        Ll1depth_scalar = 0.0

        with torch.enable_grad():
            for start_idx in range(0, B, batch_size):
                end_idx = min(start_idx + batch_size, B)
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)

                loss = loss * scale

                loss_scalar += loss.loss_scalar
                Ll1_scalar += loss.Ll1_scalar
                Ll1depth_scalar += loss.Ll1depth_scalar

                loss.loss_scalar.backward(retain_graph=False)

                del loss
                gc.collect()
                torch.cuda.empty_cache()

        grad = GaussianModelState.from_gaussians_grad(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)

        return grad, loss_scalar

    def Hv(self, v, viewpoint_cams, scale):
        """
        Compute 1 forward and backward pass to get the Hessian-vector product Hv of viewpoint_cams
        scale is a scaling factor to apply to the loss to get an unbiased estimate of the full loss.
        Damping and scaling are applied to Hv
        """
        self.gaussians.zero_grad()

        B = len(viewpoint_cams)
        batch_size = self.batch_size if self.batch_size > 0 else B

        for start_idx in range(0, B, batch_size):
            with torch.enable_grad(), fwAD.dual_level(), self.gaussians.make_dual(v):
                end_idx = min(start_idx + batch_size, B)
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss_dual = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)
                loss_primal, loss_tangent = loss_dual.unpack_dual()
                loss_primal.backward(loss_tangent, retain_graph=False)

                del loss_primal, loss_dual, loss_tangent
                gc.collect()
                torch.cuda.empty_cache()

        Hv = GaussianModelState.from_gaussians_grad(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask) * scale
        if self.damp is not None:
            Hv += self.damp * v

        return Hv


    def dot(self, v1, v2, damp=1.0):
        return v1.dot(v2, damp)

    def saxpy(self, a, x, y):
        return a * x + y

