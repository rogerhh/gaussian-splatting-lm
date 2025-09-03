import math
import torch
import torch.autograd.forward_ad as fwAD
from solver.gaussian_model_state import GaussianModelState, GaussianModelParamGroupMask, GaussianModelSplatMask
from solver.loss_image_state import MultiBatchLossImageState
import time

def print_backwards_graph(gfn, indent=0):
    print(" " * indent, gfn)
    if hasattr(gfn, 'next_functions'):
        for u in gfn.next_functions:
            if u[0] is not None:
                print_backwards_graph(u[0], indent + 2)


class LinearSolverFunctions:

    def __init__(self, loss_func, gaussians, batch_size=-1, param_mask=None, splat_mask=None, damp=None):
        self.loss_func = loss_func
        self.gaussians = gaussians
        self.batch_stats = {}
        self.param_mask = param_mask
        self.splat_mask = splat_mask
        self.damp = damp

        self.loss_scalar = None
        self.Ll1_scalar = None
        self.Ll1depth_scalar = None


    def get_initial_solution(self):
        s0 = GaussianModelState.zero_like_gaussians(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)
        return s0

    def evaluate_loss(self, viewpoint_cams, batch_size=-1, with_batch_stats=False, with_grad=False):
        """
        Evaluate the loss functions on the current Gaussian model state.
        """
        batch_stats = {} if with_batch_stats else None
        B = len(viewpoint_cams)
        batch_size = batch_size if batch_size > 0 else B
        with torch.enable_grad() if with_grad else torch.no_grad():
            losses = []
            for start_idx in range(0, B, batch_size):
                end_idx = min(start_idx + batch_size, B)
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch, batch_stats=batch_stats)
                losses.append(loss)

                if with_batch_stats:
                    for key, value in batch_stats.items():
                        if start_idx == 0:
                            self.batch_stats[key] = [value]
                        else:
                            self.batch_stats[key].append(value)

            loss = MultiBatchLossImageState(losses)

            self.Ll1_scalar = loss.Ll1_scalar
            self.Ll1depth_scalar = loss.Ll1depth_scalar
            self.loss_scalar = loss.loss_scalar

        return loss

    @property
    def get_batch_stats(self):
        """
        Return the batch statistics.
        """
        return self.batch_stats

    def jvp(self, v, viewpoint_cams, batch_size=-1):
        raise NotImplementedError("not working with damp")

        assert isinstance(v, GaussianModelState), "v must be an instance of GaussianModelState"
        B = len(viewpoint_cams)
        batch_size = batch_size if batch_size > 0 else B

        with torch.no_grad(), fwAD.dual_level(), self.gaussians.make_dual(v):
            loss_tangents = []
            for start_idx in range(0, B, batch_size):
                end_idx = min(start_idx + batch_size, B)
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss_dual = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)
                loss_primal, loss_tangent = loss_dual.unpack_dual()
                loss_tangents.append(loss_tangent)

                del loss_primal, loss_dual

            loss_tangents = MultiBatchLossImageState(loss_tangents)

            return loss_tangents

    def vjp(self, vs, viewpoint_cams, batch_size=-1):
        raise NotImplementedError("not working with damp")

        # Assume vs is a MultiBatchLossImageState with the same batch size and order as the original viewpoint_cams
        assert isinstance(vs, MultiBatchLossImageState), "vs must be an instance of MultiBatchLossImageState"
        B = len(viewpoint_cams)
        batch_size = batch_size if batch_size > 0 else B

        with torch.enable_grad():
            self.gaussians.zero_grad()

            idx = 0

            for start_idx in range(0, B, batch_size):
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

    def initial_evalulation(self, viewpoint_cams, batch_size=-1):
        """
        Run one forward pass to get loss value and one backward pass to get initial gradient.
        """
        grad = self.gradient(viewpoint_cams, batch_size=-1)
        return self.loss_scalar, grad

    def gradient(self, viewpoint_cams, batch_size=-1):
        self.evaluate_loss(viewpoint_cams, batch_size=batch_size, with_grad=True)
        self.gaussians.zero_grad()
        self.loss_scalar.backward(retain_graph=False)

        grad = GaussianModelState.from_gaussians_grad(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)

        return grad

    def Hv(self, v, viewpoint_cams, batch_size=-1):
        """
        Compute 1 forward and backward pass to get the Hessian-vector product Hv.
        """
        self.gaussians.zero_grad()

        B = len(viewpoint_cams)
        batch_size = batch_size if batch_size > 0 else B

        with torch.enable_grad(), fwAD.dual_level(), self.gaussians.make_dual(v):
            for start_idx in range(0, B, batch_size):
                end_idx = min(start_idx + batch_size, B)
                viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss_dual = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)
                loss_primal, loss_tangent = loss_dual.unpack_dual()
                loss_primal.backward(loss_tangent, retain_graph=False)

                del loss_primal, loss_dual, loss_tangent

        Hv = (self.damp + 1) * GaussianModelState.from_gaussians_grad(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)

        return Hv

    def rand_batch_Hv_and_gradient(self, v, cam_provider, batch_size=-1):
        """
        Compute one random batch Hessian-vector product and the full gradient.
        """

        viewpoint_cams_batch = next(cam_provider)

        grad = self.gradient(viewpoint_cams_batch, batch_size=batch_size)
        Hv = self.Hv(v, viewpoint_cams_batch, batch_size=batch_size)

        return Hv, grad


    def dot(self, v1, v2, damp=1.0):
        return v1.dot(v2, damp)

    def saxpy(self, a, x, y):
        return a * x + y

