import math
import torch
import torch.autograd.forward_ad as fwAD
from solver.gaussian_model_state import GaussianModelState, GaussianModelParamGroupMask, GaussianModelSplatMask
from solver.loss_image_state import MultiBatchLossImageState

def print_backwards_graph(gfn, indent=0):
    print(" " * indent, gfn)
    if hasattr(gfn, 'next_functions'):
        for u in gfn.next_functions:
            if u[0] is not None:
                print_backwards_graph(u[0], indent + 2)


class LinearSolverFunctions:

    def __init__(self, loss_func, gaussians, viewpoint_cams, batch_size=-1, param_mask=None, splat_mask=None):
        self.loss_func = loss_func
        self.gaussians = gaussians
        self.viewpoint_cams = viewpoint_cams
        self.B = len(viewpoint_cams)
        self.batch_size = self.B if batch_size < 0 else batch_size
        self.batch_stats = {}
        self.param_mask = param_mask
        self.splat_mask = splat_mask

    def get_initial_solution(self):
        s0 = GaussianModelState.from_gaussians(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)
        return s0

    def evaluate_loss(self, with_batch_stats=False, with_grad=False):
        """
        Evaluate the loss functions on the current Gaussian model state.
        """
        batch_stats = {} if with_batch_stats else None
        with torch.enable_grad() if with_grad else torch.no_grad():
            losses = []
            for start_idx in range(0, self.B, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.B)
                viewpoint_cams_batch = [self.viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch, batch_stats=batch_stats)
                losses.append(loss)

                if with_batch_stats:
                    for key, value in batch_stats.items():
                        if start_idx == 0:
                            self.batch_stats[key] = [value]
                        else:
                            self.batch_stats[key].append(value)

            self.loss = MultiBatchLossImageState(losses)

        return self.loss

    @property
    def loss_scalar(self):
        """
        Return the scalar value of the loss.
        """
        return self.loss.loss_scalar

    @property
    def Ll1_scalar(self):
        """
        Return the scalar value of the L1 loss.
        """
        return self.loss.Ll1_scalar

    @property
    def depth_loss_scalar(self):
        """
        Return the scalar value of the depth loss.
        """
        return self.loss.Ll1depth_scalar

    @property
    def get_batch_stats(self):
        """
        Return the batch statistics.
        """
        return self.batch_stats

    def matvec(self, v):
        assert isinstance(v, GaussianModelState), "v must be an instance of GaussianModelState"

        with torch.no_grad(), fwAD.dual_level(), self.gaussians.make_dual(v):
            loss_tangents = []
            for start_idx in range(0, self.B, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.B)
                viewpoint_cams_batch = [self.viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss_dual = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)
                loss_primal, loss_tangent = loss_dual.unpack_dual()
                loss_tangents.append(loss_tangent)

                del loss_primal, loss_dual

            loss_tangents = MultiBatchLossImageState(loss_tangents)

            return loss_tangents

    def matvec_T(self, vs):
        # Assume vs is a MultiBatchLossImageState with the same batch size and order as the original viewpoint_cams
        assert isinstance(vs, MultiBatchLossImageState), "vs must be an instance of MultiBatchLossImageState"

        with torch.enable_grad():
            self.gaussians.zero_grad()

            idx = 0

            for start_idx in range(0, self.B, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.B)
                vs_batch = vs.batch_losses[idx]

                # Forward pass
                viewpoint_cams_batch = [self.viewpoint_cams[i] for i in range(start_idx, end_idx)]
                loss_batch = self.loss_func(gaussians=self.gaussians, viewpoint_cams=viewpoint_cams_batch)

                # Backward pass
                loss_batch.backward(vs_batch, retain_graph=True)

                idx += 1

                del loss_batch

        assert not torch.isnan(self.gaussians._xyz.grad).any(), "NaN detected in gaussians._xyz.grad"
        assert not torch.isnan(self.gaussians._features_dc.grad).any(), "NaN detected in gaussians._features_dc.grad"
        assert not torch.isnan(self.gaussians._features_rest.grad).any(), "NaN detected in gaussians._features_rest.grad"
        assert not torch.isnan(self.gaussians._scaling.grad).any(), "NaN detected in gaussians._scaling.grad"
        assert not torch.isnan(self.gaussians._rotation.grad).any(), "NaN detected in gaussians._rotation.grad"
        assert not torch.isnan(self.gaussians._opacity.grad).any(), "NaN detected in gaussians._opacity.grad"

        return GaussianModelState.from_gaussians_grad(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)

    def dot(self, v1, v2, damp=1.0):
        return v1.dot(v2, damp)

    def saxpy(self, a, x, y):
        return a * x + y

