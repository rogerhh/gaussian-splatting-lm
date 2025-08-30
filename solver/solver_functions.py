import math
import torch
import torch.autograd.forward_ad as fwAD
from solver.gaussian_model_state import GaussianModelState, GaussianModelParamGroupMask, GaussianModelSplatMask
from solver.loss_image_state import MultiLossImageState, GroupedMultiLossImageState

def print_backwards_graph(gfn, indent=0):
    print(" " * indent, gfn)
    if hasattr(gfn, 'next_functions'):
        for u in gfn.next_functions:
            if u[0] is not None:
                print_backwards_graph(u[0], indent + 2)


class LinearSolverFunctions:

    def __init__(self, gaussians, param_mask=None, splat_mask=None):
        self.gaussians = gaussians
        self.param_mask = param_mask
        self.splat_mask = splat_mask

    def get_initial_solution(self):
        s0 = GaussianModelState.from_gaussians(self.gaussians, param_mask=self.param_mask, splat_mask=self.splat_mask)
        return s0

    def set_loss_functions(self, loss_func):
        """
        Set the loss functions used in the solver.
        """
        self.batch_stats = {}
        self.loss_func = loss_func
        # This is needed because later we will run loss_functions with no_grad
        self.evaluate_loss(with_batch_stats=True)

    def evaluate_loss(self, with_batch_stats=False):
        """
        Evaluate the loss functions on the current Gaussian model state.
        """
        batch_stats = self.batch_stats if with_batch_stats else None
        self.loss = self.loss_func(gaussians=self.gaussians, batch_stats=batch_stats)
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
            loss_dual = self.loss_func(gaussians=self.gaussians)
            loss_primal, loss_tangent = loss_dual.unpack_dual()

            return loss_tangent

    def matvec_T(self, vs):
        assert isinstance(vs, MultiLossImageState) or isinstance(vs, GroupedMultiLossImageState), "vs must be an instance of MultiLossImageState or GroupedMultiLossImageState"

        with torch.enable_grad():
            self.gaussians.zero_grad()
            self.loss.backward(vs, retain_graph=True)

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

