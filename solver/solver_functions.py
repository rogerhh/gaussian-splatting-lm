import math
import torch
import torch.autograd.forward_ad as fwAD
from solver.gaussian_model_state import GaussianModelState
from solver.loss_image_state import MultiLossImageState

class MultiLoss:
    """
    Aggregates multiple loss functions into one
    """
    def __init__(self, loss_functions):
        self.loss_functions = loss_functions

    def __call__(self, gaussians, batch_stats=None):
        """
        Compute the aggregated loss from the Gaussian model state.
        """
        return MultiLossImageState([loss_function(gaussians=gaussians, 
                                                  batch_stats=batch_stats) 
                                    for loss_function in self.loss_functions])

def print_backwards_graph(gfn, indent=0):
    print(" " * indent, gfn)
    if hasattr(gfn, 'next_functions'):
        for u in gfn.next_functions:
            if u[0] is not None:
                print_backwards_graph(u[0], indent + 2)


class LinearSolverFunctions:

    def __init__(self, gaussians):
        self.gaussians = gaussians

    def set_loss_functions(self, loss_functions):
        """
        Set the loss functions used in the solver.
        """
        visibility_mask = torch.zeros(self.gaussians.get_xyz.shape[0], 
                                                 dtype=torch.bool, 
                                                 device=self.gaussians.get_xyz.device)
        max_radii2D = torch.zeros(self.gaussians.get_xyz.shape[0], 
                                       dtype=torch.int, 
                                       device=self.gaussians.get_xyz.device)
        self.batch_stats = {"visibility_mask": visibility_mask,
                            "max_radii2D": max_radii2D}
        self.multiloss = MultiLoss(loss_functions)
        # This is needed because later we will run loss_functions with no_grad
        self.evaluate_loss(accum_stats=True)

    def evaluate_loss(self, accum_stats=False):
        """
        Evaluate the loss functions on the current Gaussian model state.
        """
        batch_stats = self.batch_stats if accum_stats else None
        self.loss = self.multiloss(self.gaussians, batch_stats=batch_stats)
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

        with fwAD.dual_level(), self.gaussians.make_dual(v):
            loss_dual = self.multiloss(self.gaussians)
            loss_primal, loss_tangent = loss_dual.unpack_dual()

            return loss_tangent

    def matvec_T(self, vs):
        assert isinstance(vs, MultiLossImageState), "vs must be an instance of MultiLossImageState"

        self.gaussians.zero_grad()
        self.loss.backward(vs, retain_graph=True)

        assert not torch.isnan(self.gaussians._xyz.grad).any(), "NaN detected in gaussians._xyz.grad"
        assert not torch.isnan(self.gaussians._features_dc.grad).any(), "NaN detected in gaussians._features_dc.grad"
        assert not torch.isnan(self.gaussians._features_rest.grad).any(), "NaN detected in gaussians._features_rest.grad"
        assert not torch.isnan(self.gaussians._scaling.grad).any(), "NaN detected in gaussians._scaling.grad"
        assert not torch.isnan(self.gaussians._rotation.grad).any(), "NaN detected in gaussians._rotation.grad"
        assert not torch.isnan(self.gaussians._opacity.grad).any(), "NaN detected in gaussians._opacity.grad"

        return GaussianModelState.from_gaussians_grad(self.gaussians)

    def dot(self, v1, v2):
        return v1.dot(v2)

    def saxpy(self, a, x, y):
        return a * x + y

