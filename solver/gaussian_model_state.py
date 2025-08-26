import torch
import torch.autograd.forward_ad as fwAD

class GaussianModelState:
    """
    Represents updates to Gaussian parameters as a generalized vector
    """

    def __init__(self, xyz_grad, features_dc_grad, features_rest_grad,
                 scaling_grad, rotation_grad, opacity_grad, exposure_grad):
        self.xyz_grad = xyz_grad
        self.features_dc_grad = features_dc_grad
        self.features_rest_grad = features_rest_grad
        self.scaling_grad = scaling_grad
        self.rotation_grad = rotation_grad
        self.opacity_grad = opacity_grad
        self.exposure_grad = exposure_grad

    @classmethod
    def from_gaussians(cls, gaussians):
        return cls(torch.zeros_like(gaussians._xyz),
                   torch.zeros_like(gaussians._features_dc),
                   torch.zeros_like(gaussians._features_rest),
                   torch.zeros_like(gaussians._scaling),
                   torch.zeros_like(gaussians._rotation),
                   torch.zeros_like(gaussians._opacity),
                   torch.zeros_like(gaussians._exposure))

    @classmethod
    def from_gaussians_grad(cls, gaussians):
        gaussians._xyz.grad = gaussians._xyz.grad if gaussians._xyz.grad is not None else torch.zeros_like(gaussians._xyz)
        gaussians._features_dc.grad = gaussians._features_dc.grad if gaussians._features_dc.grad is not None else torch.zeros_like(gaussians._features_dc)
        gaussians._features_rest.grad = gaussians._features_rest.grad if gaussians._features_rest.grad is not None else torch.zeros_like(gaussians._features_rest)
        gaussians._scaling.grad = gaussians._scaling.grad if gaussians._scaling.grad is not None else torch.zeros_like(gaussians._scaling)
        gaussians._rotation.grad = gaussians._rotation.grad if gaussians._rotation.grad is not None else torch.zeros_like(gaussians._rotation)
        gaussians._opacity.grad = gaussians._opacity.grad if gaussians._opacity.grad is not None else torch.zeros_like(gaussians._opacity)
        gaussians._exposure.grad = gaussians._exposure.grad if gaussians._exposure.grad is not None else torch.zeros_like(gaussians._exposure)
        return cls(gaussians._xyz.grad,
                   gaussians._features_dc.grad,
                   gaussians._features_rest.grad,
                   gaussians._scaling.grad,
                   gaussians._rotation.grad,
                   gaussians._opacity.grad,
                   gaussians._exposure.grad)

    @property
    def length(self):
        """
        Returns the length of the generalized vector
        """
        return (self.xyz_grad.numel() + self.features_dc_grad.numel() +
                self.features_rest_grad.numel() + self.scaling_grad.numel() +
                self.rotation_grad.numel() + self.opacity_grad.numel() +
                self.exposure_grad.numel())

    def load_1d_tensor(self, T):
        """
        Creates a GaussianModelState from a flattened tensor
        """
        assert T.shape[0] == self.length, "Input tensor must match the length of the model state"
        N1 = self.xyz_grad.numel()
        N2 = self.features_dc_grad.numel()
        N3 = self.features_rest_grad.numel()
        N4 = self.scaling_grad.numel()
        N5 = self.rotation_grad.numel()
        N6 = self.opacity_grad.numel()
        N7 = self.exposure_grad.numel()

        offset = 0
        xyz_grad = T[offset:offset + N1].view(self.xyz_grad.shape)
        offset += N1
        features_dc_grad = T[offset:offset + N2].view(self.features_dc_grad.shape)
        offset += N2
        features_rest_grad = T[offset:offset + N3].view(self.features_rest_grad.shape)
        offset += N3
        scaling_grad = T[offset:offset + N4].view(self.scaling_grad.shape)
        offset += N4
        rotation_grad = T[offset:offset + N5].view(self.rotation_grad.shape)
        offset += N5
        opacity_grad = T[offset:offset + N6].view(self.opacity_grad.shape)
        offset += N6
        exposure_grad = T[offset:offset + N7].view(self.exposure_grad.shape)

        self.__init__(xyz_grad, features_dc_grad, features_rest_grad,
                      scaling_grad, rotation_grad,
                      opacity_grad, exposure_grad)

    def as_1d_tensor(self):
        """
        Returns the model state as a flattened vector
        """
        return torch.cat((self.xyz_grad.flatten(),
                          self.features_dc_grad.flatten(),
                          self.features_rest_grad.flatten(),
                          self.scaling_grad.flatten(),
                          self.rotation_grad.flatten(),
                          self.opacity_grad.flatten(),
                          self.exposure_grad.flatten()), dim=0)

    def index_to_desc(self, index):
        N1 = self.xyz_grad.numel()
        N2 = self.features_dc_grad.numel()
        N3 = self.features_rest_grad.numel()
        N4 = self.scaling_grad.numel()
        N5 = self.rotation_grad.numel()
        N6 = self.opacity_grad.numel()
        N7 = self.exposure_grad.numel()

        assert index < self.length, "Index out of bounds for GaussianModelState"

        def find_coord(offset, shape):
            coords = []
            for dim in reversed(shape):
                coords.append(offset % dim)
                offset //= dim
            return tuple(reversed(coords))

        name_offsets = [(N1, "xyz_grad"), (N2, "features_dc_grad"), (N3, "features_rest_grad"),
                        (N4, "scaling_grad"), (N5, "rotation_grad"),
                        (N6, "opacity_grad"), (N7, "exposure_grad")]
        
        for l, name in name_offsets:
            if index < l:
                offset = index
                return name, find_coord(offset, getattr(self, name).shape)
            index -= l


    def __add__(self, other):
        return GaussianModelState(
            self.xyz_grad + other.xyz_grad,
            self.features_dc_grad + other.features_dc_grad,
            self.features_rest_grad + other.features_rest_grad,
            self.scaling_grad + other.scaling_grad,
            self.rotation_grad + other.rotation_grad,
            self.opacity_grad + other.opacity_grad,
            self.exposure_grad + other.exposure_grad
        )

    def __sub__(self, other):
        return GaussianModelState(
            self.xyz_grad - other.xyz_grad,
            self.features_dc_grad - other.features_dc_grad,
            self.features_rest_grad - other.features_rest_grad,
            self.scaling_grad - other.scaling_grad,
            self.rotation_grad - other.rotation_grad,
            self.opacity_grad - other.opacity_grad,
            self.exposure_grad - other.exposure_grad
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelState(
                self.xyz_grad * other,
                self.features_dc_grad * other,
                self.features_rest_grad * other,
                self.scaling_grad * other,
                self.rotation_grad * other,
                self.opacity_grad * other,
                self.exposure_grad * other
            )
        else:
            raise TypeError(f"Can only multiply by scalar values, not {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other):
        s = torch.sum(self.xyz_grad * other.xyz_grad) + \
            torch.sum(self.features_dc_grad * other.features_dc_grad) + \
            torch.sum(self.features_rest_grad * other.features_rest_grad) + \
            torch.sum(self.scaling_grad * other.scaling_grad) + \
            torch.sum(self.rotation_grad * other.rotation_grad) + \
            torch.sum(self.opacity_grad * other.opacity_grad) + \
            torch.sum(self.exposure_grad * other.exposure_grad)
        return s.item()
     
