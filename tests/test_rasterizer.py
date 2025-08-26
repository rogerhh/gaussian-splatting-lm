import torch
import torch.autograd.forward_ad as fwAD
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_orig import GaussianRasterizationSettings as GaussianRasterizationSettingsOrig, GaussianRasterizer as GaussianRasterizerOrig

load_dict = torch.load("rasterizer_debug.pth")
rasterizer, rasterizer_orig, means3D, means2D, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp = \
    load_dict["rasterizer"], load_dict["rasterizer_orig"], load_dict["means3D"], \
    load_dict["means2D"], load_dict["shs"], load_dict["colors_precomp"], load_dict["opacities"], \
    load_dict["scales"], load_dict["rotations"], load_dict["cov3D_precomp"]

rendered_image, radii, depth_image = rasterizer(
    means3D = means3D,
    means2D = means2D,
    shs = shs,
    colors_precomp = colors_precomp,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3D_precomp = cov3D_precomp)

means3D_grad = torch.zeros_like(means3D)
with fwAD.dual_level():
    means3D_dual = fwAD.make_dual(means3D, means3D_grad)
    rendered_image_dual, radii_dual, depth_image_dual = rasterizer(
        means3D = means3D_dual,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

rendered_image_ref, radii_ref, depth_image_ref = rasterizer_orig(
    means3D = means3D,
    means2D = means2D,
    shs = shs,
    colors_precomp = colors_precomp,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3D_precomp = cov3D_precomp)

import code; code.interact(local=locals(), banner="Debugging rasterizer")
