#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms in ../../THIRD_PARTY.md#gaussian-splatting-license.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from easydict import EasyDict as edict
import numpy as np
from ..representations.gaussian import Gaussian
from .sh_utils import eval_sh
import torch.nn.functional as F
from easydict import EasyDict as edict


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = 2 * cy - 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def render(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor,
           scaling_modifier=1.0, override_color=None,
           return_types=("color",)):
    """
    Render the gaussian. Background tensor (bg_color) must be on GPU.

    ``return_types``: subset of ``{"color", "depth", "alpha"}``.

      * ``"color"`` uses the mip-splatting fork (``diff_gaussian_rasterization``)
        for screen-space antialiasing.
      * ``"depth"`` / ``"alpha"`` use the depth-emitting fork
        (``depth_diff_gaussian_rasterization``) since the mip-splat fork
        doesn't expose depth. Alpha needs a second rasterization pass with
        white precomp colors on a black background (premultiplied accumulation
        of white = alpha on every channel).
    """
    return_types = tuple(return_types)
    want_color = "color" in return_types
    want_depth = "depth" in return_types
    want_alpha = "alpha" in return_types

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    out = edict({})

    # ------------------------------------------------------------------
    # Color path (mip-splatting fork). Preserves screen-space antialiasing
    # via ``kernel_size`` / ``subpixel_offset`` for visualization quality.
    # ------------------------------------------------------------------
    if want_color:
        from diff_gaussian_rasterization import (
            GaussianRasterizer as _MipRasterizer,
            GaussianRasterizationSettings as _MipSettings,
        )
        kernel_size = pipe.kernel_size
        subpixel_offset = torch.zeros(
            (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
            dtype=torch.float32, device="cuda",
        )
        mip_settings = _MipSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy,
            kernel_size=kernel_size, subpixel_offset=subpixel_offset,
            bg=bg_color, scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False, debug=pipe.debug,
        )
        rendered_image, radii = _MipRasterizer(raster_settings=mip_settings)(
            means3D=means3D, means2D=means2D, shs=shs,
            colors_precomp=colors_precomp, opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
        )
        out.color = rendered_image
        out.viewspace_points = screenspace_points
        out.visibility_filter = radii > 0
        out.radii = radii

    # ------------------------------------------------------------------
    # Depth / alpha path (depth-emitting fork). Mip-splatting fork does
    # not expose depth, so a second rasterizer is used here. Depth is
    # independent of colors/bg (it's a function of positions + opacities),
    # so a single pass with white precomp on a black bg yields both:
    # color output = premultiplied accumulation of white = alpha,
    # depth output = depth.
    # ------------------------------------------------------------------
    if want_depth or want_alpha:
        from depth_diff_gaussian_rasterization import (
            GaussianRasterizer as _DepthRasterizer,
            GaussianRasterizationSettings as _DepthSettings,
        )
        da_settings = _DepthSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy,
            bg=torch.zeros(3, device=means3D.device, dtype=torch.float32),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=0,
            campos=viewpoint_camera.camera_center,
            prefiltered=False, debug=pipe.debug,
        )
        white = torch.ones_like(means3D)
        alpha_rgb, _radii_da, depth_map = _DepthRasterizer(raster_settings=da_settings)(
            means3D=means3D, means2D=means2D.detach(), shs=None,
            colors_precomp=white, opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
        )
        if want_depth:
            out.depth = depth_map[0]
        if want_alpha:
            out.alpha = alpha_rgb[0]

    # Backward-compat alias: callers used to read ``ret.render`` for color.
    if want_color and "render" not in out:
        out.render = out.color
    return out


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
    
    def render(
            self,
            gausssian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None,
            return_types=("color",),
        ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color
            return_types: subset of ``{"color", "depth", "alpha"}``.

        Returns:
            edict containing the requested keys (``color``, ``depth``, and/or ``alpha``).
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })

        # Render (only requested outputs are computed)
        render_ret = render(
            camera_dict, gausssian, self.pipe, self.bg_color,
            override_color=colors_overwrite,
            scaling_modifier=self.pipe.scale_modifier,
            return_types=return_types,
        )

        ret = edict({})
        if "color" in return_types:
            color = render_ret['render']
            if ssaa > 1:
                color = F.interpolate(
                    color[None], size=(resolution, resolution),
                    mode='bilinear', align_corners=False, antialias=True,
                ).squeeze()
            ret.color = color
        if "depth" in return_types:
            depth = render_ret['depth']
            if ssaa > 1:
                depth = F.interpolate(
                    depth[None, None], size=(resolution, resolution),
                    mode='nearest',
                ).squeeze(0).squeeze(0)
            ret.depth = depth
        if "alpha" in return_types:
            alpha = render_ret['alpha']
            if ssaa > 1:
                alpha = F.interpolate(
                    alpha[None, None], size=(resolution, resolution),
                    mode='bilinear', align_corners=False, antialias=True,
                ).squeeze(0).squeeze(0)
            ret.alpha = alpha
        return ret
