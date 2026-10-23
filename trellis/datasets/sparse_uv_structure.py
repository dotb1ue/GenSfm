import os
from typing import Union
import numpy as np
import torch
from .components import TransformConditionedMixin, ImageConditionedMixin, StandardDatasetBase
from .sparse_structure import SparseStructure
from ..utils.voxel_utils import *
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer


class SparseUVStructure(TransformConditionedMixin, SparseStructure):
    """
    Dataset that combines sparse structure (binary occupancy) and UV volume data.
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
        discard_occluded_voxels (bool): whether to discard occluded voxels for the UV volume
        return_relative_depth (bool): whether to return relative depth
        dilation_sigma (float): standard deviation of Gaussian kernel for dilation
        dilation_min_weight (float): minimum weight to include in the dilated grid
        const_ssuv (bool or str): whether to use constant sparse structure for the UV volume
    """
    def __init__(
        self, 
        roots,
        discard_occluded_voxels: bool = False,
        return_relative_depth: bool = False,
        dilation_sigma: float = 0.0, 
        dilation_min_weight: float = 0.05, 
        const_ssuv: Union[bool, str] = True,
        **kwargs
    ):
        super().__init__(roots, **kwargs)
        self.discard_occluded_voxels = discard_occluded_voxels
        self.return_relative_depth = return_relative_depth
        self.dilation_sigma = dilation_sigma
        self.dilation_min_weight = dilation_min_weight
        self.const_ssuv = const_ssuv
        self._xyzs = None

    def _get_uv_volume(self, extrinsics, intrinsics):
        if self._xyzs is None:
            x = torch.arange(self.resolution, dtype=torch.float32)
            y = torch.arange(self.resolution, dtype=torch.float32)
            z = torch.arange(self.resolution, dtype=torch.float32)
            x = (x + 0.5) / self.resolution - 0.5
            y = (y + 0.5) / self.resolution - 0.5
            z = (z + 0.5) / self.resolution - 0.5
            self._xyzs = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=0)  # [3 x 64 x 64 x 64]

        uvs, depths = project_cv(self._xyzs.view(3, -1).T, intrinsics, extrinsics)
        uvs = uvs.T.reshape(2, self.resolution, self.resolution, self.resolution)
        return uvs, depths

    def _get_ssuv_coords(self, root, instance, frame_data, extrinsics, intrinsics):
        """Get the (visible) coordinates for sparse UV structure."""
        # Load occupied voxel positions from occupancy voxels
        view_name = os.path.splitext(frame_data['file_path'])[0]
        position = read_voxels(os.path.join(root, 'voxels', f'{instance}.ply'))
        position = torch.tensor(position, dtype=torch.float32)

        # Compute visible mask with depth map
        if self.discard_occluded_voxels:
            depth_map_path = os.path.join(frame_data['render_dir'], f'{view_name}_depth.png')
            depth_map = torch.tensor(decode_depth_map(Image.open(depth_map_path), frame_data))
            visible_mask = compute_voxel_visibility(
                position, 
                depth_map.unsqueeze(0), 
                intrinsics.unsqueeze(0), 
                extrinsics.unsqueeze(0), 
                1 / self.resolution
            )
            position = position[visible_mask.squeeze(0)]

        # Compute coords and apply optional dilation
        coords = ((position + 0.5) * self.resolution).int().contiguous()
        if self.dilation_sigma > 0:
            coords, _ = dilate_sparse_grid(coords, self.resolution, self.dilation_sigma, self.dilation_min_weight)

        # Sanity check that there are valid coords
        if coords.numel() == 0:
            raise ValueError(f"No valid coords for instance {instance} and view {view_name}")

        return coords
    
    def get_instance(self, root, instance, frame_data, **kwargs):
        """Get both sparse structure and UV data for an instance."""
        # Get sparse structure data (from SparseStructure parent)
        pack = super().get_instance(root, instance, frame_data=frame_data, **kwargs)

        # Create the full uv volume by projecting the voxel centers
        uvs, depths = self._get_uv_volume(pack['extrinsics'], pack['intrinsics'])
        pack['uv_volume'] = uvs.clamp(0.0, 1.0)

        # Create UV's sparse structure
        ssuv = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.int)
        if self.const_ssuv:
            # Note that this uvs is not cropped
            in_image_mask = torch.logical_and(uvs >= 0.01, uvs <= 0.99).all(dim=0)
            ssuv[:, in_image_mask] = 1
        else:
            coords = self._get_ssuv_coords(root, instance, frame_data, pack['extrinsics'], pack['intrinsics'])
            ssuv[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        pack['ssuv'] = ssuv

        # Normalize depths as relative depth
        if self.return_relative_depth:
            depths = depths.reshape(1, self.resolution, self.resolution, self.resolution)
            pack['depth_volume'] = (depths - depths.mean()) / depths.std().clamp_min(1e-2)
        
        return pack

    @torch.no_grad()
    def visualize_sample(self, sample: Union[torch.Tensor, dict]):
        if isinstance(sample, torch.Tensor):
            return SparseStructure.visualize_sample(self, sample)
        else:
            values = sample['uv_volume'].cuda()
            weight = sample['ssuv'].cuda()

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'

        # Build camera
        yaws = np.arange(4) * np.pi / 2 + np.random.uniform(-np.pi / 4, np.pi / 4)
        pitch = np.random.uniform(-np.pi / 4, np.pi / 4, 4)
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, rs=2, fovs=30)

        # Build representation for each sample
        images = []
        for i in range(weight.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(weight[i, 0] > 0, as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')

            # values[i] has shape (1/2/3, resolution, resolution, resolution)
            if values is not None:
                u_values = values[i, 0, coords[:, 0], coords[:, 1], coords[:, 2]]
                if values.shape[1] >= 2:
                    v_values = values[i, 1, coords[:, 0], coords[:, 1], coords[:, 2]]
                else:
                    v_values = torch.zeros_like(u_values)
                if values.shape[1] >= 3:
                    w_values = values[i, 2, coords[:, 0], coords[:, 1], coords[:, 2]]
                else:
                    w_values = torch.zeros_like(u_values)
                colors = torch.stack([u_values, v_values, w_values], dim=1).float().clamp(0, 1)
            else:
                colors = representation.position

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=colors)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)


class ImageConditionedSparseUVStructure(ImageConditionedMixin, SparseUVStructure):
    """
    Image-conditioned sparse UV structure dataset. Use this for UV volumes of cropped views.
    """
    pass
