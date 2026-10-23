import os
from typing import Union
import numpy as np
import torch
from ..utils.voxel_utils import read_voxels
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
from .components import StandardDatasetBase
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer


class SparseStructure(StandardDatasetBase):
    """
    Sparse structure dataset

    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(self,
        roots,
        resolution: int = 64,
        **kwargs,
    ):
        self.resolution = resolution
        self.value_range = (0, 1)

        super().__init__(roots, **kwargs)
        
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'voxelized']]
        stats['Voxelized'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance, **kwargs):
        pack = super().get_instance(root, instance, **kwargs)
        position = read_voxels(os.path.join(root, 'voxels', f'{instance}.ply'))
        coords = ((torch.tensor(position) + 0.5) * self.resolution).int().contiguous()
        ss = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.int)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        pack['ss'] = ss
        return pack

    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        ss = ss if isinstance(ss, torch.Tensor) else ss['ss']
        
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
        
        # Build each representation
        images = []
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0], as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)
