import torch
import numpy as np
from tqdm import tqdm
from typing import Iterable

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from .random_utils import sphere_hammersley_sequence
from .pose_utils import extrinsics_look_at, intrinsics_from_fov


def yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws: Iterable[float] | float, 
    pitchs: Iterable[float] | float, 
    rs: Iterable[float] | float = 2, 
    fovs: Iterable[float] | float = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert yaw, pitch, r, fov to extrinsics and intrinsics.
    
    Args:
        yaws: A list of yaw angles or a single yaw angle
        pitchs: A list of pitch angles or a single pitch angle
        rs: A list of r values or a single r value
        fovs: A list of fov values or a single fov value
    Returns:
        extrinsics: A batch of extrinsic matrices on CUDA
        intrinsics: A batch of intrinsic matrices on CUDA
    """
    is_list = isinstance(yaws, Iterable)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, Iterable):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, Iterable):
        fovs = [fovs] * len(yaws)

    extrinsics = []
    intrinsics = []
    center = torch.tensor([0, 0, 0], dtype=torch.float32)
    up = torch.tensor([0, 0, 1], dtype=torch.float32)
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        r = torch.tensor(float(r))
        fov = torch.deg2rad(torch.tensor(float(fov)))
        yaw = torch.tensor(float(yaw))
        pitch = torch.tensor(float(pitch))
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]) * r
        extr = extrinsics_look_at(orig, center, up)
        intr = intrinsics_from_fov(fov_x=fov, fov_y=fov)
        extrinsics.append(extr)
        intrinsics.append(intr)

    if is_list:
        extrinsics = torch.stack(extrinsics)
        intrinsics = torch.stack(intrinsics)
    else:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]

    return extrinsics.cuda(), intrinsics.cuda()


def get_renderer(sample, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None,
                  verbose=True, return_types=None, **kwargs):
    """Render a sequence of frames.

    ``return_types`` (subset of ``{"color", "depth", "alpha"}``) is forwarded
    to ``GaussianRenderer.render`` / ``OctreeRenderer.render`` so callers can
    request depth / alpha directly. ``None`` (default) means the renderer's
    own default — color only for Gaussian, all native outputs for mesh.
    """
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            if 'mask' not in rets: rets['mask'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'mask' in res:
                rets['mask'].append((res['mask'].detach().cpu().numpy() * 255).astype(np.uint8))
            if 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
        else:
            render_kwargs = {'colors_overwrite': colors_overwrite}
            if return_types is not None:
                render_kwargs['return_types'] = tuple(return_types)
            res = renderer.render(sample, extr, intr, **render_kwargs)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            if 'alpha' not in rets: rets['alpha'] = []
            if 'color' in res:
                rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            else:
                rets['color'].append(None)
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
            if 'alpha' in res:
                rets['alpha'].append((res['alpha'].detach().cpu().numpy() * 255).astype(np.uint8))
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30, colors_overwrite=None, bg_color=(0,0,0)):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitch = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, colors_overwrite=colors_overwrite)
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_pose(samples, pose, resolution=512, bg_color=(0,0,0), colors_overwrite=None):
    if type(pose['extrinsic']) is not list:
        extrinsics = [pose['extrinsic']]
        intrinsics = [pose['intrinsic']]
    else:
        extrinsics = pose['extrinsic']
        intrinsics = pose['intrinsic']
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, colors_overwrite=colors_overwrite)


def coord2field(coords, resolution, density=None):
    representation_coord = Octree(
        depth=10,
        aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
        device='cuda',
        primitive='voxel',
        sh_degree=0,
        primitive_config={'solid': True if density is None else False},
    )
                
    position = (coords.contiguous() + 0.5) / resolution # should we use helf pixel shift
    representation_coord.position = position
    representation_coord.depth = torch.full(
        (representation_coord.position.shape[0], 1), 
        int(np.log2(64)), dtype=torch.uint8, device='cuda'
    )

    if density is not None:
        representation_coord.density = density.reshape(-1,1).contiguous()

    return representation_coord