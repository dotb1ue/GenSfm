import numpy as np
import torch
from torch import Tensor


def read_voxels(ply_path):
    """
    Load voxel positions from a ply file.

    Args:
        ply_path: path to the voxel ply file

    Returns:
        vertices: (N, 3) array of voxel positions
    """
    import plyfile
    plydata = plyfile.PlyData.read(ply_path)
    vertices = np.stack([plydata['vertex'][k] for k in ['x', 'y', 'z']], axis=-1)
    return vertices


def write_voxels(ply_path, vertices, text=False):
    """
    Write voxel positions to a ply file.

    Args:
        ply_path: path to the voxel ply file
        vertices: (N, 3) array of voxel positions
        text: whether to write the ply file in text format
    """
    import plyfile
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    vertices = vertices.astype(np.float32)
    vertices_data = np.array([tuple(v) for v in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply_data = [plyfile.PlyElement.describe(vertices_data, 'vertex')]
    plyfile.PlyData(ply_data, text=text).write(ply_path)


def decode_depth_map(depth_img, frame_data, background_depth=-np.inf):
    """
    Decode depth map from depth image and renormalize it to 3D space.
    
    Args:
        depth_img: (H, W) array of depth values in pixels
        frame_data: frame data containing depth min/max values
        background_depth: depth value to set for background pixels
        
    Returns:
        depth_m: (H, W) array of depth values in meters
    """
    depth_raw = np.array(depth_img, dtype=np.uint16)
    
    # Build invalid mask for background pixels
    invalid_mask = (depth_raw == 65535)
    
    # Normalize to [0,1]
    depth_norm = depth_raw.astype(np.float32) / 65535.0
    
    # Get depth range from frame data
    d_min, d_max = frame_data['depth']['min'], frame_data['depth']['max']
    
    # Linear remap to [d_min, d_max]
    depth_m = depth_norm * (d_max - d_min) + d_min
    
    # Invalidate background
    depth_m[invalid_mask] = background_depth
 
    return depth_m


def project_cv(
    points: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Project 3D points to 2D following the OpenCV convention

    Args:
        points (Tensor): [..., N, 3] 3D points
        intrinsics (Tensor): [..., 3, 3] intrinsics matrix
        extrinsics (Tensor): [..., 4, 4] extrinsics matrix

    Returns:
        uv_coord (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (Tensor): [..., N] linear depth
    """
    intrinsics = torch.cat([
        torch.cat([intrinsics, torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device)], dim=-1),
        torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(*intrinsics.shape[:-2], 1, 4)
    ], dim=-2)
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)], dim=-1)
    points = points @ transform.mT
    uv_coord = points[..., :2] / points[..., 2:3]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


def unproject_cv(
    uv: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor | None = None,
) -> Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (Tensor): [..., N] depth value
        intrinsics (Tensor): [..., 3, 3] intrinsics matrix
        extrinsics (Tensor): [..., 4, 4] extrinsics matrix

    Returns:
        points (Tensor): [..., N, 3] 3d points
    """
    intrinsics = torch.cat([
        torch.cat([intrinsics, torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device)], dim=-1),
        torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(*intrinsics.shape[:-2], 1, 4)
    ], dim=-2)
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = torch.cat([uv, torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1) * depth[..., None]
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    points = points @ torch.linalg.inv(transform).mT
    points = points[..., :3]
    return points


def compute_voxel_visibility(occupied_voxels, depth_maps, intrinsics, extrinsics, voxel_size):
    """
    Compute voxel visibility using depth map comparison for batched inputs.
    
    Args:
        occupied_voxels: Tensor (N, 3) array of occupied voxel centers
        depth_maps: Tensor (B, H, W) batch of depth maps in meters
        intrinsics: Tensor (B, 3, 3) batch of camera intrinsic matrices
        extrinsics: Tensor (B, 4, 4) batch of camera extrinsic matrices (world-to-camera)
        voxel_size: size of each voxel
        
    Returns:
        visible_masks: Tensor (B, N) boolean array indicating which voxels are visible for each view
    """
    B = depth_maps.shape[0]  # batch size
    N = len(occupied_voxels)
    
    # Generate sample points for each voxel
    half_size = voxel_size / 2
    sample_offsets = torch.tensor([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
        [ 0,  0,  0],
    ], dtype=torch.float32, device=occupied_voxels.device) * half_size
    
    # Get all sample points for all voxels: (N, n_samples, 3) -> (N*n_samples, 3)
    sample_points = occupied_voxels[:, None, :] + sample_offsets  # (N, n_samples, 3)
    sample_points = sample_points.reshape(-1, 3)  # (N*n_samples, 3)
    
    # Project using project_cv (UV in [0,1] range)
    uv, depth = project_cv(sample_points, intrinsics, extrinsics)
    uv = uv.reshape(B, N, sample_offsets.shape[0], 2)  # (B, N, n_samples, 2)
    depth = depth.reshape(B, N, sample_offsets.shape[0])  # (B, N, n_samples)
    
    # Sample depths for all batch items at once
    sampled_depths = torch.nn.functional.grid_sample(
        depth_maps[:, None, :, :],  # (B, 1, H, W)
        uv * 2 - 1,                 # (B, N, n_samples, 2), UV in [-1,1] range
        mode='bilinear', 
        align_corners=False,
        padding_mode='border'
    ).squeeze(1)  # (B, N, n_samples) - squeeze the channel dimension
    
    # Check visibility: sample point is visible if its depth is less than map depth
    sample_points_visible = (depth > 0) & (depth <= sampled_depths)  # (B, N, n_samples)
    
    # A voxel is visible if any of its sample points are visible
    return sample_points_visible.any(dim=2)  # (B, N)


def dilate_sparse_grid(coords: Tensor, resolution: int, sigma: float, min_weight: float):
    """
    Perform dilation on sparse 3D grid using Gaussian kernel.
    
    Args:
        coords: Tensor (N, 3) - original voxel coordinates in [0, resolution-1]
        resolution: int - grid resolution
        sigma: float - standard deviation of Gaussian kernel
        min_weight: float - minimum weight to include in the dilated grid
        
    Returns:
        dilated_coords: Tensor (M, 3) - expanded coordinates
        weights: Tensor (M,) - corresponding weights
    """
    assert coords.ndim == 2 and coords.size(1) == 3

    # Create Gaussian kernel for the radius calculated from sigma
    chi2_value = 7.814728  # 95% percentile of chi2 distribution with 3 degrees of freedom
    r = int(np.ceil(sigma * np.sqrt(chi2_value)))
    coord_range = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)

    # Create 3D grid for kernel
    X, Y, Z = torch.meshgrid(coord_range, coord_range, coord_range, indexing='ij')
    offsets = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1) # (kernel_size^3, 3)
    dist_squared = offsets.float().square().sum(dim=1) # (kernel_size^3,)
    kernel = torch.exp(-dist_squared / (2 * sigma**2))

    # Filter out low-weight offsets
    kernel_mask = kernel >= min_weight
    kernel = kernel[kernel_mask]
    offsets = offsets[kernel_mask]
    
    # Apply all offsets to all coordinates at once and filter out invalid coordinates
    all_new_coords = coords[:, None, :] + offsets[None, :, :]  # (N, num_offsets, 3)
    valid = ((all_new_coords >= 0) & (all_new_coords < resolution)).all(dim=2)  # (N, num_offsets)
    
    # Apply mask to get flattened valid coordinates and weights
    valid_coords = all_new_coords[valid]
    valid_weights = kernel[None, :].expand(coords.size(0), -1)[valid]

    # Linearize coords as key to remove duplicates and aggregate weights
    keys = valid_coords[:, 0] * resolution**2 + valid_coords[:, 1] * resolution + valid_coords[:, 2]
    unique_keys, inverse_indices = torch.unique(keys, sorted=True, return_inverse=True)
    
    # Aggregate weights for duplicate coordinates
    weights = torch.full((unique_keys.numel(),), 0.0, device=coords.device, dtype=torch.float32)
    weights.scatter_reduce_(0, inverse_indices, valid_weights, reduce="amax", include_self=False)

    # Recover (x,y,z)
    x = (unique_keys // resolution**2)
    y = ((unique_keys // resolution) % resolution)
    z = (unique_keys % resolution)
    dilated_coords = torch.stack([x, y, z], dim=1)
    
    return dilated_coords, weights


def dilate_binary_volume(volume: Tensor, sigma: float, min_weight: float):
    """
    Perform dilation on a binary 3D volume using a Gaussian kernel.
    
    Args:
        volume: Tensor (res, res, res) - binary input volume (0s and 1s)
        sigma: float - standard deviation of Gaussian kernel
        min_weight: float - minimum weight to include in the dilated volume
        
    Returns:
        dilated_volume: Tensor (res, res, res) - binary dilated volume
    """
    assert volume.ndim == 3
    assert volume.shape[0] == volume.shape[1] == volume.shape[2], "Input volume must be a cube"
    res = volume.shape[0]

    # Extract coordinates of active voxels (where volume == 1)
    coords = torch.nonzero(volume, as_tuple=False)  # (N, 3)

    # Use dilate_sparse_grid to perform the dilation
    dilated_coords, _ = dilate_sparse_grid(coords, res, sigma, min_weight)

    # Create output binary volume
    dilated_volume = torch.zeros_like(volume)
    dilated_volume[dilated_coords[:, 0], dilated_coords[:, 1], dilated_coords[:, 2]] = 1

    return dilated_volume


def expand_binary_volume(
    logits: Tensor,
    min_num_activated: int = 10,
    threshold_step: float = 0.05,
    activation_level: float = 0.0,
) -> Tensor:
    """
    Binarize logits and progressively lower the threshold until each sample
    has at least `min_num_activated` active voxels.

    Args:
        logits: (N, ...) float tensor — raw logits.
        min_num_activated: minimum active voxels per sample.
        threshold_step: step size for lowering the threshold.
        activation_level: initial binarization threshold.

    Returns:
        (N, ...) int tensor — binary volume with guaranteed minimum occupancy.
    """
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits found in expand_binary_volume()!")
    threshold = activation_level
    expanded = (logits > threshold).int()
    num_activated = expanded.flatten(start_dim=1).sum(dim=1)  # [N]
    mask = num_activated < min_num_activated
    while mask.any():
        threshold -= threshold_step
        expanded[mask] = (logits[mask] > threshold).int()
        num_activated = expanded.flatten(start_dim=1).sum(dim=1)  # [N]
        mask = num_activated < min_num_activated
    return expanded


@torch.no_grad()
def render_occlusion(occupied_voxels, intrinsics, extrinsics, voxel_size, dilation=False):
    """
    Compute voxel visibility using depth map comparison for batched inputs.
    
    Args:
        occupied_voxels: Tensor (N, 3) array of occupied voxel centers
        intrinsics: Tensor (B, 3, 3) batch of camera intrinsic matrices
        extrinsics: Tensor (B, 4, 4) batch of camera extrinsic matrices (world-to-camera)
        voxel_size: size of each voxel
        dilation: whether to dilate the occupied voxels
        
    Returns:
        visible_masks: Tensor (B, N) boolean array indicating which voxels are visible for each view
    """
    from .render_utils import coord2field, get_renderer
    assert occupied_voxels.dtype in (torch.long, torch.int32, torch.int64), "occupied_voxels must be integer type"
    occupied_voxels = occupied_voxels.long()
    input_device = occupied_voxels.device
    # If device is CPU, cast everything to CUDA (renderer requires CUDA)
    if input_device.type == "cpu":
        device = torch.device("cuda")
    else:
        device = input_device

    resolution = int(1 / voxel_size)
    xyz_fields = coord2field(occupied_voxels, resolution)
    renderer = get_renderer(xyz_fields)
    renderer.rendering_options.ssaa = 1
    
    occupied_voxels = occupied_voxels.to(device)
    intrinsics = intrinsics.to(device)
    extrinsics = extrinsics.to(device)
    xyz_fields = coord2field(occupied_voxels, resolution)
    renderer = get_renderer(xyz_fields)
    renderer.rendering_options.ssaa = 1
    visible_masks = []
    renderings = []
    # For each camera/view
    for extr, intr in zip(extrinsics, intrinsics):
        extr = extr.to(device)
        intr = intr.to(device)
        res = renderer.render(xyz_fields, extr, intr, colors_overwrite=occupied_voxels.to(torch.float32))
        color = res['color']
        alpha = res['alpha']

        uvw = (xyz_fields.position @ extr[:3,:3].T + extr[:3,3:].T) @ intr.T
        res = renderer.render(xyz_fields, extr, intr, colors_overwrite=uvw)
        renderings.append(res)

        visible_voxels = color.permute(1, 2, 0)[alpha > 0]
        visible_voxels = visible_voxels.round().long().clamp(0, resolution - 1)
        if visible_voxels.numel() == 0:
            visible_mask = torch.zeros(len(occupied_voxels), dtype=torch.bool, device=device)
        else:
            visible_voxels = visible_voxels.unique(dim=0)
            binary_volume = torch.zeros((resolution, resolution, resolution), dtype=torch.bool, device=device)
            binary_volume[visible_voxels[:, 0], visible_voxels[:, 1], visible_voxels[:, 2]] = True
            # Apply 3x3 max pooling to binary_volume to dilate visible voxels
            if dilation:
                binary_volume = torch.nn.functional.max_pool3d(
                    binary_volume.unsqueeze(0).unsqueeze(0).float(), 
                    kernel_size=3, stride=1, padding=1
                ).squeeze(0).squeeze(0).bool()

            visible_mask = binary_volume[occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]]
        visible_masks.append(visible_mask)

    visible_masks = torch.stack(visible_masks, dim=0).to(input_device)
    return visible_masks, renderings
