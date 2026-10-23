from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import Tensor


def calibrate_camera(
    obj_points: Tensor,
    img_points: Tensor,
    initial_camera_matrix: Tensor = None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calibrates the camera for a batch of single-view observations using OpenCV's calibrateCamera.
    Each batch item represents an independent calibration with one view of corresponding 3D object
    points and 2D image points. Assumes image points are in normalized coordinates (0-1 range).

    Args:
        obj_points: Tensor of 3D world points (X, Y, Z) with shape (batch_size, num_points, 3).
        img_points: Tensor of 2D image points (u, v) with shape (batch_size, num_points, 2).
        initial_camera_matrix: Optional Tensor with initial guess for the intrinsic matrix,
            shape (batch_size, 3, 3). If None, a default normalized matrix is used.

    Returns:
        intrinsics: Tensor of recovered intrinsic matrices, shape (batch_size, 3, 3).
        extrinsics: Tensor of extrinsic matrices [R | t], shape (batch_size, 3, 4).
        reproj_errors: Tensor of mean reprojection errors, shape (batch_size,).
    """
    # Convert inputs to NumPy for OpenCV compatibility
    obj_points_np = obj_points.float().contiguous().cpu().numpy() if isinstance(obj_points, Tensor) else obj_points
    img_points_np = img_points.float().contiguous().cpu().numpy() if isinstance(img_points, Tensor) else img_points

    batch_size = obj_points_np.shape[0]
    
    if initial_camera_matrix is None:
        # Default initial guess for normalized intrinsics (assuming u,v in [0,1])
        default_matrix = np.array([
            [0.8, 0.0, 0.5],
            [0.0, 0.8, 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        initial_camera_matrix_np = np.tile(default_matrix, (batch_size, 1, 1))
    else:
        initial_camera_matrix_np = initial_camera_matrix.cpu().numpy()

    extrinsics_list = []
    intrinsics_list = []
    reproj_errors_list = []

    for i in range(batch_size):
        # Prepare inputs for single view: lists as required by calibrateCamera
        current_obj_points = [obj_points_np[i]]
        current_img_points = [img_points_np[i]]
        
        # Dummy image size since we're using normalized coordinates and initial guess
        img_size = (1, 1)

        # Perform calibration with fixed distortion parameters (assuming minimal distortion)
        ret, recovered_mtx, recovered_dist, rvecs, tvecs = cv2.calibrateCamera(
            current_obj_points,
            current_img_points,
            img_size,
            cameraMatrix=initial_camera_matrix_np[i],
            distCoeffs=None,
            flags=(
                cv2.CALIB_USE_INTRINSIC_GUESS |
                cv2.CALIB_FIX_K1 |
                cv2.CALIB_FIX_K2 |
                cv2.CALIB_FIX_K3 |
                cv2.CALIB_FIX_K4 |
                cv2.CALIB_FIX_K5 |
                cv2.CALIB_ZERO_TANGENT_DIST
            )
        )

        # Construct extrinsic matrix [R | t]
        rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
        translation_vector = tvecs[0].reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))
        extrinsics_list.append(extrinsic_matrix)

        intrinsics_list.append(recovered_mtx)

        # Compute mean reprojection error for this view
        mean_error = 0.0
        num_views = len(current_obj_points)  # Always 1 in this case
        for view_idx in range(num_views):
            projected_points, _ = cv2.projectPoints(
                current_obj_points[view_idx],
                rvecs[view_idx],
                tvecs[view_idx],
                recovered_mtx,
                recovered_dist
            )

            diff = current_img_points[view_idx] - projected_points.reshape(-1,2)
            per_point_errors = np.linalg.norm(diff, axis=1)
            error = np.mean(per_point_errors)
            mean_error += error
        mean_error /= num_views
        reproj_errors_list.append(mean_error)

    # Stack results and convert back to torch tensors
    intrinsics = torch.from_numpy(np.stack(intrinsics_list)).to(obj_points.device).float()
    extrinsics = torch.from_numpy(np.stack(extrinsics_list)).to(obj_points.device).float()
    reproj_errors = torch.from_numpy(np.stack(reproj_errors_list)).to(obj_points.device).float()

    return intrinsics, extrinsics, reproj_errors


def calibrate_camera_dlt(
    obj_points: Tensor,
    img_points: Tensor,
    initial_camera_matrix: Tensor = None,
    weights: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calibrates the camera for a batch of single-view observations using OpenCV's calibrateCamera.
    Each batch item represents an independent calibration with one view of corresponding 3D object
    points and 2D image points. Assumes image points are in normalized coordinates (0-1 range).

    Args:
        obj_points: Tensor of 3D world points (X, Y, Z) with shape (batch_size, num_points, 3).
        img_points: Tensor of 2D image points (u, v) with shape (batch_size, num_points, 2).
        initial_camera_matrix: Optional Tensor with initial guess for the intrinsic matrix,
            shape (batch_size, 3, 3). If None, a default normalized matrix is used.

    Returns:
        intrinsics: Tensor of recovered intrinsic matrices, shape (batch_size, 3, 3).
        extrinsics: Tensor of extrinsic matrices [R | t], shape (batch_size, 3, 4).
        reproj_errors: Tensor of mean reprojection errors, shape (batch_size,).
    """
    transform = dlt_calibration(obj_points.permute(0,2,1), img_points.permute(0,2,1), weights=weights)
    intrinsics, extrinsics = decompose_projection(transform)
    xyz_cam = obj_points @ transform[:, :3, :3].permute(0,2,1) + transform[:, :3, 3:].permute(0,2,1)
    reproj_errors = ((xyz_cam[..., :2] / xyz_cam[..., 2:]) - img_points).norm(dim=-1).mean(dim=-1)
    return intrinsics, extrinsics, reproj_errors


def solve_pnp_known_k(
    obj_points: Tensor,
    img_points: Tensor,
    K: Tensor,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Solves for camera extrinsics [R | t] given known intrinsics K, using a
    linear DLT-PnP with SE(3) Procrustes projection. Bypasses the fx/t_z
    near-ambiguity that biases per-camera translation scale under full DLT.

    Args:
        obj_points: 3D world points, shape (batch_size, num_points, 3).
        img_points: 2D image points in the normalized frame matched by K,
            shape (batch_size, num_points, 2).
        K: Known intrinsic matrices, shape (batch_size, 3, 3).
        weights: Optional per-point weights, shape (batch_size, num_points).

    Returns:
        extrinsics: [R | t] matrices of shape (batch_size, 3, 4).
        reproj_errors: Mean reprojection error in the normalized pixel frame,
            shape (batch_size,).
    """
    assert obj_points.dim() == 3 and obj_points.shape[-1] == 3
    assert img_points.dim() == 3 and img_points.shape[-1] == 2
    assert K.dim() == 3 and K.shape[-2:] == (3, 3)
    assert obj_points.shape[:2] == img_points.shape[:2]

    B = obj_points.shape[0]
    ones = torch.ones_like(img_points[..., :1])
    hom = torch.cat([img_points, ones], dim=-1)                  # [B, N, 3]
    K_inv = torch.linalg.inv(K)                                  # [B, 3, 3]
    rays = hom @ K_inv.transpose(-1, -2)                         # [B, N, 3]
    normalized = rays[..., :2] / rays[..., 2:].clamp(min=1e-9)   # [B, N, 2]

    P = dlt_calibration(
        obj_points.permute(0, 2, 1),
        normalized.permute(0, 2, 1),
        weights=weights,
    )                                                            # [B, 3, 4]

    R_raw = P[:, :3, :3]
    t_raw = P[:, :3, 3]
    U, S, Vh = torch.linalg.svd(R_raw)
    det_sign = torch.sign(torch.linalg.det(U @ Vh))
    D = torch.eye(3, device=P.device, dtype=P.dtype).expand(B, -1, -1).clone()
    D[:, 2, 2] = det_sign
    R = U @ D @ Vh                                               # [B, 3, 3]
    scale = S.mean(dim=-1).clamp(min=1e-9)                       # [B]
    t = t_raw / scale.unsqueeze(-1)                              # [B, 3]
    extrinsics = torch.cat([R, t.unsqueeze(-1)], dim=-1)         # [B, 3, 4]

    xyz_cam = obj_points @ R.permute(0, 2, 1) + t.unsqueeze(1)
    proj = xyz_cam[..., :2] / xyz_cam[..., 2:].clamp(min=1e-9)
    reproj_errors = (proj - normalized).norm(dim=-1).mean(dim=-1)
    return extrinsics, reproj_errors


def decompose_projection(proj_matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Decomposes a batch of 3x4 projection matrices into intrinsic matrices (K), 
    rotation matrices (R), and translation vectors (t).
    
    Args:
        proj_matrix (Tensor): Batch of projection matrices of shape (B, 3, 4) where P = K [R | t]
    
    Returns:
        K (Tensor): Batch of intrinsic matrices of shape (B, 3, 3).
        R (Tensor): Batch of rotation matrices of shape (B, 3, 3).
        t (Tensor): Batch of translation vectors of shape (B, 3).
    """
    # Ensure input is a batch of 3x4 matrices
    assert proj_matrix.dim() == 3 and proj_matrix.shape[1:] == (3, 4), "Input must be of shape (B, 3, 4)."
    
    # Step 1: Extract the left 3x3 submatrix (M = K * R)
    M = proj_matrix[:, :3, :3].double()
    
    # Step 2: Decompose M into K and R using RQ decomposition
    # Note: PyTorch doesn't have RQ decomposition; use QR and adjust.
    M_inv = torch.linalg.pinv(M)
    Q, R = torch.linalg.qr(M_inv)
    K = torch.linalg.pinv(R).float()
    R = torch.linalg.pinv(Q).float()
    # Step 3: resolve sign ambiguities
    diag = torch.sign(torch.diagonal(K, dim1=-2, dim2=-1))  # Get the sign of the diagonal
    diag[diag == 0] = 1.0  # Set 0 to 1 to avoid division by zero
    D = torch.diag_embed(diag)  # Create a diagonal matrix with the signs
    K = K @ D  # Multiply the intrinsic matrix by the diagonal matrix
    R = D @ R
    
    # Step 4: Compute translation vector t = K^{-1} * P[:, 3]
    t = torch.linalg.pinv(K) @ proj_matrix[:, :, 3:4]
    
    # Step 5
    det = (torch.linalg.det(R) < 0)
    if det.any():
        # should not arrive here because we have normalize proj_matrix
        R[det] *= -1 
    
    # Step 6 normalize scale
    scale = K[:, 2:3, 2:3]

    K = K / scale
    
    return K, torch.cat([R, t], dim=-1)


def dlt_calibration(obj_points: Tensor, img_points: Tensor, weights: Tensor | None = None) -> Tensor:
    """
    Compute the camera matrix P using the Direct Linear Transformation (DLT) algorithm in PyTorch.
    https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-21-DLT.pptx.pdf

    Args:
        obj_points: Tensor of 3D world points (X, Y, Z) of shape (batch_size, 3, N) or (1, 3, N)
        img_points: Tensor of 2D image points (u, v) of shape (batch_size, 2, N)
        weights: Tensor of weights (w) of shape (batch_size, N) or None

    Returns:
        P: Tensor, the camera matrix of shape (batch_size, 3, 4)
        A: Tensor, the coefficient matrix of shape (batch_size, 2N, 12)
    """
    # Check input shapes for batched version
    assert obj_points.dim() == 3 and obj_points.shape[1] == 3, "obj_points must be of shape (batch_size, 3, N)"
    assert img_points.dim() == 3 and img_points.shape[1] == 2, "img_points must be of shape (batch_size, 2, N)"
    assert obj_points.shape[2] == img_points.shape[2], "Number of points must match"
    assert weights is None or (weights.dim() == 2 and weights.shape[0] == img_points.shape[0] and weights.shape[1] == img_points.shape[2]), "weights must be of shape (batch_size, N)"
    
    batch_size, _, num_points = img_points.shape
    
    # Ensure the number of input points is at least 6
    if num_points < 6:
        raise ValueError("At least 6 point correspondences are required (N >= 6)")
    
    # Add homogeneous coordinate (1s) to 3D points
    # obj_points: (batch_size, 3, N) -> (batch_size, 4, N)
    obj_points_homo = F.pad(obj_points, (0, 0, 0, 1), mode='constant', value=1)
    
    if weights is not None:
        # weights: (batch_size, N) -> (batch_size, 1, N)
        obj_points_homo = obj_points_homo * weights.unsqueeze(1)
    
    # Separate u and v coordinates
    # img_points: (batch_size, 2, N) -> (batch_size, 1, N) each
    u = img_points[:, 0:1, :]  # (batch_size, 1, N)
    v = img_points[:, 1:2, :]  # (batch_size, 1, N)
    
    # Initialize coefficient matrix A for each batch
    # A: (batch_size, 2N, 12)
    A = torch.zeros(batch_size, 2 * num_points, 12, device=obj_points.device)
    
    # Construct the matrix A for each batch as follows:
    # For each point i:
    #   Row 2i: [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
    #   Row 2i+1: [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    
    # First 4 columns (block for p11, p12, p13, p14)
    A[:, 0::2, 0:4] = obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    
    # Next 4 columns (block for p21, p22, p23, p24)
    A[:, 1::2, 4:8] = obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    
    # Last 4 columns (block for p31, p32, p33, p34)
    A[:, 0::2, 8:12] = -u.transpose(1, 2) * obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    A[:, 1::2, 8:12] = -v.transpose(1, 2) * obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    
    # Solve Ap = 0 using SVD for each batch
    # Use torch.linalg.svd with full_matrices=False for efficiency
    _, _, V = torch.linalg.svd(A, full_matrices=False)
    
    # Solution is the last row of V for each batch (minimizes ||Ap||)
    # V: (batch_size, 2N, 12) -> P_flat: (batch_size, 12)
    P_flat = V[:, -1, :]
    
    # Reshape to camera matrices
    # P: (batch_size, 3, 4)
    P = P_flat.view(batch_size, 3, 4)

    # there is a scale ambiguity (with sign), because we use opencv coordinate
    # the determine should be positive
    P[torch.linalg.det(P[:,:3,:3]) < 0] *= -1

    return P


def numpy_quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices
    
    ## Parameters
        quaternion (ndarray): shape (..., 4), the quaternions to convert
    
    ## Returns
        ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), np.finfo(quaternion.dtype).tiny)
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    zeros = np.zeros_like(w)
    I = np.eye(3, dtype=quaternion.dtype)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * (xyz ** 2).sum(axis=-1)[..., None, None]
    B = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], axis=-1).reshape(*quaternion.shape[:-1], 3, 3)
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def numpy_matrix_to_quaternion(rot_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    ## Parameters
        rot_mat (ndarray): shape (..., 3, 3), the rotation matrices to convert

    ## Returns
        ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [rot_mat[..., i, j] for i in range(3) for j in range(3)]

    diag = np.diagonal(rot_mat, axis1=-2, axis2=-1)
    M = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype)
    wxyz = 0.5 * np.clip(1 + diag @ M.T, 0.0, None) ** 0.5
    max_idx = np.argmax(wxyz, axis=-1)
    xw = np.sign(m21 - m12)
    yw = np.sign(m02 - m20)
    zw = np.sign(m10 - m01)
    yz = np.sign(m21 + m12)
    xz = np.sign(m02 + m20)
    xy = np.sign(m01 + m10)
    ones = np.ones_like(xw)
    sign = np.where(
        max_idx[..., None] == 0,
        np.stack([ones, xw, yw, zw], axis=-1),
        np.where(
            max_idx[..., None] == 1,
            np.stack([xw, ones, xy, xz], axis=-1),
            np.where(
                max_idx[..., None] == 2,
                np.stack([yw, xy, ones, yz], axis=-1),
                np.stack([zw, xz, yz, ones], axis=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), np.finfo(quat.dtype).tiny)
    return quat


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last, as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp(min=1e-8).sqrt()

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part last, as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def closed_form_inverse_se3(se3: Tensor, R: Tensor | None = None, T: Tensor | None = None) -> Tensor:
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: (N, 4, 4) or (N, 3, 4) tensor of SE3 matrices.
        R (optional): (N, 3, 3) tensor of rotation matrices.
        T (optional): (N, 3, 1) tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.
    """

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Inverse rotation by transposing R
    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
    inverted_matrix = torch.eye(4, dtype=R.dtype, device=R.device)
    inverted_matrix = inverted_matrix[None].repeat(R.shape[0], 1, 1)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def rotation_angle(rot_gt: Tensor, rot_pred: Tensor, ambiguity: bool = False, eps: float = 1e-15) -> Tensor:
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: (N, 3, 3) tensor of ground truth rotation matrices
        rot_pred: (N, 3, 3) tensor of predicted rotation matrices
        ambiguity: whether to handle direction ambiguity
        eps: small value to avoid numerical issues

    Returns:
        (N,) tensor of rotation angle error in degrees
    """
    q_pred = matrix_to_quaternion(rot_pred)
    q_gt = matrix_to_quaternion(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * (180.0 / torch.pi)

    if ambiguity:
        rel_rangle_deg = torch.min(rel_rangle_deg, (180 - rel_rangle_deg).abs())

    return rel_rangle_deg


def translation_angle(tvec_gt: Tensor, tvec_pred: Tensor, ambiguity: bool = False) -> Tensor:
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: (N, 3) tensor of ground truth translation vectors
        tvec_pred: (N, 3) tensor of predicted translation vectors
        ambiguity: whether to handle direction ambiguity

    Returns:
        (N,) tensor of translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * (180.0 / torch.pi)

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    return rel_tangle_deg


def compare_translation_by_angle(t_gt: Tensor, t: Tensor, eps: float = 1e-15, default_err: float = 1e6) -> Tensor:
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: (N, 3) tensor of ground truth translation vectors
        t: (N, 3) tensor of predicted translation vectors
        eps: small value to avoid division by zero
        default_err: default error value for invalid cases

    Returns:
        (N,) tensor of angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    cos_theta = torch.clamp(torch.sum(t * t_gt, dim=1), -1.0 + eps, 1.0 - eps)
    err_t = torch.acos(cos_theta)

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def intrinsics_from_focal_center(fx: float | Tensor, fy: float | Tensor, cx: float | Tensor, cy: float | Tensor) -> Tensor:
    """
    Get OpenCV intrinsics matrix

    ## Parameters
        focal_x (float | Tensor): focal length in x axis
        focal_y (float | Tensor): focal length in y axis
        cx (float | Tensor): principal point in x axis
        cy (float | Tensor): principal point in y axis

    ## Returns
        (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    if not isinstance(fx, Tensor):
        fx = torch.tensor(fx, dtype=torch.float32)
    if not isinstance(fy, Tensor):
        fy = torch.tensor(fy, dtype=torch.float32)
    if not isinstance(cx, Tensor):
        cx = torch.tensor(cx, dtype=torch.float32, device=fx.device)
    if not isinstance(cy, Tensor):
        cy = torch.tensor(cy, dtype=torch.float32, device=fx.device)
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    ret = torch.stack([
        fx, zeros, cx, 
        zeros, fy, cy, 
        zeros, zeros, ones
    ], dim=-1).unflatten(-1, (3, 3))
    return ret


def intrinsics_from_fov(
    *,
    fov_x: float | Tensor | None = None,
    fov_y: float | Tensor | None = None,
    fov_max: float | Tensor | None = None,
    fov_min: float | Tensor | None = None,
    cx: float | Tensor = 0.5,
    cy: float | Tensor = 0.5,
    aspect_ratio: float | Tensor | None = None,
) -> Tensor:
    """
    Get normalized OpenCV intrinsics matrix from given field of view.
    You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

    Parameters
    ----
        fov_x (float | Tensor): field of view in x axis
        fov_y (float | Tensor): field of view in y axis
        fov_max (float | Tensor): field of view in largest dimension
        fov_min (float | Tensor): field of view in smallest dimension
        cx (float | Tensor): principal point x coordinate
        cy (float | Tensor): principal point y coordinate
        aspect_ratio (float | Tensor): aspect ratio of the image

    Returns
    ----
        (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    if fov_max is not None:
        fx = torch.maximum(1, 1 / aspect_ratio) / (2 * torch.tan(fov_max / 2))
        fy = torch.maximum(1, aspect_ratio) / (2 * torch.tan(fov_max / 2))
    elif fov_min is not None:
        fx = torch.minimum(1, 1 / aspect_ratio) / (2 * torch.tan(fov_min / 2))
        fy = torch.minimum(1, aspect_ratio) / (2 * torch.tan(fov_min / 2))
    elif fov_x is not None and fov_y is not None:
        fx = 1 / (2 * torch.tan(fov_x / 2))
        fy = 1 / (2 * torch.tan(fov_y / 2))
    elif fov_x is not None:
        fx = 1 / (2 * torch.tan(fov_x / 2))
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / (2 * torch.tan(fov_y / 2))
        fx = fy / aspect_ratio
    ret = intrinsics_from_focal_center(fx, fy, cx, cy)
    return ret


def fov_from_intrinsics(intrinsics: Tensor) -> Tensor:
    """
    Calculate the field of view from the intrinsic matrix intrinsics.

    Args:
        intrinsics: (N, 3, 3) tensor of intrinsic matrices

    Returns:
        (N, 2) tensor of field of view in degrees, (fov_x, fov_y)
    """
    assert intrinsics.dim() == 3 and intrinsics.shape[1:] == (3, 3), "intrinsics must be of shape (N, 3, 3)"
    
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    fov_x = 2 * torch.atan(0.5 / fx)
    fov_y = 2 * torch.atan(0.5 / fy)
    fov_x_deg = fov_x * (180.0 / torch.pi)
    fov_y_deg = fov_y * (180.0 / torch.pi)
    return torch.stack([fov_x_deg, fov_y_deg], dim=-1)


def perspective_from_fov(
    *,
    fov_x: float | Tensor | None = None,
    fov_y: float | Tensor | None = None,
    fov_min: float | Tensor | None = None,
    fov_max: float | Tensor | None = None,
    aspect_ratio: float | Tensor | None = None,
    near: float | Tensor = 1,
    far: float | Tensor = 3,
) -> Tensor:
    """
    Get OpenGL perspective matrix from field of view 

    ## Returns
        (Tensor): [..., 4, 4] perspective matrix
    """
    if fov_max is not None:
        fx = torch.maximum(1, 1 / aspect_ratio) / torch.tan(fov_max / 2)
        fy = torch.maximum(1, aspect_ratio) / torch.tan(fov_max / 2)
    elif fov_min is not None:
        fx = torch.minimum(1, 1 / aspect_ratio) / torch.tan(fov_min / 2)
        fy = torch.minimum(1, aspect_ratio) / torch.tan(fov_min / 2)
    elif fov_x is not None and fov_y is not None:
        fx = 1 / torch.tan(fov_x / 2)
        fy = 1 / torch.tan(fov_y / 2)
    elif fov_x is not None:
        fx = 1 / torch.tan(fov_x / 2)
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / torch.tan(fov_y / 2)
        fx = fy / aspect_ratio
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    if not isinstance(near, Tensor):
        near = torch.full_like(fx, near)
    if not isinstance(far, Tensor):
        far = torch.full_like(fx, far)
    perspective = torch.stack([
        fx, zeros, zeros, zeros,
        zeros, fy, zeros, zeros,
        zeros, zeros, (near / far + 1) / (near / far - 1), 2. * near / (near / far - 1),
        zeros, zeros, -ones, zeros
    ], dim=-1).unflatten(-1, (4, 4))
    return perspective


def perspective_to_intrinsics(perspective: Tensor) -> Tensor:
    """
    OpenGL perspective matrix to OpenCV intrinsics

    ## Parameters
        perspective (Tensor): [..., 4, 4] OpenGL perspective matrix

    ## Returns
        (Tensor): shape [..., 3, 3] OpenCV intrinsics
    """
    assert torch.allclose(perspective[:, [0, 1, 3], 3], 0), "The matrix is not a perspective projection matrix"
    ret = torch.tensor([[0.5, 0., 0.5], [0., -0.5, 0.5], [0., 0., 1.]], dtype=perspective.dtype, device=perspective.device) \
        @ perspective[:, [0, 1, 3], :3] \
        @ torch.diag(torch.tensor([1, -1, -1], dtype=perspective.dtype, device=perspective.device))
    return ret / ret[:, 2, 2, None, None]


def intrinsics_to_perspective(intrinsics: Tensor, near: float | Tensor, far: float | Tensor) -> Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix
    NOTE: not work for tile-shifting intrinsics currently

    ## Parameters
        intrinsics (Tensor): [..., 3, 3] OpenCV intrinsics matrix
        near (float | Tensor): [...] near plane to clip
        far (float | Tensor): [...] far plane to clip
    ## Returns
        (Tensor): [..., 4, 4] OpenGL perspective matrix
    """
    device, dtype = intrinsics.device, intrinsics.dtype
    batch_shape = intrinsics.shape[:-2]
    if not isinstance(near, Tensor):
        near = torch.full(batch_shape, near, dtype=dtype, device=device)
    if not isinstance(far, Tensor):
        far = torch.full(batch_shape, far, dtype=dtype, device=device)
    m = torch.tensor([[2, 0, -1], [0, -2, 1], [0, 0, 1]], dtype=dtype, device=device) @ intrinsics @ torch.diag(torch.tensor([1, -1, -1], dtype=dtype, device=device))
    perspective = torch.cat([
        torch.cat([m[..., :2, :], torch.zeros((*batch_shape, 2, 1), dtype=dtype, device=device)], dim=-1),
        torch.cat([torch.zeros((*batch_shape, 1, 2), dtype=dtype, device=device), ((near / far + 1) / (near / far - 1))[..., None, None], (2. * near / (near / far - 1))[..., None, None]], dim=-1),
        torch.tensor([0., 0., -1, 0], dtype=dtype, device=device).expand(*batch_shape, 1, 4)
    ], dim=-2)
    return perspective


def make_affine_matrix(M: Tensor, t: Tensor):
    """
    Make an affine transformation matrix from a linear matrix and a translation vector.

    ## Parameters
        M (Tensor): [..., D, D] linear matrix (rotation, scaling or general deformation)
        t (Tensor): [..., D] translation vector

    ## Returns
        Tensor: [..., D + 1, D + 1] affine transformation matrix
    """
    return torch.cat([
        torch.cat([M, t[..., None]], dim=-1),
        torch.tensor([0, 0, 0, 1], dtype=M.dtype, device=M.device).expand(*M.shape[:-2], 1, 4)
    ], dim=-2)


def view_look_at(eye: Tensor, look_at: Tensor, up: Tensor) -> Tensor:
    """
    Get OpenGL view matrix looking at something

    ## Parameters
        eye (Tensor): [..., 3] the eye position
        look_at (Tensor): [..., 3] the position to look at
        up (Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (Tensor): [..., 4, 4], view matrix
    """
    z = eye - look_at
    x = torch.cross(up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-2)
    R = F.normalize(R, dim=-1)
    t = -torch.matmul(R, eye[..., None])
    return make_affine_matrix(R, t.squeeze(-1))


def extrinsics_look_at(eye: Tensor, look_at: Tensor, up: Tensor) -> Tensor:
    """
    Get OpenCV extrinsics matrix looking at something

    ## Parameters
        eye (Tensor): [..., 3] the eye position
        look_at (Tensor): [..., 3] the position to look at
        up (Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (Tensor): [..., 4, 4], extrinsics matrix
    """
    return view_to_extrinsics(view_look_at(eye, look_at, up))


def extrinsics_to_view(extrinsics: Tensor) -> Tensor:
    """
    OpenCV camera extrinsics to OpenGL view matrix

    ## Parameters
        extrinsics (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix

    ## Returns
        (Tensor): [..., 4, 4] OpenGL view matrix
    """
    return extrinsics * torch.tensor([1, -1, -1, 1], dtype=extrinsics.dtype, device=extrinsics.device)[:, None]


def view_to_extrinsics(view: Tensor) -> Tensor:
    """
    OpenGL view matrix to OpenCV camera extrinsics

    ## Parameters
        view (Tensor): [..., 4, 4] OpenGL view matrix

    ## Returns
        (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix
    """
    return view  * torch.tensor([1, -1, -1, 1], dtype=view.dtype, device=view.device)[:, None]


def camera_parameters_from_frame_data(frame_data: dict) -> tuple[Tensor, Tensor]:
    """
    Get camera extrinsic and intrinsic parameters from frame data.
    """
    c2w = torch.tensor(frame_data['transform_matrix'], dtype=torch.float32)
    c2w[:3, 1:3] *= -1  # Flip Y and Z axes
    extrinsics = torch.inverse(c2w)
    fov = torch.tensor(frame_data['camera_angle_x'])
    intrinsics = intrinsics_from_fov(fov_x=fov, fov_y=fov)

    return extrinsics, intrinsics


def load_camera_parameters(json_path: str, ind: int, device: str = 'cpu'):
    """
    Load camera extrinsic and intrinsic parameters from transforms.json file.
    
    Args:
        json_path: Path to transforms.json file
        device: Target device for tensors ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing:
        - 'extrinsic': Camera-to-world matrix (4x4)
        - 'intrinsic': Camera intrinsic matrix (3x3)
    """
    import json
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    frame_data = metadata['frames'][ind]
    extrinsic, intrinsic = camera_parameters_from_frame_data(frame_data)
    
    return {
        'extrinsic': extrinsic.to(device),
        'intrinsic': intrinsic.to(device),
    }


def rodrigues_torch(axis, theta, dtype=torch.float32, device=None):
    """
    Axis-angle (Rodrigues) to rotation matrix.

    Inputs:
      axis: (3,) iterable or tensor (will be normalized)
      theta: float or 0-dim tensor (radians)

    Returns:
      R: (3, 3) rotation matrix
    """
    axis = torch.as_tensor(axis, dtype=dtype, device=device)
    axis = axis / axis.norm(p=2)
    ux, uy, uz = axis.unbind(-1)

    K = torch.tensor([[0.0, -uz,  uy],
                      [uz,  0.0, -ux],
                      [-uy, ux,  0.0]], dtype=dtype, device=device)

    I = torch.eye(3, dtype=dtype, device=device)
    theta = torch.as_tensor(theta, dtype=dtype, device=device)
    s, c = torch.sin(theta), torch.cos(theta)

    R = I + s * K + (1.0 - c) * (K @ K)
    return R


def rotate_extrinsics_world_to_cam_batch(T_cw, axis, theta, pivot):
    """
    Rotate world-to-camera extrinsics around a world-space axis/point.

    Inputs:
      T_cw: (N, 4, 4) world->camera extrinsics (X_c = R X_w + t)
      axis: (3,) constant axis (world)
      theta: scalar angle (radians)
      pivot: (3,) constant point p (world)

    Returns:
      T_cw_new: (N, 4, 4) updated extrinsics
    """
    assert T_cw.ndim == 3 and T_cw.shape[-2:] == (4, 4), "T_cw must be (N,4,4)"
    device, dtype = T_cw.device, T_cw.dtype
    N = T_cw.shape[0]

    R_axis = rodrigues_torch(axis, theta, dtype=dtype, device=device)  # (3,3)
    R_axis_T = R_axis.transpose(0, 1)                                  # (3,3)
    I3 = torch.eye(3, dtype=dtype, device=device)
    p = torch.as_tensor(pivot, dtype=dtype, device=device).reshape(3)  # (3,)

    # Decompose batch extrinsics
    R = T_cw[:, :3, :3]   # (N,3,3)
    t = T_cw[:, :3, 3]    # (N,3)

    # R' = R R_axis^T
    R_new = R @ R_axis_T  # (N,3,3)

    # t' = t + R (I - R_axis^T) p
    p_term = (I3 - R_axis_T) @ p  # (3,)
    t_new = t + (R @ p_term)      # (N,3)

    # Assemble homogeneous matrices
    T_new = torch.zeros_like(T_cw)
    T_new[:, :3, :3] = R_new
    T_new[:, :3, 3] = t_new
    T_new[:, 3, 3] = 1.0
    return T_new


def compute_pose_diff_metrics(
    ext_a: Tensor,
    K_a: Tensor,
    ext_b: Tensor,
    K_b: Tensor,
    *,
    ambiguity: bool = False,
) -> dict[str, Tensor]:
    """Standard pose-fidelity metrics between two pose lists.

    Args:
        ext_a, ext_b: ``[B, 4, 4]`` extrinsics.
        K_a, K_b:     ``[B, 3, 3]`` intrinsics.

    Returns:
        Dict with keys (each shape ``[B]``):
          - ``rel_rangle_deg``, ``rel_tangle_deg``, ``rel_fov_deg``, ``cxcy_dist``
          - ``rel_rangle_acc_5deg``, ``rel_rangle_acc_15deg`` — per-pair
            binary indicator ``(rel_rangle_deg < τ)``. When averaged over
            views (e.g. via :func:`emit_per_view_metrics`) this gives the
            fraction-of-views Acc.@τ° defined in evaluation_protocol §4.3.
    """
    R_a, R_b = ext_a[:, :3, :3], ext_b[:, :3, :3]
    t_a, t_b = ext_a[:, :3, 3],  ext_b[:, :3, 3]
    rangle = rotation_angle(R_a, R_b, ambiguity=ambiguity)
    return {
        'rel_rangle_deg':       rangle,
        'rel_tangle_deg':       translation_angle(t_a, t_b, ambiguity=ambiguity),
        'rel_fov_deg':          (fov_from_intrinsics(K_a) - fov_from_intrinsics(K_b)).abs().mean(dim=-1),
        'cxcy_dist':            torch.norm(t_a[:, :2] - t_b[:, :2], dim=-1),
        'rel_rangle_acc_5deg':  (rangle < 5.0).float(),
        'rel_rangle_acc_15deg': (rangle < 15.0).float(),
    }


def emit_per_view_metrics(
    metrics: dict,
    key: str,
    x_flat: Tensor,
    N: int,
    V: int,
    *,
    nan_safe: bool = False,
) -> None:
    """Reshape ``[N*V]`` → ``[N, V]`` and write into ``metrics`` in place:
    ``metrics[key]`` is the per-instance mean over views, plus
    ``metrics[f'{key}_v{i}']`` for each view ``i``.
    """
    v_nv = x_flat.reshape(N, V)
    metrics[key] = v_nv.nanmean(dim=1) if nan_safe else v_nv.mean(dim=1)
    for vi in range(V):
        metrics[f'{key}_v{vi}'] = v_nv[:, vi]


def pairwise_relative_extrinsics(ext: Tensor) -> Tensor:
    """Relative pose ``T_{i->j} = T_j @ inv(T_i)`` for all unordered pairs ``i<j``.

    Args:
        ext: ``[N, V, 4, 4]`` W2C extrinsics. Requires ``V >= 2``.

    Returns:
        ``[N, P, 4, 4]`` relative extrinsics, where ``P = V*(V-1)/2``.
        Pair order matches ``torch.triu_indices(V, V, offset=1)``.
    """
    assert ext.ndim == 4 and ext.shape[-2:] == (4, 4), \
        f"expected [N, V, 4, 4], got {tuple(ext.shape)}"
    N, V = ext.shape[:2]
    assert V >= 2, f"need V >= 2 for pairwise relative pose, got V={V}"
    inv = torch.inverse(ext)                          # [N, V, 4, 4]
    rel = ext[:, None] @ inv[:, :, None]              # [N, V_i, V_j, 4, 4]
    iu, ju = torch.triu_indices(V, V, offset=1, device=ext.device)
    return rel[:, iu, ju]                             # [N, P, 4, 4]


def pairwise_pose_diff_metrics(
    ext_pred: Tensor,
    ext_gt: Tensor,
    *,
    acc_thresholds_deg: tuple = (5.0, 15.0, 30.0),
    valid_views: Optional[Tensor] = None,
    ambiguity: bool = False,
) -> dict[str, Tensor]:
    """Pairwise relative-pose metrics between two extrinsics tensors.

    For each unordered pair of views ``(i, j)`` we form the relative pose
    in both pred and GT, compare them, and average over pairs.

    Args:
        ext_pred, ext_gt: ``[N, V, 4, 4]`` W2C extrinsics. ``V >= 2``.
        acc_thresholds_deg: thresholds for ``Acc.@τ°``, applied to both
            rotation and translation angular errors.
        valid_views: optional ``[N, V]`` boolean mask. A pair ``(i, j)`` is
            kept only if both views are valid; invalid pairs contribute
            NaN and are excluded from the per-instance mean (``nanmean``).
            ``None`` (default) uses every pair.

    Returns:
        Dict of ``[N]`` tensors, mean over the (valid) ``V*(V-1)/2`` pairs:
          - ``pair_rel_rangle_deg``: mean RRE (degrees)
          - ``pair_rel_tangle_deg``: mean angular RTE (degrees)
          - ``pair_rel_rangle_acc_{τ}deg``, ``pair_rel_tangle_acc_{τ}deg``:
            fraction of pairs with the corresponding error below ``τ``.
    """
    assert ext_pred.shape == ext_gt.shape, (ext_pred.shape, ext_gt.shape)
    pred_rel = pairwise_relative_extrinsics(ext_pred)  # [N, P, 4, 4]
    gt_rel   = pairwise_relative_extrinsics(ext_gt)
    N, V = ext_pred.shape[:2]
    P = pred_rel.shape[1]

    R_p = pred_rel[..., :3, :3].reshape(N * P, 3, 3)
    R_g = gt_rel  [..., :3, :3].reshape(N * P, 3, 3)
    t_p = pred_rel[..., :3,  3].reshape(N * P, 3)
    t_g = gt_rel  [..., :3,  3].reshape(N * P, 3)

    rre = rotation_angle(R_p, R_g, ambiguity=ambiguity).reshape(N, P)
    rte = translation_angle(t_p, t_g, ambiguity=ambiguity).reshape(N, P)

    if valid_views is not None:
        assert valid_views.shape == (N, V), \
            f"valid_views must be [N={N}, V={V}], got {tuple(valid_views.shape)}"
        iu, ju = torch.triu_indices(V, V, offset=1, device=ext_pred.device)
        pair_valid = valid_views[:, iu] & valid_views[:, ju]   # [N, P]
        nan = torch.full_like(rre, float('nan'))
        rre = torch.where(pair_valid, rre, nan)
        rte = torch.where(pair_valid, rte, nan)
        rangle_mean = rre.nanmean(dim=1)
        tangle_mean = rte.nanmean(dim=1)
        agg = lambda x: x.nanmean(dim=1)
    else:
        rangle_mean = rre.mean(dim=1)
        tangle_mean = rte.mean(dim=1)
        agg = lambda x: x.mean(dim=1)

    out = {'pair_rel_rangle_deg': rangle_mean, 'pair_rel_tangle_deg': tangle_mean}
    for tau in acc_thresholds_deg:
        tag = f'{tau:g}'.replace('.', 'p')
        # NaN propagation: comparison on NaN -> False, so set NaN explicitly
        # to keep the "fraction over valid pairs" semantics under masking.
        rre_acc = (rre < tau).float()
        rte_acc = (rte < tau).float()
        if valid_views is not None:
            nan_f = torch.full_like(rre_acc, float('nan'))
            rre_acc = torch.where(pair_valid, rre_acc, nan_f)
            rte_acc = torch.where(pair_valid, rte_acc, nan_f)
        out[f'pair_rel_rangle_acc_{tag}deg'] = agg(rre_acc)
        out[f'pair_rel_tangle_acc_{tag}deg'] = agg(rte_acc)
    return out
