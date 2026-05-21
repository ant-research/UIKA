import torch
import numpy as np
import torch.nn.functional as F

from math import cos, sin
from typing import Optional, List
from dreifus.camera import PoseType
from dreifus.matrix import Pose
from dreifus.vector import Vec3


def interpolate_multiview_cameras(c2ws, intrs, frames_per_loop: int = 128):
    """
    Interpolate multi-view camera parameters into a monocular camera sequence.
    
    Args:
        c2ws (torch.Tensor): inverse camera extrinsics, shape (N, 4, 4)
        intrs (torch.Tensor): camera intrinsics, shape (N, 4, 4)
        frames_per_loop (int): number of frames for one camera trajectory loop
    
    Returns:
        tuple: (res_c2ws, res_intrs, nearest_indices)
            - res_c2ws: (N // 16, 4, 4) interpolated inverse camera extrinsics
            - res_intrs: (N // 16, 4, 4) interpolated camera intrinsics
            - nearest_indices: (N // 16,) nearest actually used camera indices
    """
    # Define the interpolation order as a closed loop.
    LOOP_CYCLE = (15, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 14)
    CAMS_PER_FRAME = 16
    
    # Compute the number of frames.
    num_frames = c2ws.shape[0] // CAMS_PER_FRAME
    device = c2ws.device
    dtype = c2ws.dtype
    
    # Precompute all interpolation parameters.
    frame_indices = torch.arange(num_frames, device=device, dtype=torch.float32)
    idx_rates = (frame_indices / frames_per_loop) * CAMS_PER_FRAME
    idxs = idx_rates.long()
    rates = idx_rates - idxs  # (num_frames,)
    
    # Handle loop boundaries.
    idxs = idxs % len(LOOP_CYCLE)
    next_idxs = (idxs + 1) % len(LOOP_CYCLE)
    
    # Get the corresponding camera indices.
    cam_indices1 = torch.tensor([LOOP_CYCLE[i] for i in idxs.tolist()], device=device)
    cam_indices2 = torch.tensor([LOOP_CYCLE[i] for i in next_idxs.tolist()], device=device)
    
    # Compute actual array indices.
    base_frame_indices = torch.arange(num_frames, device=device) * CAMS_PER_FRAME
    actual_indices1 = base_frame_indices + cam_indices1
    actual_indices2 = base_frame_indices + cam_indices2
    
    # Gather all needed camera parameters in batch.
    c2w1_batch = c2ws[actual_indices1]  # (num_frames, 4, 4)
    c2w2_batch = c2ws[actual_indices2]  # (num_frames, 4, 4)
    intr1_batch = intrs[actual_indices1]  # (num_frames, 4, 4)
    intr2_batch = intrs[actual_indices2]  # (num_frames, 4, 4)
    
    # Interpolate in batch.
    res_c2ws = _batch_c2w_interpolate(c2w1_batch, c2w2_batch, rates)
    res_intrs = _batch_intr_interpolate(intr1_batch, intr2_batch, rates)

    # Nearest indices among the actually used cameras.
    nearest_indices = torch.where(rates < 0.5, actual_indices1, actual_indices2)
    
    return res_c2ws, res_intrs, nearest_indices

def _batch_c2w_interpolate(c2w1_batch, c2w2_batch, rates):
    """
    Batch-interpolate inverse camera extrinsics with quaternion slerp.
    
    Args:
        c2w1_batch (torch.Tensor): first inverse extrinsics batch, shape (B, 4, 4)
        c2w2_batch (torch.Tensor): second inverse extrinsics batch, shape (B, 4, 4)
        rates (torch.Tensor): interpolation ratios, shape (B,)
    
    Returns:
        torch.Tensor: interpolated result, shape (B, 4, 4)
    """
    batch_size = c2w1_batch.shape[0]
    device = c2w1_batch.device
    dtype = c2w1_batch.dtype
    
    # Separate rotation and translation.
    rot1 = c2w1_batch[:, :3, :3]  # (B, 3, 3)
    trans1 = c2w1_batch[:, :3, 3]  # (B, 3)
    rot2 = c2w2_batch[:, :3, :3]  # (B, 3, 3)
    trans2 = c2w2_batch[:, :3, 3]  # (B, 3)
    
    # Convert rotation matrices to quaternions.
    quat1 = _batch_matrix_to_quaternion(rot1)  # (B, 4)
    quat2 = _batch_matrix_to_quaternion(rot2)  # (B, 4)
    
    # Spherical linear interpolation.
    interpolated_quat = _batch_slerp(quat1, quat2, rates)  # (B, 4)
    
    # Convert quaternions back to rotation matrices.
    interpolated_rot = _batch_quaternion_to_matrix(interpolated_quat)  # (B, 3, 3)
    
    # Linearly interpolate translation.
    rates_expanded = rates.unsqueeze(-1)  # (B, 1)
    interpolated_trans = trans1 + rates_expanded * (trans2 - trans1)  # (B, 3)
    
    # Assemble the result.
    result = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
    result[:, :3, :3] = interpolated_rot
    result[:, :3, 3] = interpolated_trans
    result[:, 3, 3] = 1.0
    
    return result

def _batch_intr_interpolate(intr1_batch, intr2_batch, rates):
    """
    Batch-linear-interpolate camera intrinsics.
    
    Args:
        intr1_batch (torch.Tensor): first intrinsics batch, shape (B, 4, 4)
        intr2_batch (torch.Tensor): second intrinsics batch, shape (B, 4, 4)
        rates (torch.Tensor): interpolation ratios, shape (B,)
    
    Returns:
        torch.Tensor: interpolated result, shape (B, 4, 4)
    """
    rates_expanded = rates.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    return intr1_batch + rates_expanded * (intr2_batch - intr1_batch)

def _batch_matrix_to_quaternion(matrices):
    """
    Batch-convert 3x3 rotation matrices to quaternions [w, x, y, z].
    
    Args:
        matrices (torch.Tensor): rotation matrix batch, shape (B, 3, 3)
    
    Returns:
        torch.Tensor: quaternion batch, shape (B, 4)
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    # Compute traces.
    trace = torch.einsum('bii->b', matrices)  # (B,)
    
    # Initialize quaternions.
    quats = torch.empty(batch_size, 4, device=device, dtype=dtype)
    
    # Handle the trace > 0 case.
    cond_positive = trace > 0
    if cond_positive.any():
        indices_pos = torch.where(cond_positive)[0]
        mats_pos = matrices[indices_pos]
        traces_pos = trace[indices_pos]
        
        s = torch.sqrt(traces_pos + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (mats_pos[:, 2, 1] - mats_pos[:, 1, 2]) / s
        qy = (mats_pos[:, 0, 2] - mats_pos[:, 2, 0]) / s
        qz = (mats_pos[:, 1, 0] - mats_pos[:, 0, 1]) / s
        
        quats[indices_pos, 0] = qw
        quats[indices_pos, 1] = qx
        quats[indices_pos, 2] = qy
        quats[indices_pos, 3] = qz
    
    # Handle the trace <= 0 case.
    cond_negative = ~cond_positive
    if cond_negative.any():
        indices_neg = torch.where(cond_negative)[0]
        mats_neg = matrices[indices_neg]
        
        # Find the largest diagonal element.
        diag_elements = torch.stack([
            mats_neg[:, 0, 0], mats_neg[:, 1, 1], mats_neg[:, 2, 2]
        ], dim=-1)  # (num_neg, 3)
        max_indices = torch.argmax(diag_elements, dim=-1)  # (num_neg,)
        
        # Choose the computation branch based on the largest element.
        for i in range(3):
            mask = (max_indices == i)
            if mask.any():
                sub_indices = indices_neg[mask]
                mats_sub = mats_neg[mask]
                
                if i == 0:  # m[0,0] is largest
                    s = torch.sqrt(1.0 + mats_sub[:, 0, 0] - mats_sub[:, 1, 1] - mats_sub[:, 2, 2]) * 2
                    qw = (mats_sub[:, 2, 1] - mats_sub[:, 1, 2]) / s
                    qx = 0.25 * s
                    qy = (mats_sub[:, 0, 1] + mats_sub[:, 1, 0]) / s
                    qz = (mats_sub[:, 0, 2] + mats_sub[:, 2, 0]) / s
                elif i == 1:  # m[1,1] is largest
                    s = torch.sqrt(1.0 + mats_sub[:, 1, 1] - mats_sub[:, 0, 0] - mats_sub[:, 2, 2]) * 2
                    qw = (mats_sub[:, 0, 2] - mats_sub[:, 2, 0]) / s
                    qx = (mats_sub[:, 0, 1] + mats_sub[:, 1, 0]) / s
                    qy = 0.25 * s
                    qz = (mats_sub[:, 1, 2] + mats_sub[:, 2, 1]) / s
                else:  # m[2,2] is largest
                    s = torch.sqrt(1.0 + mats_sub[:, 2, 2] - mats_sub[:, 0, 0] - mats_sub[:, 1, 1]) * 2
                    qw = (mats_sub[:, 1, 0] - mats_sub[:, 0, 1]) / s
                    qx = (mats_sub[:, 0, 2] + mats_sub[:, 2, 0]) / s
                    qy = (mats_sub[:, 1, 2] + mats_sub[:, 2, 1]) / s
                    qz = 0.25 * s
                
                quats[sub_indices, 0] = qw
                quats[sub_indices, 1] = qx
                quats[sub_indices, 2] = qy
                quats[sub_indices, 3] = qz
    
    # Normalize quaternions.
    return F.normalize(quats, p=2, dim=-1)

def _batch_quaternion_to_matrix(quaternions):
    """
    Batch-convert quaternions to 3x3 rotation matrices.
    
    Args:
        quaternions (torch.Tensor): quaternion batch, shape (B, 4)
    
    Returns:
        torch.Tensor: rotation matrix batch, shape (B, 3, 3)
    """
    batch_size = quaternions.shape[0]
    device = quaternions.device
    dtype = quaternions.dtype
    
    qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Build rotation matrices.
    R = torch.empty(batch_size, 3, 3, device=device, dtype=dtype)
    
    R[:, 0, 0] = 1 - 2 * (qy*qy + qz*qz)
    R[:, 0, 1] = 2 * (qx*qy - qw*qz)
    R[:, 0, 2] = 2 * (qx*qz + qw*qy)
    R[:, 1, 0] = 2 * (qx*qy + qw*qz)
    R[:, 1, 1] = 1 - 2 * (qx*qx + qz*qz)
    R[:, 1, 2] = 2 * (qy*qz - qw*qx)
    R[:, 2, 0] = 2 * (qx*qz - qw*qy)
    R[:, 2, 1] = 2 * (qy*qz + qw*qx)
    R[:, 2, 2] = 1 - 2 * (qx*qx + qy*qy)
    
    return R

def _batch_slerp(quat1, quat2, t):
    """
    Batch spherical linear interpolation for quaternions.
    
    Args:
        quat1 (torch.Tensor): first quaternion batch, shape (B, 4)
        quat2 (torch.Tensor): second quaternion batch, shape (B, 4)
        t (torch.Tensor): interpolation ratios, shape (B,)
    
    Returns:
        torch.Tensor: interpolated result, shape (B, 4)
    """
    # Ensure quaternions are in the same hemisphere.
    dot = torch.sum(quat1 * quat2, dim=-1)  # (B,)
    
    # Find quaternions that need flipping.
    flip_mask = dot < 0
    quat2_flipped = quat2.clone()
    quat2_flipped[flip_mask] = -quat2_flipped[flip_mask]
    dot_flipped = torch.abs(dot)
    
    # Avoid division by zero.
    eps = 1e-8
    dot_clamped = torch.clamp(dot_flipped, -1.0 + eps, 1.0 - eps)
    
    # Compute angles.
    theta = torch.acos(dot_clamped)  # (B,)
    sin_theta = torch.sin(theta)  # (B,)
    
    # Handle small angles.
    small_angle_mask = sin_theta.abs() < eps
    large_angle_mask = ~small_angle_mask
    
    result = torch.empty_like(quat1)
    
    # Large-angle case: use spherical interpolation.
    if large_angle_mask.any():
        indices_large = torch.where(large_angle_mask)[0]
        theta_large = theta[indices_large]
        sin_theta_large = sin_theta[indices_large]
        t_large = t[indices_large]
        
        w1 = torch.sin((1 - t_large) * theta_large) / sin_theta_large
        w2 = torch.sin(t_large * theta_large) / sin_theta_large
        
        result[indices_large] = (w1.unsqueeze(-1) * quat1[indices_large] + 
                                w2.unsqueeze(-1) * quat2_flipped[indices_large])
    
    # Small-angle case: use linear interpolation.
    if small_angle_mask.any():
        indices_small = torch.where(small_angle_mask)[0]
        t_small = t[indices_small]
        result[indices_small] = (quat1[indices_small] + 
                                t_small.unsqueeze(-1) * (quat2_flipped[indices_small] - quat1[indices_small]))
        # Normalize.
        result[indices_small] = F.normalize(result[indices_small], p=2, dim=-1)
    
    return result


def point_around_axis(theta: float,
                      axis: Vec3 = Vec3(0, 0, 1)) -> Vec3:
    """
    Compute a point with unit distance from `axis` that is rotated by `theta` around it.
    It is somewhat arbitrary where `theta=0` lands.
     - (1, 0, 0) -> (0, 0, 1)
     - (0, 1, 0) -> (0, 0, -1)
     - (0, 0, 1) -> (0, 1, 0)

    Computed points via `point_around_axis()` will be centered around the origin.

    Parameters
    ----------
        theta: angle between [0, 2pi) specifying the rotation around the axis
        axis: the axis to rotate around
    """

    axis = Vec3(axis)
    non_parallel = Vec3(0, 1, 0) if axis == Vec3(1, 0, 0) else Vec3(1, 0, 0)
    v = axis.cross(non_parallel).normalize()
    v_rotated = cos(theta) * v + sin(theta) * (np.cross(axis, v)) + (1 - cos(theta)) * axis * (np.dot(axis, v))

    return v_rotated


def circle_around_axis(n_poses: int,
                       axis=Vec3(0, 0, 1),
                       up: Vec3 = Vec3(0, 0, 1),
                       move=Vec3(),
                       distance: float = 1.0,
                       distance_end: Optional[float] = None,
                       theta_from: float = 0,
                       theta_to: float = 2 * np.pi,
                       look_at: Vec3 = Vec3(0, 0, 0)) -> List[Pose]:
    """
    Computes `n_poses` many camera poses (cam2world) that circle with distance `distance` around the specified `axis`
    that is moved by `move`.
    Per default, one full circle is computed. With `theta_from` and `theta_to` one can specify parts of the circle
    or even multiple circulations around the axis.

    Parameters
    ----------
        n_poses:
            how many poses should be computed
        axis:
            The axis (direction) around which we rotate
        up:
            which direction should be up for the camera
        move:
            the location of the axis
        distance:
            distance of the pose locations from the axis
        distance_end:
            if specified, `distance` will be interpreted as a start distance for the first pose and distance_end
            defines the distance from the axis for the last pose. Distances in for poses in between are linearly
            interpolated. This gives a spiraling effect.
        theta_from:
            orientation of the first pose
        theta_to:
            orientation of the last pose
        look_at:
            all circle poses will look at the specified point in space. Per default, this is the origin
    """

    if distance_end is None:
        distance_end = distance
        distance_start = distance
    else:
        distance_start = distance

    poses = []
    for i_pose in range(n_poses):
        alpha = i_pose / n_poses
        theta = theta_from + alpha * (theta_from - theta_to)
        distance = distance_start + alpha * (distance_end - distance_start)
        location = distance * point_around_axis(theta, axis=axis)

        pose = Pose(pose_type=PoseType.CAM_2_WORLD)
        location += move
        pose.set_translation(location)
        pose.look_at(look_at, up=up)

        poses.append(pose)

    return poses


def ellipse_around_axis(n_poses: int,
                        a_len: float,
                        b_len: float,
                        axis: Vec3 = Vec3(0, 0, 1),
                        up: Vec3 = Vec3(0, 0, 1),
                        move: Vec3 = Vec3(0, 0, 0),
                        theta_from: float = 0,
                        theta_to: float = 2 * np.pi,
                        look_at: Vec3 = Vec3(0, 0, 0)) -> List[Pose]:
    """
    Compute camera poses along an elliptical trajectory around the given axis.
    
    Parameters
    ----------
        n_poses: number of generated poses
        a_len: ellipse semi-axis length along the first orthogonal basis
        b_len: ellipse semi-axis length along the second orthogonal basis
        axis: rotation axis direction
        up: camera up direction
        move: ellipse center position
        theta_from: starting angle in radians
        theta_to: ending angle in radians
        look_at: camera look-at point
    """
    axis = Vec3(axis).normalize()
    
    # Find two orthogonal basis vectors u and v in the plane perpendicular to axis.
    # This follows the same idea as point_around_axis.
    non_parallel = Vec3(0, 1, 0) if axis == Vec3(1, 0, 0) else Vec3(1, 0, 0)
    u = axis.cross(non_parallel).normalize()
    v = Vec3(np.cross(axis, u)).normalize() # Second orthogonal basis.

    poses = []
    for i_pose in range(n_poses):
        # Linearly interpolate the current angle.
        alpha = i_pose / n_poses if n_poses > 1 else 0
        theta = theta_from + alpha * (theta_from - theta_to)

        # Ellipse parametric equation: P = a * cos(theta) * u + b * sin(theta) * v.
        # This keeps points elliptically distributed in the plane perpendicular to axis.
        location = a_len * cos(theta) * u + b_len * sin(theta) * v
        location += move

        pose = Pose(pose_type=PoseType.CAM_2_WORLD)
        pose.set_translation(location)
        pose.look_at(look_at, up=up)

        poses.append(pose)

    return poses
