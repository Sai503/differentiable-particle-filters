import numpy as np
import torch
from utils.data_utils_kitti import wrap_angle

def compute_sq_distance(a, b, state_step_sizes):
    """
    Compute the scaled squared distance between tensors a and b.
    
    Args:
        a (torch.Tensor): Tensor of shape (..., D)
        b (torch.Tensor): Tensor of the same shape as a.
        state_step_sizes: A list or tensor of scaling factors of length D.
    
    Returns:
        torch.Tensor: The computed squared distance.
    """
    result = 0.0
    for i in range(a.shape[-1]):
        diff = a[..., i] - b[..., i]
        # For the angle dimension (assumed to be index 2), wrap the difference.
        if i == 2:
            diff = wrap_angle(diff)
        result = result + (diff / state_step_sizes[i]) ** 2
    return result

def atan2(x, y, epsilon=1.0e-12):
    """
    Compute the elementwise arctan2 in a numerically safe manner.
    
    This function adds a small epsilon to x and y when they are zero to avoid division by zero.
    
    Args:
        x (torch.Tensor): The first input tensor.
        y (torch.Tensor): The second input tensor.
        epsilon (float): A small value to add to avoid zero divisions.
    
    Returns:
        torch.Tensor: The computed angles (in radians).
    """
    # Avoid zeros to prevent division issues
    x_safe = torch.where(x == 0, x + epsilon, x)
    y_safe = torch.where(y == 0, y + epsilon, y)
    # Note: torch.atan2 expects (y, x)
    return torch.atan2(y_safe, x_safe)
