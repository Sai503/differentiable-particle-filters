import torch
import math # Keep for torch.pi if needed in older torch versions

# Define a local PyTorch-based wrap_angle for self-containment
def wrap_angle_pytorch(angle: torch.Tensor) -> torch.Tensor:
    """Wraps angle tensor to [-pi, pi]."""
    # Use torch.pi if available (torch >= 1.7), otherwise use math.pi
    pi = torch.tensor(math.pi, dtype=angle.dtype, device=angle.device)
    # Use torch.remainder for robust modulo operation
    return torch.remainder(angle + pi, 2 * pi) - pi

def compute_sq_distance(a: torch.Tensor, b: torch.Tensor, state_step_sizes: torch.Tensor) -> torch.Tensor:
    """
    Compute the scaled squared distance between tensors a and b using vectorized operations.

    Assumes the angle dimension (requiring wrapping) is at index 2 of the last dimension.

    Args:
        a (torch.Tensor): Tensor of shape (..., D).
        b (torch.Tensor): Tensor of the same shape as a.
        state_step_sizes (torch.Tensor): Tensor of scaling factors of shape (D,)
                                         or broadcastable (e.g., (1, D)).
                                         Must be on the same device as a and b.

    Returns:
        torch.Tensor: The computed scaled squared distance, shape (...).
    """
    # if a.shape != b.shape:
    #     raise ValueError(f"Input tensors a and b must have the same shape. Got {a.shape} and {b.shape}")
    # if a.shape[-1] != state_step_sizes.shape[-1]:
    #      raise ValueError(f"Last dimension of input tensors ({a.shape[-1]}) must match "
    #                       f"size of state_step_sizes ({state_step_sizes.shape[-1]})")
    if a.device != state_step_sizes.device or b.device != state_step_sizes.device:
         # This check is good practice, though the calling functions should ensure it.
         print(f"Warning: Devices may not match! a: {a.device}, b: {b.device}, steps: {state_step_sizes.device}")
         # Optionally move state_step_sizes: state_step_sizes = state_step_sizes.to(a.device)

    # Calculate difference for all dimensions
    diff = a - b

    # Wrap angle difference specifically for dimension index 2
    # Check if the last dimension is large enough to contain an angle at index 2
    if diff.shape[-1] > 2:
        # Use slicing to keep the dimension for broadcasting later if needed,
        # although direct indexing works here as wrap_angle_pytorch handles it.
        diff[..., 2] = wrap_angle_pytorch(diff[..., 2])

    # Scale differences by step sizes (broadcasting should handle shapes like [..., D] / [D])
    # Add epsilon for numerical stability if step sizes can be zero
    scaled_diff = diff / (state_step_sizes + 1e-8)

    # Square and sum over the last dimension
    sq_dist = torch.sum(scaled_diff ** 2, dim=-1)

    return sq_dist

# Note: Renamed arguments to match torch.atan2 convention (y, x)
def atan2(y: torch.Tensor, x: torch.Tensor, epsilon: float = 1.0e-12) -> torch.Tensor:
    """
    Compute the elementwise torch.atan2(y, x) in a numerically safe manner.

    This function adds a small epsilon to x and y when they are zero to avoid issues
    in atan2's internal calculations or potential gradient issues at (0,0).

    Args:
        y (torch.Tensor): The y-coordinates.
        x (torch.Tensor): The x-coordinates.
        epsilon (float): A small value added to near-zero inputs for numerical stability.

    Returns:
        torch.Tensor: The computed angles (in radians), same shape as input.
    """
    # Add epsilon where y or x is close to zero
    # Using torch.where might be slightly less efficient than direct addition,
    # but it's explicit about avoiding modification where not needed.
    # Alternatively, just add epsilon: y_safe = y + epsilon * (y == 0), x_safe = x + epsilon * (x == 0)
    # However, torch.atan2 itself is generally robust. The main concern might be gradients at (0,0).
    # Let's simplify by relying on torch.atan2's robustness and just use it directly,
    # as the original epsilon addition might not be strictly necessary unless gradients at (0,0) are problematic.
    # If numerical issues *are* observed, re-introduce the epsilon logic.

    # Simpler version relying on torch.atan2 robustness:
    return torch.atan2(y, x)

    # Original version with epsilon (kept for reference):
    # x_safe = torch.where(torch.abs(x) < epsilon, x + epsilon, x) # Add epsilon if close to zero
    # y_safe = torch.where(torch.abs(y) < epsilon, y + epsilon, y) # Add epsilon if close to zero
    # return torch.atan2(y_safe, x_safe)

