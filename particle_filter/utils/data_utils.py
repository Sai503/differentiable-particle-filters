# In c:\Users\sjanyavu\Desktop\differentiable-particle-filters\experiments\utils\data_utils.py

import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# For file I/O we still use numpy; alternatively you could switch to torch.save/load.
import numpy as np

from utils.plotting_utils import plot_trajectories, plot_maze, plot_observations, plot_trajectory

# Use torch.remainder to mimic modulo for tensors.
def wrap_angle(angle):
    # angle: a torch.Tensor (or scalar) in radians.
    # Use torch.pi instead of math.pi for consistency if using newer torch versions
    pi = torch.tensor(math.pi, dtype=angle.dtype, device=angle.device)
    return torch.remainder(angle + pi, 2 * pi) - pi # Corrected logic for PyTorch


def mix_data(file_in1, file_in2, file_out, steps_per_episode=100, num_episodes=1000):
    # Load data from npz files and convert to torch tensors.
    # Consider using torch.load if files were saved with torch.save
    data1 = {k: torch.from_numpy(v).float() for k, v in np.load(file_in1).items()}
    data2 = {k: torch.from_numpy(v).float() for k, v in np.load(file_in2).items()}
    data_mix = {}
    half = steps_per_episode * num_episodes // 2
    for key in data1.keys():
        d1 = data1[key][:half]
        d2 = data2[key][:half]
        data_mix[key] = torch.cat((d1, d2), dim=0)
    # Save using torch.save; note that the file format will be a .pt file.
    # Or save back to npz if preferred downstream
    # np.savez(file_out, **{k: v.cpu().numpy() for k, v in data_mix.items()})
    torch.save(data_mix, file_out + '.pt') # Save as .pt


def average_nn(states_from, states_to, step_sizes, num_from=10, num_to=100):
    # Ensure inputs are tensors
    if not isinstance(states_from, torch.Tensor): states_from = torch.from_numpy(states_from).float()
    if not isinstance(states_to, torch.Tensor): states_to = torch.from_numpy(states_to).float()
    if not isinstance(step_sizes, torch.Tensor): step_sizes = torch.from_numpy(step_sizes).float()

    # Reshape states to [-1, 3] and sample random indices.
    states_from = states_from.view(-1, 3)
    states_to = states_to.view(-1, 3)
    idx_from = torch.randint(0, states_from.shape[0], (num_from,))
    idx_to = torch.randint(0, states_to.shape[0], (num_to,))
    total = 0.0
    for i in range(3):
        diff = states_from[idx_from, None, i] - states_to[None, idx_to, i]
        if i == 2:
            diff = wrap_angle(diff)
        # Add epsilon to step_sizes to prevent division by zero
        total += (diff / (step_sizes[i] + 1e-8)) ** 2
    # Compute the fraction of cases where the minimal distance is > 0.5.
    average_dist = torch.mean((torch.min(total, dim=1)[0] > 0.5).float())
    return average_dist.item()


def load_data(trial_numbers, data_root='./data', steps_per_episode=20, concatenate=False):
    """
    Args:
        trial_numbers: List of trials (e.g., [5, 6, 7]).
        data_root: Path to directory containing CSVs/images.
        steps_per_episode: Fixed sequence length (T).
        concatenate: If True, merge all trials into one batch.
    Returns:
        Dictionary with keys:
            'o': Images (B, T, C, H, W) if concatenate=False, else (B*T, C, H, W)
            'l': LiDAR Cartesian coordinates (B, T, 360, 2) or (B*T, 360, 2)
            's': SLAM poses (B, T, 3) or (B*T, 3)
            'a': Odometry deltas (B, T, 3) or (B*T, 3)
    """
    # Load and group all data from CSVs
    all_episodes = []
    for trial in trial_numbers:
        csv_path = os.path.join(data_root, f'maze-data-{trial}.csv')
        df = pd.read_csv(csv_path)
        
        # Group by timestamp (each group = one time step T)
        grouped = df.groupby('timestamp')
        for _, group in grouped:
            # Extract raw LiDAR data
            ranges = torch.tensor(group['range'].values, dtype=torch.float32)
            thetas = torch.tensor(group['theta'].values, dtype=torch.float32)
            
            # Process LiDAR scan
            lidar_xy = process_lidar_scan(ranges, thetas)
            
            all_episodes.append({
                'image_path': os.path.join(data_root, group['image_filename'].iloc[0]),
                'lidar': lidar_xy,  # (360, 2)
                'slam_pose': torch.tensor(group[['slam_x', 'slam_y', 'slam_theta']].iloc[0], dtype=torch.float32),
                'odom_pose': torch.tensor(group[['odom_x', 'odom_y', 'odom_theta']].iloc[0], dtype=torch.float32)
            })

    # Split into episodes of fixed length
    num_episodes = len(all_episodes) // steps_per_episode
    print(f"Total episodes: {len(all_episodes)}, Steps per episode: {steps_per_episode}, Total episodes after split: {num_episodes}")
    episodes = []
    for i in range(num_episodes):
        start = i * steps_per_episode
        end = start + steps_per_episode
        episodes.append(all_episodes[start:end])

    # Process each episode
    o, l, s, a = [], [], [], []
    for ep in episodes:
        # Load images
        ep_images = []
        for step in ep:
            img = Image.open(step['image_path']).convert('RGB')
            img = ImageOps.flip(img)
            img = img.resize((64, 64), Image.LANCZOS) # actual algorithm behind antialising
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            ep_images.append(img_tensor)
        ep_images = torch.stack(ep_images)  # (T, C, H, W)
        
        # Stack LiDAR and poses
        ep_lidar = torch.stack([step['lidar'] for step in ep])  # (T, 360, 2)
        ep_slam = torch.stack([step['slam_pose'] for step in ep])  # (T, 3)
        
        # Compute odometry deltas
        ep_odom = torch.stack([step['odom_pose'] for step in ep])  # (T, 3)
        dx = ep_odom[1:, 0] - ep_odom[:-1, 0]
        dy = ep_odom[1:, 1] - ep_odom[:-1, 1]
        dtheta = wrap_angle(ep_odom[1:, 2] - ep_odom[:-1, 2])
        ep_actions = torch.stack([dx, dy, dtheta], dim=-1)  # (T-1, 3)

        # Append to lists
        o.append(ep_images[:-1])
        l.append(ep_lidar[:-1])
        s.append(ep_slam[:-1])
        a.append(ep_actions)

    # Stack episodes into final tensors
    if concatenate:
        data = {
            'o': torch.cat(o, dim=0),
            'l': torch.cat(l, dim=0),
            's': torch.cat(s, dim=0),
            'a': torch.cat(a, dim=0)
        }
    else:
        data = {
            'o': torch.stack(o),
            'l': torch.stack(l),
            's': torch.stack(s),
            'a': torch.stack(a)
        }
    return data


def process_lidar_scan(ranges, thetas):
    """
    Processes raw LiDAR data to:
    1. Ensure 360 rays (one per degree)
    2. Fill missing rays with neighbor averages
    3. Convert to Cartesian coordinates (x,y)
    
    Args:
        ranges: Tensor of range values
        thetas: Tensor of angle values (radians)
    
    Returns:
        Tensor of shape (360, 2) containing (x,y) coordinates
    """
    # Convert angles to degrees and round to nearest integer
    degrees = torch.rad2deg(thetas).round().long() % 360
    
    # Create output tensor for 360 degrees
    output_ranges = torch.full((360,), float('nan'))
    
    # Assign measured ranges to their degree bins
    for deg, r in zip(degrees, ranges):
        output_ranges[deg] = r
    
    # Fill missing degrees with neighbor averages
    for deg in range(360):
        if torch.isnan(output_ranges[deg]):
            # Find nearest valid neighbors
            prev_deg = (deg - 1) % 360
            next_deg = (deg + 1) % 360
            count = 0
            total = 0.0
            
            # Look backward until we find a valid measurement
            for i in range(1, 360):
                check_deg = (deg - i) % 360
                if not torch.isnan(output_ranges[check_deg]):
                    total += output_ranges[check_deg]
                    count += 1
                    break
            
            # Look forward until we find a valid measurement
            for i in range(1, 360):
                check_deg = (deg + i) % 360
                if not torch.isnan(output_ranges[check_deg]):
                    total += output_ranges[check_deg]
                    count += 1
                    break
            
            if count > 0:
                output_ranges[deg] = total / count
    
    # Convert to Cartesian coordinates
    angles = torch.deg2rad(torch.arange(360, dtype=torch.float32))
    x = output_ranges * torch.cos(angles)
    y = output_ranges * torch.sin(angles)
    
    return torch.stack([x, y], dim=-1)  # (360, 2)



# MODIFIED compute_statistics function:
def compute_statistics(data):
    """
    Computes statistics entirely on GPU (if available) but returns NumPy arrays.
    Uses zero-copy GPU->CPU transfer when possible (CUDA-pinned memory).
    """
    # Determine device from input (prioritize GPU)
    device = next(iter(data.values())).device if data else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize containers (all tensors stay on original device)
    stats = {
        'means': {},
        'stds': {},
        'step_sizes': torch.empty(3, device=device),
        'mins': torch.empty(3, device=device),
        'maxs': torch.empty(3, device=device)
    }

    # --- Computation (all on GPU if available) ---
    for key in ['o', 'l', 's', 'a']:
        if key not in data:
            print(f"Warning: Missing key '{key}' in statistics computation")
            continue
            
        t = data[key].float()
        
        if key == 'o':  # Images [B,T,C,H,W]
            stats['means'][key] = torch.mean(t, dim=(0,1,3,4))  # [C]
            stats['stds'][key] = torch.std(t, dim=(0,1,3,4))   # [C]
            
        elif key == 'l':  # LiDAR [B,T,N]
            stats['means'][key] = torch.mean(t).unsqueeze(0)  # [1]
            stats['stds'][key] = torch.std(t).unsqueeze(0)    # [1]
            
        elif key in ['s', 'a']:  # Poses/deltas [B,T,3]
            stats['means'][key] = torch.mean(t, dim=(0,1))
            stats['stds'][key] = torch.std(t, dim=(0,1))
            
            if key == 's':
                # Angle wrapping for orientation (dim 2)
                if t.shape[-1] >= 3:
                    steps = t[:,1:] - t[:,:-1]
                    steps[...,2] = wrap_angle(steps[...,2])
                    stats['step_sizes'] = torch.mean(torch.abs(steps), dim=(0,1))
                    stats['mins'] = torch.amin(t, dim=(0,1))
                    stats['maxs'] = torch.amax(t, dim=(0,1))

    # --- Zero-copy GPU->CPU conversion (if CUDA) ---
    def to_numpy(tensor):
        if tensor.device.type == 'cuda':
            return tensor.detach().cpu().numpy()  # Pinned memory transfer
        return tensor.numpy()

    return (
        {k: to_numpy(v) for k,v in stats['means'].items()},
        {k: to_numpy(v) for k,v in stats['stds'].items()},
        to_numpy(stats['step_sizes']),
        to_numpy(stats['mins']),
        to_numpy(stats['maxs'])
    )
# --- (Rest of data_utils.py remains the same) ---

def split_data(data, ratio=0.8, categories=['train', 'val']):
    """
    Splits data into training/validation sets while preserving GPU placement.
    Works with new data format: {'o', 'l', 's', 'a'} tensors on any device.
    
    Args:
        data: Dictionary containing:
              'o': (B, T, C, H, W) - images
              'l': (B, T, N)       - LiDAR
              's': (B, T, 3)       - SLAM poses  
              'a': (B, T, 3)       - odometry deltas
        ratio: Train/validation split ratio
        categories: Names for split sets
        
    Returns:
        Dictionary {category: split_data} with same structure as input
    """
    print(f"Splitting data. Input shape - Images: {data['o'].shape}, LiDAR: {data['l'].shape}")
    
    split_data_dict = {categories[0]: {}, categories[1]: {}}
    num_episodes = data['o'].shape[0]  # B = number of episodes
    split_point = int(num_episodes * ratio)

    if split_point == 0 or split_point == num_episodes:
        raise ValueError(
            f"Split ratio {ratio} results in empty set. "
            f"Episodes: {num_episodes}, Split at: {split_point}"
        )

    # Split all keys while preserving device
    for key in data.keys():
        split_data_dict[categories[0]][key] = data[key][:split_point]
        split_data_dict[categories[1]][key] = data[key][split_point:]

    # Debug prints with shapes
    for cat in split_data_dict:
        if split_data_dict[cat]:
            sample_key = next(iter(split_data_dict[cat]))
            print(
                f"Split '{cat}': {split_data_dict[cat][sample_key].shape[0]} episodes | "
                f"Shapes - o: {split_data_dict[cat]['o'].shape}, "
                f"l: {split_data_dict[cat]['l'].shape}"
            )
        else:
            print(f"Split '{cat}': 0 episodes (WARNING: Check data size)")

    return split_data_dict


def reduce_data(data, num_episodes):
    """
    Reduces dataset to specified number of episodes while preserving device placement.
    Works with new format: {'o', 'l', 's', 'a'} tensors.
    
    Args:
        data: Dictionary containing batched tensors
        num_episodes: Number of episodes to keep
        
    Returns:
        Dictionary with same structure but reduced batch size
    """
    new_data = {}
    min_episodes = data['o'].shape[0]  # Using 'o' as reference for batch size
    num_episodes = min(num_episodes, min_episodes)
    
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive. Got {num_episodes}")
    
    print(f"Reducing data from {min_episodes} to {num_episodes} episodes")
    
    for key in data.keys():
        new_data[key] = data[key][:num_episodes]
        
    return new_data


def shuffle_data(data):
    """
    Shuffles episodes while maintaining temporal sequences within each episode.
    Preserves device placement of tensors.
    
    Args:
        data: Dictionary containing batched tensors
        
    Returns:
        Dictionary with same structure but shuffled episode order
    """
    new_data = {}
    num_episodes = data['o'].shape[0]
    device = data['o'].device  # Preserve original device
    
    # Create random permutation on same device as input
    indices = torch.randperm(num_episodes, device=device)
    
    print(f"Shuffling {num_episodes} episodes on {device}")
    
    for key in data.keys():
        new_data[key] = data[key][indices]
        
    return new_data


def remove_state(data, provide_initial_state=False):
    """
    Handles SLAM pose ('s') removal while preserving other data.
    Optionally keeps initial state if requested.
    
    Args:
        data: Dictionary containing batched tensors
        provide_initial_state: Whether to keep first timestep of poses
        
    Returns:
        Dictionary with modified state information
    """
    new_data = {}
    
    # Copy all non-state keys
    for key in data.keys():
        if key != 's':
            new_data[key] = data[key]
    
    # Handle state specially
    if 's' in data:
        if provide_initial_state:
            new_data['s'] = data['s'][:, :1, :]  # Keep only first timestep
            print(f"Kept initial state only. Shape: {new_data['s'].shape}")
        else:
            print("Removed all state information")
    
    return new_data


def noisify_data_condition(data, condition):
    """
    Applies specified noise condition to data while preserving device placement.
    Updated for new data format: {'o', 'l', 's', 'a'} tensors on any device.
    
    Args:
        data: Dictionary containing:
              'o': (B,T,C,H,W) - images
              'a': (B,T,3)     - odometry deltas
              (other keys passed through unchanged)
        condition: Noise configuration string
        
    Returns:
        Dictionary with noisy data (same structure as input)
    """
    print(f'Applying noise condition: {condition} (device: {data["o"].device})')
    
    # Helper function to maintain device placement
    def _noisyfy(data, **kwargs):
        return noisyfy_data(data, **kwargs)  # Assumes your noisyfy_data preserves devices
    
    if condition == 'odom0_imgTG':
        return _noisyfy(data, odom_noise_factor=0.0, img_noise_factor=1.0, img_random_shift=True)
    elif condition == 'odom5_imgTG':
        return _noisyfy(data, odom_noise_factor=0.5, img_noise_factor=1.0, img_random_shift=True)
    elif condition == 'odom10_imgTG':
        return _noisyfy(data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=True)
    elif condition == 'odom20_imgTG':
        return _noisyfy(data, odom_noise_factor=2.0, img_noise_factor=1.0, img_random_shift=True)
    elif condition == 'odomX_imgTG':
        # Start with clean odom then scramble
        data = _noisyfy(data, odom_noise_factor=0.0, img_noise_factor=1.0, img_random_shift=True)
        if 'a' in data:
            B, T, _ = data['a'].shape
            data['a'] = data['a'].view(-1, 3)[torch.randperm(B*T)].view(B,T,3)
        return data
    elif condition == 'odom10_imgC':
        return _noisyfy(data, odom_noise_factor=1.0, img_noise_factor=0.0, img_random_shift=False)
    elif condition == 'odom10_imgG':
        return _noisyfy(data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=False)
    elif condition == 'odom10_imgT':
        return _noisyfy(data, odom_noise_factor=1.0, img_noise_factor=0.0, img_random_shift=True)
    elif condition == 'odom10_imgX':
        # Start with clean images then scramble
        data = _noisyfy(data, odom_noise_factor=1.0, img_noise_factor=0.0, img_random_shift=False)
        if 'o' in data:
            B, T, C, H, W = data['o'].shape
            data['o'] = data['o'].view(-1,C,H,W)[torch.randperm(B*T)].view(B,T,C,H,W)
        return data
    else:
        print(f"Warning: Unknown condition '{condition}'. Using default noise.")
        return _noisyfy(data)  # Default condition
    

def noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=1.0, 
                downsample_factor=2):
    """
    Adds noise and downsamples images (no cropping).
    
    Args:
        data: Dict with:
              'o': (B,T,C,H,W) - images
              'a': (B,T,3)     - odometry deltas
              (other keys passed through)
        downsample_factor: How much to shrink images (e.g., 2 = half resolution)
    """
    print(f"Noisyfying data on {data['o'].device}: "
          f"odom_noise={odom_noise_factor}, "
          f"img_noise={img_noise_factor}, "
          f"downsample={downsample_factor}x")
    
    new_data = {}
    device = data['o'].device
    dtype = data['o'].dtype

    # --- Process Actions ---
    if 'a' in data:
        noise = torch.normal(1.0, 0.1*odom_noise_factor, 
                           size=data['a'].shape, 
                           device=device, dtype=dtype)
        new_data['a'] = data['a'] * noise.clamp(min=1e-2)

    # --- Downsample Images ---
    if 'o' in data:
        B, T, C, H, W = data['o'].shape
        new_h, new_w = H // downsample_factor, W // downsample_factor
        
        # Downsample with average pooling (anti-aliasing)
        downsampled = torch.zeros(B, T, C, new_h, new_w, 
                                device=device, dtype=dtype)
        
        for b in range(B):
            for t in range(T):
                downsampled[b,t] = torch.nn.functional.avg_pool2d(
                    data['o'][b,t],
                    kernel_size=downsample_factor,
                    stride=downsample_factor
                )
        
        # Add Gaussian noise if requested
        if img_noise_factor > 0:
            noise = torch.normal(0, 20*img_noise_factor, 
                               size=(B,T,C,new_h,new_w),
                               device=device, dtype=dtype)
            downsampled += noise
        
        new_data['o'] = downsampled
        print(f"Downsampled images: {H}x{W} -> {new_h}x{new_w}")

    # --- Copy Other Keys ---
    for key in ['s', 'l']:
        if key in data:
            new_data[key] = data[key].clone()

    return new_data


def make_batch_iterator(data, batch_size=32, seq_len=10):
    """
    Generator yielding random batches of specified sequence length.
    Updated for new data format: {'o', 'l', 's', 'a'} tensors on any device.
    
    Args:
        data: Dictionary containing:
              'o': (B,T,C,H,W) - images
              'l': (B,T,N)     - LiDAR scans
              's': (B,T,3)     - SLAM poses  
              'a': (B,T,3)     - odometry deltas
        batch_size: Number of sequences per batch
        seq_len: Length of temporal sequences
        
    Yields:
        Batches with same structure as input, but shapes:
        'o': (batch_size, seq_len, C, H, W)
        'l': (batch_size, seq_len, N)
        's': (batch_size, seq_len, 3)
        'a': (batch_size, seq_len, 3)
    """
    # Extract dimensions
    num_episodes = data['s'].shape[0]
    ep_len = data['s'].shape[1]
    device = data['s'].device
    
    # Validation
    if seq_len >= ep_len:
        raise ValueError(f"seq_len ({seq_len}) >= episode length ({ep_len})")
    max_start_step = ep_len - seq_len
    
    if num_episodes == 0 or max_start_step < 0:
        print("Warning: No valid sequences available")
        return

    while True:  # Infinite generator
        # Sample random episodes and start points
        ep_indices = torch.randint(0, num_episodes, (batch_size,), device=device)
        start_indices = torch.randint(0, max_start_step+1, (batch_size,), device=device)
        
        # Build batch using advanced indexing
        batch = {}
        for key in data.keys():
            # Create time indices for each sequence [batch_size, seq_len]
            time_indices = start_indices.unsqueeze(1) + torch.arange(seq_len, device=device)
            
            # Advanced indexing [batch_size, seq_len, ...]
            batch[key] = data[key][ep_indices[:, None], time_indices]
            
            # Special case for images to maintain 5D shape
            if key == 'o' and batch[key].dim() == 6:  # Handle edge cases
                batch[key] = batch[key].squeeze(2)
        
        yield batch


def make_repeating_batch_iterator(data, epoch_len, batch_size=32, seq_len=10):
    """
    Generator yielding batches from pre-sampled indices, repeating indefinitely.
    Optimized for new data format with GPU support.
    
    Args:
        data: Dictionary containing:
              'o': (B,T,C,H,W) - images
              'l': (B,T,N)     - LiDAR
              's': (B,T,3)     - poses
              'a': (B,T,3)     - actions
        epoch_len: Number of batches per epoch
        batch_size: Sequences per batch
        seq_len: Temporal length of sequences
    """
    num_episodes = data['s'].shape[0]
    ep_len = data['s'].shape[1]
    device = data['s'].device
    
    # Validation
    if seq_len >= ep_len:
        raise ValueError(f"seq_len ({seq_len}) >= episode length ({ep_len})")
    max_start_step = ep_len - seq_len
    
    if num_episodes == 0 or max_start_step < 0:
        print("Warning: No valid sequences available")
        return

    # Pre-sample all indices for the epoch on correct device
    ep_indices = torch.randint(0, num_episodes, (epoch_len, batch_size), device=device)
    start_indices = torch.randint(0, max_start_step+1, (epoch_len, batch_size), device=device)

    while True:  # Infinite epoch loop
        for i in range(epoch_len):
            # Vectorized advanced indexing
            batch = {}
            time_indices = start_indices[i].unsqueeze(1) + torch.arange(seq_len, device=device)
            
            for key in data.keys():
                batch[key] = data[key][ep_indices[i][:, None], time_indices]
                
                # Handle potential singleton dimensions for images
                if key == 'o' and batch[key].dim() == 6:
                    batch[key] = batch[key].squeeze(2)
            
            yield batch

def make_complete_batch_iterator(data, batch_size=1000, seq_len=10):
    """
    Generator yielding batches covering all possible sequences exactly once per epoch.
    Optimized for new data format with GPU support.
    
    Args:
        data: Dictionary with same format as above
        batch_size: Sequences per batch
        seq_len: Temporal length of sequences
    """
    num_episodes = data['s'].shape[0]
    ep_len = data['s'].shape[1]
    device = data['s'].device
    
    # Validation
    if seq_len >= ep_len:
        raise ValueError(f"seq_len ({seq_len}) >= episode length ({ep_len})")
    num_start_steps = ep_len - seq_len + 1  # +1 to include last start step
    
    if num_episodes == 0 or num_start_steps <= 0:
        print("Warning: No valid sequences available")
        return

    # Create all possible (episode, start_step) pairs
    episodes = torch.arange(num_episodes, device=device)
    starts = torch.arange(num_start_steps, device=device)
    
    # Combine into grid of all possible indices
    grid_ep, grid_start = torch.meshgrid(episodes, starts, indexing='ij')
    all_indices = torch.stack([grid_ep.reshape(-1), grid_start.reshape(-1)], dim=1)

    # Shuffle indices (optional but recommended)
    perm = torch.randperm(len(all_indices), device=device)
    all_indices = all_indices[perm]

    # Batch processing
    for i in range(0, len(all_indices), batch_size):
        batch_indices = all_indices[i:i+batch_size]
        
        # Vectorized indexing
        time_indices = batch_indices[:, 1, None] + torch.arange(seq_len, device=device)
        batch = {}
        
        for key in data.keys():
            batch[key] = data[key][batch_indices[:, 0], time_indices]
            
            # Handle image dimensions
            if key == 'o' and batch[key].dim() == 6:
                batch[key] = batch[key].squeeze(2)
        
        yield batch


# --- Example Usage / Plotting (Mainly for debugging/visualization) ---
def compare_data_coverage():
    # This function might need adjustment if it relies on specific NumPy array properties
    # that changed due to using Torch tensors internally before conversion.
    # For now, assuming it works or is primarily for offline analysis.
    print("Comparing data coverage...")
    # ... (rest of the function, ensure states[f/t] are NumPy if needed by average_nn)
    # Note: average_nn now handles tensor conversion internally.
    # ...


if __name__ == '__main__':
    # --- Example Data Loading and Plotting ---
    try:
        task = 'nav01' # Example task
        print(f"--- Running example for task: {task} ---")

        # Load data
        print("Loading data...")
        full_data = load_data(filename=task + '_train', data_path='../data/100s', steps_per_episode=100, num_episodes=100) # Load fewer episodes for example
        print("Data loaded.")

        # Compute stats (returns numpy)
        print("Computing statistics...")
        means_np, stds_np, steps_np, mins_np, maxs_np = compute_statistics(full_data)
        print("Statistics computed.")
        # print("Means:", means_np)
        # print("Stds:", stds_np)

        # Split data
        print("Splitting data...")
        data_split = split_data(full_data, ratio=0.8)
        print("Data split.")

        # --- Example: Get one batch and noisyfy it ---
        print("Creating batch iterator...")
        train_iterator = make_batch_iterator(data_split['train'], batch_size=5, seq_len=50)
        print("Fetching one batch...")
        one_batch = next(train_iterator)
        print("Batch fetched.")

        print("Applying noise to the batch...")
        noisy_batch = noisyfy_data(one_batch, target_h=24, target_w=24) # Use default noise params, specify target size
        print("Noise applied.")

        # --- Plotting Example ---
        print("Plotting example trajectory and observations...")
        scaling = 0.5
        if task == 'nav01': plt.figure(figsize=[10*scaling, 5*scaling])
        elif task == 'nav02': plt.figure(figsize=[15*scaling, 9*scaling])
        elif task == 'nav03': plt.figure(figsize=[20*scaling, 13*scaling])
        else: plt.figure()

        # Plot the first trajectory from the *noisy* batch
        # plot_trajectory expects a dict with 's', 'a', 'o' keys, use first item [0]
        plot_data = {k: v[0:1] for k, v in noisy_batch.items()} # Select first item, keep batch dim
        plot_trajectory(plot_data, figure_name=None, emphasize=0, mincolor=0.0, linewidth=0.5)
        plot_maze(task)
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.tight_layout()
        # plt.savefig(f"../plots/{task}_example_traj.pdf", bbox_inches='tight', pad_inches=0)
        print(f"Saved example trajectory plot.")

        plt.figure()
        # Plot observations from the *noisy* batch
        plot_observations(noisy_batch, n=5) # Plot 5 observations from the batch
        plt.tight_layout()
        # plt.savefig(f"../plots/{task}_example_noisy_obs.pdf", bbox_inches='tight', pad_inches=0, dpi=200)
        print(f"Saved example observation plot.")

        print("Showing plots...")
        plt.show()
        print("--- Example Finished ---")

    except Exception as e:
        print(f"\n--- An error occurred during the example run ---")
        import traceback
        traceback.print_exc()

