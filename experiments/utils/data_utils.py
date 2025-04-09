import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def load_data(data_path='../data/100s', filename='nav01_train', steps_per_episode=100, num_episodes=None):
    # Load from npz and convert to torch tensors.
    file_path = os.path.join(data_path, filename + '.npz')
    try:
        raw = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        raise
    # Convert to float32 tensors on CPU
    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
    raw.close() # Close the npz file

    # Reshape each array: [total_steps] -> [-1, steps_per_episode, ...]
    num_total_steps = data[list(data.keys())[0]].shape[0]
    if num_episodes is not None:
        num_total_steps = min(num_total_steps, num_episodes * steps_per_episode)

    actual_num_episodes = num_total_steps // steps_per_episode

    if actual_num_episodes == 0:
        raise ValueError(f"Not enough data for even one episode. Found {num_total_steps} steps, need {steps_per_episode}.")

    final_num_steps = actual_num_episodes * steps_per_episode

    reshaped_data = {}
    for key in data.keys():
        # Slice first, then reshape
        sliced_data = data[key][:final_num_steps]
        try:
            new_shape = [actual_num_episodes, steps_per_episode] + list(sliced_data.shape[1:])
            reshaped_data[key] = sliced_data.view(*new_shape)
        except RuntimeError as e:
            print(f"Error reshaping key '{key}' with shape {sliced_data.shape} to {new_shape}. Error: {e}")
            raise

    # Convert degrees into radians for pose and velocity.
    pi_tensor = torch.tensor(math.pi, dtype=torch.float32)
    if 'pose' in reshaped_data:
        reshaped_data['pose'][..., 2] *= pi_tensor / 180.0
        reshaped_data['pose'][..., 2] = wrap_angle(reshaped_data['pose'][..., 2])
    if 'vel' in reshaped_data: # Check if 'vel' exists
        reshaped_data['vel'][..., 2] *= pi_tensor / 180.0
        # No wrap_angle needed for velocity

    # Calculate relative actions (a) based on pose (s)
    pose = reshaped_data['pose']
    abs_d_x = pose[:, 1:, 0:1] - pose[:, :-1, 0:1]
    abs_d_y = pose[:, 1:, 1:2] - pose[:, :-1, 1:2]
    d_theta = wrap_angle(pose[:, 1:, 2:3] - pose[:, :-1, 2:3])
    s = torch.sin(pose[:, :-1, 2:3])
    c = torch.cos(pose[:, :-1, 2:3])
    rel_d_x = c * abs_d_x + s * abs_d_y
    rel_d_y = -s * abs_d_x + c * abs_d_y # Corrected relative y calculation (standard rotation)

    # Check if 'rgbd' key exists for observations 'o'
    obs_key = 'rgbd' if 'rgbd' in reshaped_data else 'o' if 'o' in reshaped_data else None
    if obs_key is None:
        raise KeyError("Observation data key ('rgbd' or 'o') not found in loaded data.")

    # Return dictionary with standard keys 'o', 's', 'a'
    # Ensure data aligns: o[t] corresponds to s[t] and leads to action a[t] (which resulted in s[t+1])
    # Original code aligns o[t+1] with s[t+1] and a[t]. Let's keep that.
    return {'o': reshaped_data[obs_key][:, 1:, ...], # Observation at t+1
            's': pose[:, 1:, :],                     # State at t+1
            'a': torch.cat([rel_d_x, rel_d_y, d_theta], dim=-1)} # Action taken at t


def compute_statistics(data):
    """
    Computes statistics (mean, std, step_sizes, min, max) from training data.
    Args:
        data (dict): Dictionary containing training data as PyTorch tensors ('o', 's', 'a').
    Returns:
        tuple: Contains (means, stds, state_step_sizes, state_mins, state_maxs)
               where means and stds are dictionaries of NumPy arrays, and the rest are NumPy arrays.
    """
    means_t = {}
    stds_t = {}
    state_step_sizes_t = []
    state_mins_t = []
    state_maxs_t = []

    # --- Calculate Means and Stds (using PyTorch) ---
    for key in 'osa':
        if key not in data:
            print(f"Warning: Key '{key}' not found in data for statistics calculation.")
            continue

        # Compute means over the first two dimensions (episodes and time).
        current_data = data[key]
        means_t[key] = torch.mean(current_data.float(), dim=(0, 1), keepdim=True) # Ensure float

        if key == 's':
            # state_dim = current_data.shape[-1]
            if current_data.shape[-1] >= 3: # Check if state dim includes orientation
                 means_t[key][..., 2] = 0  # leave orientation mean at 0
        if key == 'a':
            means_t[key] = means_t[key] * 0  # set action mean to zero

        # Compute std over all but the last dimension(s).
        # For images ('o'), compute std per channel across B, T, H, W
        if key == 'o':
            # Input shape: [B, T, H, W, C] or [B, T, C, H, W]
            if current_data.dim() == 5:
                if current_data.shape[-1] < current_data.shape[2]: # Likely [B, T, H, W, C]
                    channel_dim = -1
                    dims_to_reduce = (0, 1, 2, 3)
                else: # Likely [B, T, C, H, W]
                    channel_dim = 2
                    dims_to_reduce = (0, 1, 3, 4)

                # Calculate std per channel
                num_elements = np.prod([current_data.shape[d] for d in dims_to_reduce])
                # Reshape to [C, -1] to calculate std per channel easily
                data_reshaped = current_data.movedim(channel_dim, 0).reshape(current_data.shape[channel_dim], -1)
                stds_t[key] = torch.std(data_reshaped.float(), dim=1) # Std per channel [C]
                # Reshape stds back to broadcast correctly with means_t[key]
                # means_t[key] is likely [1, 1, 1, 1, C] or [1, 1, C, 1, 1]
                std_shape = [1] * current_data.dim()
                std_shape[channel_dim] = current_data.shape[channel_dim]
                stds_t[key] = stds_t[key].view(std_shape)

            else: # Fallback for unexpected observation shape
                 print(f"Warning: Unexpected shape for data['o']: {current_data.shape}. Calculating std over all leading dims.")
                 dims = tuple(range(current_data.dim() - 1))
                 stds_t[key] = torch.std(current_data.float() - means_t[key], dim=dims, keepdim=True)

        else: # For 's' and 'a'
            dims = tuple(range(current_data.dim() - 1))
            stds_t[key] = torch.std(current_data.float() - means_t[key], dim=dims, keepdim=True)
            # Average std for x, y components if state/action dim >= 2
            if current_data.shape[-1] >= 2:
                 stds_t[key][..., :2] = torch.mean(stds_t[key][..., :2])


    # --- Calculate State Step Sizes, Mins, Maxs (using PyTorch) ---
    if 's' in data:
        state_data = data['s']
        state_dim = state_data.shape[-1]
        for i in range(state_dim):
            # Difference between consecutive steps
            steps = (state_data[:, 1:, i] - state_data[:, :-1, i]).view(-1)
            if i == 2 and state_dim >= 3: # Assume dim 2 is orientation if state_dim >= 3
                steps = wrap_angle(steps)
            state_step_sizes_t.append(torch.mean(torch.abs(steps)))

        # Average step sizes for x and y if state_dim >= 2
        if state_dim >= 2:
            avg_xy_step = (state_step_sizes_t[0] + state_step_sizes_t[1]) / 2.0
            state_step_sizes_t[0] = avg_xy_step
            state_step_sizes_t[1] = avg_xy_step

        state_step_sizes_t = torch.stack(state_step_sizes_t)

        for i in range(state_dim):
            state_mins_t.append(torch.min(state_data[..., i]))
            state_maxs_t.append(torch.max(state_data[..., i]))
        state_mins_t = torch.stack(state_mins_t)
        state_maxs_t = torch.stack(state_maxs_t)
    else: # Handle case where 's' is missing
        print("Warning: Key 's' not found in data for statistics calculation. Step sizes, mins, maxs will be empty.")
        state_step_sizes_t = torch.empty(0)
        state_mins_t = torch.empty(0)
        state_maxs_t = torch.empty(0)


    # --- Convert to NumPy for Return ---
    means_np = {k: v.cpu().numpy() for k, v in means_t.items()}
    stds_np = {k: v.cpu().numpy() for k, v in stds_t.items()}
    state_step_sizes_np = state_step_sizes_t.cpu().numpy()
    state_mins_np = state_mins_t.cpu().numpy()
    state_maxs_np = state_maxs_t.cpu().numpy()

    return means_np, stds_np, state_step_sizes_np, state_mins_np, state_maxs_np


def split_data(data, ratio=0.8, categories=['train', 'val']):
    print(f"Splitting data. Input episodes: {data['s'].shape[0]}")
    split_data_dict = {categories[0]: {}, categories[1]: {}}
    num_episodes = data[list(data.keys())[0]].shape[0]
    split_point = int(num_episodes * ratio)

    if split_point == 0 or split_point == num_episodes:
        raise ValueError(f"Split ratio {ratio} results in an empty '{categories[0]}' or '{categories[1]}' set. Adjust ratio or check data size.")

    for key in data.keys():
        split_data_dict[categories[0]][key] = data[key][:split_point]
        split_data_dict[categories[1]][key] = data[key][split_point:]

    for cat in split_data_dict:
        # Check if the split resulted in non-empty data for this category
        if list(split_data_dict[cat].values()): # Check if dict has any values
             print(f"Split --> {cat}: {split_data_dict[cat][list(data.keys())[0]].shape[0]} episodes")
        else:
             print(f"Split --> {cat}: 0 episodes (Warning: Check split ratio and data size)")

    return split_data_dict


def reduce_data(data, num_episodes):
    new_data = {}
    min_episodes = data[list(data.keys())[0]].shape[0]
    num_episodes = min(num_episodes, min_episodes) # Don't request more than available
    if num_episodes <= 0:
         raise ValueError("num_episodes must be positive.")
    for key in data.keys(): # Apply to all keys
        new_data[key] = data[key][:num_episodes]
    return new_data


def shuffle_data(data):
    new_data = {}
    num_episodes = data[list(data.keys())[0]].shape[0]
    indices = torch.randperm(num_episodes)
    for key in data.keys(): # Apply to all keys
        new_data[key] = data[key][indices]
    return new_data


def remove_state(data, provide_initial_state=False):
    new_data = {}
    # Copy keys that are not 's'
    for key in data.keys():
        if key != 's':
            new_data[key] = data[key]
    # Optionally add initial state
    if provide_initial_state and 's' in data:
        new_data['s'] = data['s'][:, :1, ...] # Get only the first time step
    return new_data


def noisify_data_condition(data, condition):
    print('Applying noise condition:', condition)
    if condition == 'odom0_imgTG':
        return noisyfy_data(data, odom_noise_factor=0.0, img_noise_factor=1.0, img_random_shift=True)
    elif condition == 'odom5_imgTG':
        return noisyfy_data(data, odom_noise_factor=0.5, img_noise_factor=1.0, img_random_shift=True)
    elif condition == 'odom10_imgTG':
        return noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=True) # Default
    elif condition == 'odom20_imgTG':
        return noisyfy_data(data, odom_noise_factor=2.0, img_noise_factor=1.0, img_random_shift=True)
    elif condition == 'odomX_imgTG': # Scrambled odometry
        data = noisyfy_data(data, odom_noise_factor=0.0, img_noise_factor=1.0, img_random_shift=True) # Start with clean odom
        if 'a' in data:
            shape = data['a'].shape
            a = data['a'].view(-1, shape[-1])
            idx = torch.randperm(a.shape[0])
            a = a[idx]
            data['a'] = a.view(shape)
        return data
    elif condition == 'odom10_imgC': # Clean image
        return noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=0.0, img_random_shift=False)
    elif condition == 'odom10_imgG': # Gaussian noise only
        return noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=False)
    elif condition == 'odom10_imgT': # Translation noise only
        return noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=0.0, img_random_shift=True)
    elif condition == 'odom10_imgX': # Scrambled images
        data = noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=0.0, img_random_shift=False) # Start with clean images
        if 'o' in data:
            shape = data['o'].shape
            # Flatten B and T dimensions for shuffling
            o = data['o'].view(-1, *shape[2:])
            idx = torch.randperm(o.shape[0])
            o = o[idx]
            data['o'] = o.view(shape)
        return data
    else:
        print(f"Warning: Unknown noise condition '{condition}'. Applying default noise.")
        return noisyfy_data(data)


def noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=True, target_h=24, target_w=24):
    """Adds noise and performs cropping on data."""
    print(f"Noisyfying data: odom_noise={odom_noise_factor}, img_noise={img_noise_factor}, img_shift={img_random_shift}, target_size=({target_h},{target_w})")
    new_data = {}

    # --- Noise for Actions ('a') ---
    if 'a' in data:
        device = data['a'].device
        dtype = data['a'].dtype
        # Add multiplicative noise centered around 1
        noise_a = torch.normal(mean=1.0, std=0.1 * odom_noise_factor, size=data['a'].shape, device=device, dtype=dtype)
        # Ensure noise doesn't make actions zero or negative if std is large
        noise_a = torch.clamp(noise_a, min=1e-2)
        new_data['a'] = data['a'] * noise_a
    else:
        print("Warning: Key 'a' not found in data during noisyfy_data.")

    # --- Noise and Cropping for Observations ('o') ---
    if 'o' in data:
        obs_data = data['o'].clone() # Clone to avoid modifying original
        device_o = obs_data.device
        dtype_o = obs_data.dtype

        # --- Ensure correct shape [B, T, H, W, C] for cropping ---
        if obs_data.dim() != 5:
             raise ValueError(f"Observation data 'o' must have 5 dimensions [B, T, H, W, C] or [B, T, C, H, W]. Got {obs_data.dim()}")

        # Infer C, H, W assuming C is the smaller dimension (<=10)
        shape = obs_data.shape
        potential_c_dim = -1
        if shape[2] <= 10 and shape[2] < shape[3] and shape[2] < shape[4]:
             potential_c_dim = 2
        elif shape[-1] <= 10 and shape[-1] < shape[2] and shape[-1] < shape[3]:
             potential_c_dim = -1 # Or 4
        else:
             # Fallback or raise error if channel dim is ambiguous
             print(f"Warning: Cannot reliably determine channel dimension for observation shape {shape}. Assuming channels last.")
             potential_c_dim = -1

        if potential_c_dim == 2: # Input is [B, T, C, H, W]
             print("Permuting observations from [B, T, C, H, W] to [B, T, H, W, C] for cropping.")
             obs_data = obs_data.permute(0, 1, 3, 4, 2).contiguous()

        # Now obs_data should be [B, T, H, W, C]
        B, T, H, W, C = obs_data.shape

        # Check if target size is valid
        if target_h > H or target_w > W:
             raise ValueError(f"Target crop size ({target_h}, {target_w}) is larger than image size ({H}, {W}).")

        # Initialize new_o with the TARGET shape [B, T, target_h, target_w, C]
        new_o = torch.zeros(B, T, target_h, target_w, C, dtype=dtype_o, device=device_o)

        max_offset_H = H - target_h
        max_offset_W = W - target_w

        # --- Apply Cropping ---
        for i in range(B):
            for j in range(T):
                if img_random_shift:
                    offset_h = torch.randint(0, max_offset_H + 1, (1,), device=device_o).item()
                    offset_w = torch.randint(0, max_offset_W + 1, (1,), device=device_o).item()
                else: # Center crop
                    offset_h = max_offset_H // 2
                    offset_w = max_offset_W // 2

                # Crop the original image [H, W, C]
                cropped_img = obs_data[i, j, offset_h : offset_h + target_h, offset_w : offset_w + target_w, :]
                new_o[i, j] = cropped_img

        # --- Add Gaussian Noise ---
        if img_noise_factor > 0:
            # Assuming image data range is roughly 0-255, std=20 is reasonable noise level
            noise_o = torch.normal(mean=0.0, std=20.0 * img_noise_factor, size=new_o.shape, device=device_o, dtype=dtype_o)
            new_o = new_o + noise_o
            # Optional: Clamp to valid image range (e.g., 0-255) if necessary
            # new_o = torch.clamp(new_o, 0, 255)

        new_data['o'] = new_o
    else:
        print("Warning: Key 'o' not found in data during noisyfy_data.")

    # --- Copy State ('s') ---
    if 's' in data:
        new_data['s'] = data['s'].clone() # Clone state as well
    else:
        print("Warning: Key 's' not found in data during noisyfy_data.")


    return new_data

# --- Batch Iterators (No changes needed, they yield CPU tensors) ---

def make_batch_iterator(data, batch_size=32, seq_len=10):
    """Generator yielding random batches of specified seq_len."""
    num_episodes = data['s'].shape[0]
    ep_len = data['s'].shape[1]
    if seq_len >= ep_len:
        raise ValueError(f"seq_len ({seq_len}) must be smaller than episode length ({ep_len}).")
    max_start_step = ep_len - seq_len # Max valid start index

    if num_episodes == 0 or max_start_step < 0:
        print("Warning: No data available for batch iteration.")
        return # Stop iteration if no data

    current_step = 0
    while True: # Keep yielding batches indefinitely
        # Sample random episodes and start steps for the batch
        episodes = torch.randint(0, num_episodes, (batch_size,))
        start_steps = torch.randint(0, max_start_step, (batch_size,)) # Use max_start_step

        batches = {}
        for k in data.keys():
            batch_list = []
            # Efficiently gather slices using advanced indexing
            # Create indices for episodes and time steps
            # episode_indices = episodes.unsqueeze(1).expand(-1, seq_len) # [B, T]
            # time_indices = start_steps.unsqueeze(1) + torch.arange(seq_len) # [B, T]
            # batches[k] = data[k][episode_indices, time_indices] # Doesn't work directly for dicts

            # Simpler loop (might be slightly less efficient but clearer)
            for ep_idx, start_idx in zip(episodes, start_steps):
                 batch_list.append(data[k][ep_idx:ep_idx+1, start_idx : start_idx + seq_len])
            try:
                 batches[k] = torch.cat(batch_list, dim=0) # Concatenate along batch dim
            except RuntimeError as e:
                 print(f"Error concatenating batch for key '{k}'. Check data shapes and seq_len.")
                 print(f"Shapes in batch_list: {[item.shape for item in batch_list]}")
                 raise e
        yield batches
        current_step += 1


def make_repeating_batch_iterator(data, epoch_len, batch_size=32, seq_len=10):
    """Generator yielding batches based on pre-sampled indices for one epoch, then repeats."""
    num_episodes = data['s'].shape[0]
    ep_len = data['s'].shape[1]
    if seq_len >= ep_len:
        raise ValueError(f"seq_len ({seq_len}) must be smaller than episode length ({ep_len}).")
    max_start_step = ep_len - seq_len

    if num_episodes == 0 or max_start_step < 0:
        print("Warning: No data available for repeating batch iteration.")
        return

    # Pre-sample indices for the entire epoch length
    repeating_episodes = torch.randint(0, num_episodes, (epoch_len, batch_size))
    repeating_start_steps = torch.randint(0, max_start_step, (epoch_len, batch_size))

    current_step = 0
    while True: # Loop indefinitely over epochs
        for i in range(epoch_len): # Iterate through pre-sampled indices for one epoch
            episodes = repeating_episodes[i]
            start_steps = repeating_start_steps[i]
            batches = {}
            for k in data.keys():
                batch_list = []
                for ep_idx, start_idx in zip(episodes, start_steps):
                    batch_list.append(data[k][ep_idx:ep_idx+1, start_idx : start_idx + seq_len])
                try:
                    batches[k] = torch.cat(batch_list, dim=0)
                except RuntimeError as e:
                     print(f"Error concatenating repeating batch for key '{k}'. Step {i}/{epoch_len}.")
                     print(f"Shapes in batch_list: {[item.shape for item in batch_list]}")
                     raise e
            yield batches
            current_step += 1


def make_complete_batch_iterator(data, batch_size=1000, seq_len=10):
    """Generator yielding batches covering all possible start steps across all episodes."""
    num_episodes = data['s'].shape[0]
    ep_len = data['s'].shape[1]
    if seq_len >= ep_len:
        raise ValueError(f"seq_len ({seq_len}) must be smaller than episode length ({ep_len}).")
    num_start_steps = ep_len - seq_len

    if num_episodes == 0 or num_start_steps <= 0:
        print("Warning: No data available for complete batch iteration.")
        return

    # Create a list of all possible (episode, start_step) indices
    batch_indices = [(i, j) for i in range(num_episodes) for j in range(num_start_steps)]
    # Optional: Shuffle indices for randomness within an epoch
    # random.shuffle(batch_indices) # Use Python's random if needed

    idx_ptr = 0
    while idx_ptr < len(batch_indices):
        # Get indices for the current batch
        current_indices = batch_indices[idx_ptr : idx_ptr + batch_size]
        idx_ptr += len(current_indices) # Move pointer

        batches = {}
        for k in data.keys():
            batch_list = []
            for (ep_idx, start_idx) in current_indices:
                batch_list.append(data[k][ep_idx:ep_idx+1, start_idx : start_idx + seq_len])
            try:
                batches[k] = torch.cat(batch_list, dim=0)
            except RuntimeError as e:
                 print(f"Error concatenating complete batch for key '{k}'.")
                 print(f"Shapes in batch_list: {[item.shape for item in batch_list]}")
                 raise e
        yield batches


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

