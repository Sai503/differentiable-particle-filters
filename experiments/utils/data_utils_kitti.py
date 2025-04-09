import numpy as np
import matplotlib.pyplot as plt
import os
import math
import glob
from time import time
from PIL import Image
import torch # Import torch for compute_statistics

# Assuming plotting_utils works with NumPy
from utils.plotting_utils import plot_trajectories, plot_maze, plot_observations

# Keep using NumPy for basic math ops on loaded data
def wrap_angle(angle):
    """Wraps angle to [-pi, pi]. Input/Output: NumPy array."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

def rotation_matrix(x):
    """Computes rotation matrix. Input/Output: NumPy array."""
    rot_psi = np.array([[math.cos(x[2]), -math.sin(x[2]), 0], [math.sin(x[2]), math.cos(x[2]), 0], [0, 0, 1]])
    rot_theta = np.array([[math.cos(x[1]), 0, math.sin(x[1])], [0, 1, 0], [-math.sin(x[1]), 0, math.cos(x[1])]])
    rot_phi = np.array([[1, 0, 0], [0, math.cos(x[0]), -math.sin(x[0])], [0, math.sin(x[0]), math.cos(x[0])]])
    R = np.dot(rot_psi,np.dot(rot_theta,rot_phi))
    return R

def read_oxts_data(oxts, oxts_prev, oxts_init):
    """Reads specific OXTS files. Output: NumPy arrays."""
    # This function seems specific and might not be used by the main loader. Keep as is.
    with open(oxts, 'r') as f:
        oxts_data = np.loadtxt(f)
    with open(oxts_init, 'r') as f:
        oxts_init = np.loadtxt(f)
    with open(oxts_prev, 'r') as f:
        oxts_prev = np.loadtxt(f)

    scale = 6378137 * math.pi / 180
    north = (oxts_data[0] - oxts_init[0]) * scale
    east = (oxts_data[1] - oxts_init[1]) * scale * math.cos(oxts_init[0] * math.pi / 180)
    # Time difference assumed 0.103s for alpha calculation
    alpha = (oxts_data[22] - oxts_prev[22]) / 0.103 if abs(oxts_data[22] - oxts_prev[22]) > 1e-6 else 0.0
    # State: [east, north, yaw, forward_vel, angular_vel_z] - Check yaw sign convention if needed
    state = np.array([east, north, -oxts_data[5], oxts_data[8], -alpha]) # Using alpha for angular vel? Check consistency
    # Action: [accel_x, accel_y, alpha?] - Original used [14]=ay, [15]=az, alpha. Check if this is correct.
    action = np.array([oxts_data[14], oxts_data[15], alpha])

    return state, action

def load_image(img_file):
    """Loads image as NumPy array."""
    try:
        img = Image.open(img_file)
        return np.asarray(img, dtype='float32')
    except FileNotFoundError:
        print(f"Warning: Image file not found: {img_file}")
        return None # Or raise error
    except Exception as e:
        print(f"Warning: Failed to load image {img_file}: {e}")
        return None

def image_input(img1, img2):
    """Concatenates img1 and img1-img2. Input/Output: NumPy arrays."""
    if img1 is None or img2 is None:
        return None
    # Ensure images have 3 channels for concatenation
    if img1.shape[-1] != 3 or img2.shape[-1] != 3:
         print(f"Warning: image_input expects 3-channel images. Got shapes {img1.shape}, {img2.shape}")
         # Handle error or return None
         return None
    return np.concatenate((img1, img1 - img2), axis=2) # Result has 6 channels

def load_data_for_stats(oxts_data_files, image_files, diff_image_files, seq_num_indices, base_frame_file):
    """
    Loads specified data files purely for statistics calculation.
    Returns dict of NumPy arrays.
    """
    print("Loading data specifically for statistics calculation...")
    num_files = len(oxts_data_files)
    state = np.zeros((num_files, 6)) # Original state format used here? [N, E, yaw, alt?, ?, ang_vel_z?]
    action = np.zeros((num_files, 3)) # [forward_vel, acc_y, acc_z?]

    try:
        with open(base_frame_file, 'r') as f:
            data = np.loadtxt(f)
            base_lat = data[0]
            base_long = data[1]
    except FileNotFoundError:
        print(f"Error: Base frame file not found: {base_frame_file}")
        raise

    scale = 6378137 * math.pi / 180
    for ii in range(num_files):
        try:
            with open(oxts_data_files[ii], 'r') as f:
                data = np.loadtxt(f)
            north = (data[0] - base_lat) * scale
            east = (data[1] - base_long) * scale * math.cos(base_lat * math.pi / 180)
            # State: [N, E, yaw, alt, roll, pitch] - Using indices 6, 7 from file? Check KITTI docs.
            state[ii,:] = np.array([north, east, data[5], data[6], data[7], data[22]]) # Assuming index 22 is ang_vel_z
            # Action: [forward_vel, acc_y, acc_z]
            action[ii,:] = np.array([data[8], data[14], data[15]])
        except FileNotFoundError:
            print(f"Warning: OXTS file not found: {oxts_data_files[ii]}")
        except Exception as e:
            print(f"Warning: Failed to process OXTS file {oxts_data_files[ii]}: {e}")

    # Load observations (assuming image_files and diff_image_files align with oxts_data_files)
    # This part seems complex and potentially slow if loading all images.
    # Consider if stats can be computed differently or if this loading is necessary.
    # Assuming seq_num_indices helps select relevant images.
    # The original logic here was confusing. Let's simplify assuming we load relevant images.
    # For simplicity, let's assume image_files and diff_image_files correspond 1:1 to oxts_data_files
    # and we just need to load them.
    obs_list = []
    img_h, img_w = 50, 150 # Assuming fixed size after potential preprocessing
    num_channels = 6
    for ii in range(num_files):
        img1 = load_image(image_files[ii])
        img2 = load_image(diff_image_files[ii]) # Assuming diff images are pre-calculated or loaded
        obs_concat = image_input(img1, img2)
        if obs_concat is not None:
            # Ensure consistent size (resize/crop if needed here, though ideally done beforehand)
            if obs_concat.shape[0] != img_h or obs_concat.shape[1] != img_w:
                 print(f"Warning: Image size mismatch for stats ({obs_concat.shape[:2]} vs {(img_h, img_w)}). Resizing needed?")
                 # Add resizing logic here if necessary, e.g., using PIL resize
                 # For now, assume sizes match or skip
                 continue
            obs_list.append(obs_concat)
        else:
            # Append zeros or handle missing data if an image failed to load
            obs_list.append(np.zeros((img_h, img_w, num_channels), dtype='float32'))

    obs = np.stack(obs_list, axis=0) if obs_list else np.zeros((0, img_h, img_w, num_channels), dtype='float32')

    data_for_stats = {'s': state, 'a': action, 'o': obs}
    print("Finished loading data for statistics.")
    return data_for_stats


def load_kitti_sequences(sequence_list=None, data_dir='../data/kitti', cache_file='../data/kitti_processed.npz'):
    """Loads KITTI sequences, processes them, and optionally caches."""
    print('Loading KITTI DATA')
    t1 = time()

    # --- Try loading from cache ---
    if sequence_list is None and os.path.exists(cache_file):
        try:
            print(f'Trying to load from cache: {cache_file} ... ')
            # Use allow_pickle=True for dictionaries saved within npz
            data = dict(np.load(cache_file, allow_pickle=True))
            # Convert mirrored data back if saved as separate keys
            if 'o-m' in data and 's-m' in data:
                 print("Loaded mirrored data from cache.")
            else:
                 print("Adding mirrored data after loading.")
                 data = add_mirrored_data(data) # Add mirrored if not in cache
            t2 = time()
            print('Done! ({:.2f}s)'.format(t2-t1))
            # No need for np.newaxis here
            return data
        except Exception as e:
            print(f"Failed to load from cache ({e}). Loading from raw files.")

    # --- Load from raw KITTI files ---
    if sequence_list is None:
        sequence_list = list(range(11)) # Default to all sequences 00-10

    print(f'Cache not found or sequence list specified. Loading sequences {sequence_list} from {data_dir}')

    all_image_files = []
    all_oxts_files = []
    sequence_lengths = []

    # --- Gather file paths ---
    for seq in sequence_list:
        seq_str = f"{seq:02d}"
        image_dir = os.path.join(data_dir, seq_str, 'image_2') # Use image_2 (left color)
        oxts_dir = os.path.join(data_dir, seq_str, 'oxts')
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        oxts_files = sorted(glob.glob(os.path.join(oxts_dir, '*.txt')))

        # Ensure matching number of images and oxts files
        num_frames = min(len(image_files), len(oxts_files))
        if num_frames == 0:
            print(f"Warning: No matching image/oxts files found for sequence {seq_str}. Skipping.")
            continue

        all_image_files.extend(image_files[:num_frames])
        all_oxts_files.extend(oxts_files[:num_frames])
        sequence_lengths.append(num_frames)

    if not all_image_files:
        raise FileNotFoundError(f"No valid data found for sequences {sequence_list} in {data_dir}")

    # --- Calculate sequence indices and weights ---
    seq_num = np.cumsum(sequence_lengths).astype(int)
    weights = np.array(sequence_lengths, dtype=float) / sum(sequence_lengths)
    print("Sequence lengths:", sequence_lengths)
    print("Cumulative indices (seq_num):", seq_num)
    print("Sequence weights:", weights)

    total_frames = seq_num[-1]
    img_h, img_w = 50, 150 # Target image size after potential preprocessing
    num_channels_raw = 3 # Raw images are 3 channels
    num_channels_obs = 6 # Observation includes difference image

    # --- Pre-allocate NumPy arrays ---
    # We need T-1 observations/states/actions per sequence segment
    total_valid_steps = total_frames - len(sequence_list)
    o = np.zeros((total_valid_steps, img_h, img_w, num_channels_obs), dtype='float32')
    s = np.zeros((total_valid_steps, 5), dtype='float32') # x, y, theta, v, theta_dot
    a = np.zeros((total_valid_steps, 3), dtype='float32') # Placeholder for actions if calculated differently

    # --- Process each sequence ---
    count = 0 # Index for the combined arrays (o, s, a)
    global_frame_idx = 0 # Index for all_image_files / all_oxts_files
    scale = 6378137 * math.pi / 180

    for seq_idx, seq_len in enumerate(sequence_lengths):
        print(f"Processing sequence {sequence_list[seq_idx]} ({seq_len} frames)...")
        # Load initial OXTS data for this sequence to establish origin
        try:
            oxts_init_data = np.loadtxt(all_oxts_files[global_frame_idx])
            base_lat = oxts_init_data[0]
            base_long = oxts_init_data[1]
        except Exception as e:
            print(f"Error loading initial OXTS for sequence {sequence_list[seq_idx]}: {e}. Skipping sequence.")
            global_frame_idx += seq_len
            continue

        # Load first image for difference calculation
        prev_image = load_image(all_image_files[global_frame_idx])
        if prev_image is None:
             print(f"Error loading initial image for sequence {sequence_list[seq_idx]}. Skipping sequence.")
             global_frame_idx += seq_len
             continue
        # Preprocess first image (resize/crop if necessary)
        # Assuming load_image returns raw image, resize/crop here
        # Example using PIL (ensure consistent size: img_h, img_w)
        prev_image = np.array(Image.fromarray(prev_image.astype(np.uint8)).resize((img_w, img_h)), dtype='float32')


        # Iterate through frames of the current sequence (from the second frame)
        for frame_in_seq in range(1, seq_len):
            current_frame_idx = global_frame_idx + frame_in_seq
            prev_frame_idx = current_frame_idx - 1

            # --- Load Image Data ---
            cur_image = load_image(all_image_files[current_frame_idx])
            if cur_image is None: continue # Skip if image loading failed
            # Preprocess current image
            cur_image = np.array(Image.fromarray(cur_image.astype(np.uint8)).resize((img_w, img_h)), dtype='float32')

            # Create observation (current image + difference)
            obs_concat = image_input(cur_image, prev_image)
            if obs_concat is None: continue # Skip if concatenation failed

            # --- Load OXTS Data ---
            try:
                oxts_curr_data = np.loadtxt(all_oxts_files[current_frame_idx])
                oxts_prev_data = np.loadtxt(all_oxts_files[prev_frame_idx])
            except Exception as e:
                print(f"Error loading OXTS data around frame {current_frame_idx}: {e}. Skipping step.")
                prev_image = cur_image # Update prev_image even if skipping
                continue

            # --- Calculate State (at current time step) ---
            north = (oxts_curr_data[0] - base_lat) * scale
            east = (oxts_curr_data[1] - base_long) * scale * math.cos(base_lat * math.pi / 180)
            # Yaw from OXTS (check sign convention: neg z is often clockwise?)
            # KITTI raw data: yaw is counter-clockwise, 0 = east, pi/2 = north
            yaw = oxts_curr_data[5]
            # Forward velocity (m/s)
            forward_vel = oxts_curr_data[8]
            # Angular velocity around z (rad/s) - approximate from yaw difference
            # Use a small time step (approx 0.1s between frames)
            time_step = 0.103 # Approximate average time step
            yaw_prev = oxts_prev_data[5]
            delta_yaw = wrap_angle(yaw - yaw_prev)
            angular_vel_z = delta_yaw / time_step if time_step > 1e-3 else 0.0

            current_state = np.array([east, north, yaw, forward_vel, angular_vel_z], dtype='float32')

            # --- Calculate Action (taken between prev and curr step) ---
            # Action could be defined differently (e.g., control inputs if available).
            # Here, let's use velocity/acceleration from OXTS as a proxy.
            # Action: [forward_vel_prev, acc_y_prev, acc_z_prev]? Or [delta_dist, delta_yaw]?
            # Let's use the previous step's velocities as the 'action' that led to the current state.
            # This is just one possible definition.
            forward_vel_prev = oxts_prev_data[8]
            acc_y_prev = oxts_prev_data[14] # Lateral acceleration
            acc_z_prev = oxts_prev_data[15] # Vertical acceleration (might not be useful)
            # Alternative action: delta distance and delta yaw
            # delta_dist = forward_vel_prev * time_step
            # delta_yaw_action = angular_vel_z_prev * time_step # Need angular_vel_z_prev

            # Using velocities/accel for now:
            current_action = np.array([forward_vel_prev, acc_y_prev, acc_z_prev], dtype='float32')


            # --- Store results ---
            if count < total_valid_steps:
                o[count] = obs_concat
                s[count] = current_state
                a[count] = current_action # Store the calculated action
                count += 1
            else:
                print("Warning: Exceeded pre-allocated array size. Check calculations.")
                break # Avoid writing out of bounds

            # Update previous image for next iteration
            prev_image = cur_image

        # Move global index past the current sequence
        global_frame_idx += seq_len

    # Trim arrays if fewer steps were processed than allocated
    if count < total_valid_steps:
        print(f"Trimming arrays from {total_valid_steps} to {count} valid steps.")
        o = o[:count]
        s = s[:count]
        a = a[:count]
        # Adjust seq_num to reflect the actual number of steps processed per sequence segment
        # This requires tracking counts per sequence, more complex. For now, assume seq_num remains based on original lengths.

    t2 = time()
    print('Data loading finished! ({:.2f}s)'.format(t2 - t1))

    # --- Save to cache ---
    if sequence_list is None or len(sequence_list) == 11: # Only cache if loading all default sequences
        try:
            print(f'Saving processed data to cache: {cache_file}')
            # Save mirrored data explicitly if needed later
            data_to_save = {'s': s, 'a': a, 'o': o, 'seq_num': seq_num, 'weights': weights}
            # mirrored = add_mirrored_data({'s': s, 'o': o}) # Create mirrored versions
            # data_to_save['s-m'] = mirrored['s-m']
            # data_to_save['o-m'] = mirrored['o-m']
            np.savez(cache_file, **data_to_save)
        except Exception as e:
            print(f"Warning: Failed to save cache file {cache_file}: {e}")

    data = {'s': s, 'a': a, 'o': o, 'seq_num': seq_num, 'weights': weights}

    # Add mirrored data for use during training/batching
    data = add_mirrored_data(data)

    # NO np.newaxis needed here
    # for key in 'osa':
    #     data[key] = data[key][:, np.newaxis, :]
    # if 'o-m' in data: data['o-m'] = data['o-m'][:, np.newaxis, :]
    # if 's-m' in data: data['s-m'] = data['s-m'][:, np.newaxis, :]

    print("Final data shapes:")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")

    return data


# Removed the conflicting load_data function


def compute_statistics(data):
    """
    Computes statistics (mean, std, step_sizes, min, max) from training data.
    Uses PyTorch for calculations, returns NumPy arrays.
    Args:
        data (dict): Dictionary containing training data as NumPy arrays ('o', 's', 'a', 'seq_num').
    Returns:
        tuple: Contains (means_np, stds_np, state_step_sizes_np, state_mins_np, state_maxs_np)
               where all elements are NumPy arrays or dicts of NumPy arrays.
    """
    print("Computing statistics using PyTorch...")
    means_t = {}
    stds_t = {}
    state_step_sizes_t = []
    state_mins_t = []
    state_maxs_t = []

    # --- Convert relevant data to Torch tensors (on CPU) ---
    data_t = {}
    for key in 'osa':
        if key in data:
            try:
                data_t[key] = torch.from_numpy(data[key]).float() # Ensure float32
            except TypeError:
                print(f"Warning: Could not convert data['{key}'] to tensor. Skipping stats for this key.")
                continue
        else:
            print(f"Warning: Key '{key}' not found in data for statistics calculation.")

    # --- Calculate Means and Stds ---
    for key in data_t.keys(): # Iterate over keys successfully converted to tensors
        current_data_t = data_t[key]

        # Compute means over the first dimension (time/steps).
        # Keepdims=True helps with broadcasting later.
        means_t[key] = torch.mean(current_data_t, dim=0, keepdim=True)

        # Special handling for state/action means if needed (e.g., zero mean for angle/action)
        # if key == 's' and current_data_t.shape[-1] >= 3:
        #     means_t[key][..., 2] = 0 # Zero mean for angle (dim 2)
        # if key == 'a':
        #     means_t[key] = means_t[key] * 0 # Zero mean for actions

        # Compute std over the first dimension (time/steps).
        if key == 'o':
            # Input shape: [T, H, W, C] (assuming channels last after loading)
            if current_data_t.dim() == 4:
                # Calculate std per channel across T, H, W
                num_channels = current_data_t.shape[-1]
                # Reshape to [C, -1]
                data_reshaped = current_data_t.permute(3, 0, 1, 2).reshape(num_channels, -1)
                stds_t[key] = torch.std(data_reshaped, dim=1) # Std per channel [C]
                # Reshape stds back to broadcast correctly: [1, 1, 1, C]
                stds_t[key] = stds_t[key].view(1, 1, 1, num_channels)
            else: # Fallback
                 print(f"Warning: Unexpected shape for data_t['o']: {current_data_t.shape}. Calculating std over dim 0.")
                 stds_t[key] = torch.std(current_data_t - means_t[key], dim=0, keepdim=True)
        else: # For 's' and 'a'
            stds_t[key] = torch.std(current_data_t - means_t[key], dim=0, keepdim=True)
            # Average std for x, y components if state/action dim >= 2
            # if key == 's' and current_data_t.shape[-1] >= 2:
            #     stds_t[key][..., :2] = torch.mean(stds_t[key][..., :2])


    # --- Calculate State Step Sizes, Mins, Maxs (using PyTorch) ---
    if 's' in data_t:
        state_data_t = data_t['s'] # Shape [T, 5]
        state_dim = state_data_t.shape[-1]
        seq_num = data['seq_num'] # Use NumPy seq_num for indexing

        all_steps = []
        last_idx = 0
        for end_idx in seq_num:
            # Get steps within the current sequence segment
            segment = state_data_t[last_idx:end_idx]
            if len(segment) > 1:
                steps = segment[1:] - segment[:-1] # Shape [SeqLen-1, 5]
                all_steps.append(steps)
            last_idx = end_idx

        if all_steps:
            all_steps_t = torch.cat(all_steps, dim=0) # Shape [TotalValidSteps, 5]

            # Handle angle wrapping for dim 2
            if state_dim >= 3:
                pi_tensor = torch.tensor(math.pi, dtype=all_steps_t.dtype, device=all_steps_t.device)
                all_steps_t[:, 2] = torch.remainder(all_steps_t[:, 2] + pi_tensor, 2 * pi_tensor) - pi_tensor

            # Calculate mean absolute step size per dimension
            state_step_sizes_t = torch.mean(torch.abs(all_steps_t), dim=0) # Shape [5]

            # Average step sizes for x and y (dims 0, 1)
            if state_dim >= 2:
                avg_xy_step = (state_step_sizes_t[0] + state_step_sizes_t[1]) / 2.0
                state_step_sizes_t[0] = avg_xy_step
                state_step_sizes_t[1] = avg_xy_step
        else:
             state_step_sizes_t = torch.zeros(state_dim) # Default if no valid steps

        # Calculate mins and maxs over the whole dataset
        state_mins_t = torch.min(state_data_t, dim=0)[0] # Shape [5]
        state_maxs_t = torch.max(state_data_t, dim=0)[0] # Shape [5]

    else: # Handle case where 's' is missing
        print("Warning: Key 's' not found in data_t. Step sizes, mins, maxs will be zeros.")
        # Need to know state_dim, assume 5 for KITTI if 's' is missing? Or get from elsewhere?
        assumed_state_dim = 5
        state_step_sizes_t = torch.zeros(assumed_state_dim)
        state_mins_t = torch.zeros(assumed_state_dim)
        state_maxs_t = torch.zeros(assumed_state_dim)

    # --- Convert results back to NumPy for Return ---
    means_np = {k: v.cpu().numpy() for k, v in means_t.items()}
    stds_np = {k: v.cpu().numpy() for k, v in stds_t.items()}
    state_step_sizes_np = state_step_sizes_t.cpu().numpy()
    state_mins_np = state_mins_t.cpu().numpy()
    state_maxs_np = state_maxs_t.cpu().numpy()

    print("Statistics computation finished.")
    return means_np, stds_np, state_step_sizes_np, state_mins_np, state_maxs_np


def split_data(data, ratio=0.8, categories=['train', 'val']):
    """Splits data based on sequence boundaries."""
    print(f"Splitting data based on sequences. Ratio: {ratio}")
    split_data_dict = {categories[0]: {}, categories[1]: {}}
    num_sequences = len(data['seq_num'])
    split_point_seq_idx = math.floor(num_sequences * ratio) # Index of the sequence boundary

    if split_point_seq_idx == 0 or split_point_seq_idx == num_sequences:
        raise ValueError(f"Split ratio {ratio} results in an empty '{categories[0]}' or '{categories[1]}' set.")

    # Data index where the split occurs (end of the last training sequence)
    split_point_data_idx = data['seq_num'][split_point_seq_idx - 1]

    for key in data.keys():
        if key == 'seq_num':
            split_data_dict[categories[0]][key] = data[key][:split_point_seq_idx]
            # Adjust seq_num for validation set to be relative to its start
            split_data_dict[categories[1]][key] = data[key][split_point_seq_idx:] - split_point_data_idx
        elif key == 'weights':
            # Split weights and re-normalize
            split_data_dict[categories[0]][key] = data[key][:split_point_seq_idx]
            split_data_dict[categories[0]][key] /= np.sum(split_data_dict[categories[0]][key]) # Normalize train weights
            split_data_dict[categories[1]][key] = data[key][split_point_seq_idx:]
            split_data_dict[categories[1]][key] /= np.sum(split_data_dict[categories[1]][key]) # Normalize val weights
        else: # For 's', 'a', 'o', 's-m', 'o-m'
            split_data_dict[categories[0]][key] = data[key][:split_point_data_idx]
            split_data_dict[categories[1]][key] = data[key][split_point_data_idx:]

    for cat in split_data_dict:
        num_seq = len(split_data_dict[cat]['seq_num'])
        num_steps = split_data_dict[cat]['s'].shape[0] # Use 's' to get step count
        print(f"Split --> {cat}: {num_seq} sequences, {num_steps} steps")

    return split_data_dict


# --- reduce_data, shuffle_data, remove_state (Keep as NumPy operations) ---
def reduce_data(data, num_episodes):
    # This function name is misleading for KITTI (uses sequences, not episodes)
    # It seems intended to reduce the number of *sequences*.
    print(f"Reducing data to {num_episodes} sequences.")
    if num_episodes <= 0 or num_episodes > len(data['seq_num']):
        raise ValueError(f"Invalid number of sequences requested: {num_episodes}")

    new_data = {}
    end_idx_data = data['seq_num'][num_episodes - 1]

    for key in data.keys():
        if key == 'seq_num':
            new_data[key] = data[key][:num_episodes]
        elif key == 'weights':
            new_data[key] = data[key][:num_episodes]
            new_data[key] /= np.sum(new_data[key]) # Re-normalize
        else: # 's', 'a', 'o', etc.
            new_data[key] = data[key][:end_idx_data]
    return new_data

def shuffle_data(data):
    # Shuffling sequences in KITTI might break temporal dependencies if not careful.
    # This function shuffles individual *steps*, which is likely incorrect for sequence data.
    # Re-implementing to shuffle *sequences* while keeping steps within sequences intact.
    print("Shuffling sequences...")
    new_data = {k: [] for k in data.keys() if k not in ['seq_num', 'weights']} # Init lists for data
    num_sequences = len(data['seq_num'])
    shuffled_seq_indices = np.random.permutation(num_sequences)

    new_seq_lengths = []
    last_idx_orig = 0
    for seq_idx_orig in shuffled_seq_indices:
        start_idx_orig = 0 if seq_idx_orig == 0 else data['seq_num'][seq_idx_orig - 1]
        end_idx_orig = data['seq_num'][seq_idx_orig]
        seq_len = end_idx_orig - start_idx_orig
        new_seq_lengths.append(seq_len)

        for key in new_data.keys(): # 's', 'a', 'o', 's-m', 'o-m'
            new_data[key].append(data[key][start_idx_orig:end_idx_orig])

    # Concatenate shuffled sequences
    for key in new_data.keys():
        new_data[key] = np.concatenate(new_data[key], axis=0)

    # Update seq_num and weights based on shuffled order
    new_data['seq_num'] = np.cumsum(new_seq_lengths).astype(int)
    new_data['weights'] = np.array(new_seq_lengths, dtype=float) / sum(new_seq_lengths)

    print("Finished shuffling sequences.")
    return new_data


def remove_state(data, provide_initial_state=False):
    """Removes 's' key, optionally keeps first step."""
    new_data = {}
    for key in data.keys():
        if key not in ['s', 's-m']: # Also remove mirrored state
            new_data[key] = data[key]
    # Initial state isn't meaningful in the same way for KITTI as it depends on sequence start.
    # This function might not be suitable for KITTI prediction setup.
    if provide_initial_state:
        print("Warning: 'provide_initial_state' not typically used with KITTI sequence structure in remove_state.")
        # If needed, logic to extract first state of each sequence would go here.
    return new_data


# --- noisify_data_condition, noisyfy_data (Keep as NumPy operations) ---
# Note: noisyfy_data currently only adds noise to 'a'. Image noise/cropping is missing.
def noisify_data_condition(data, condition):
    print('Applying noise condition:', condition)
    # Add more conditions or modify existing ones as needed
    if condition == 'odom0_imgC': # Example: Clean odom, clean image
        return noisyfy_data(data, odom_noise_factor=0.0)
    elif condition == 'odom10_imgC': # Example: Noisy odom, clean image
        return noisyfy_data(data, odom_noise_factor=1.0)
    # Add conditions involving image noise/cropping if implemented in noisyfy_data
    else:
        print(f"Warning: Unknown noise condition '{condition}'. Applying default noise (odom_noise=1.0).")
        return noisyfy_data(data, odom_noise_factor=1.0)

def noisyfy_data(data, odom_noise_factor=1.0): # Removed img params as they weren't used
    """Adds noise to actions ('a'). Image noise/cropping NOT implemented here."""
    print(f"Noisyfying data (NumPy): odom_noise={odom_noise_factor}")
    new_data = data.copy() # Shallow copy is usually sufficient for dicts of arrays

    if 'a' in new_data:
        # Add multiplicative noise centered around 1 to actions
        noise = np.random.normal(1.0, 0.1 * odom_noise_factor, new_data['a'].shape)
        noise = np.maximum(noise, 1e-2) # Prevent zero/negative noise factor
        new_data['a'] = new_data['a'] * noise.astype(new_data['a'].dtype)
    else:
        print("Warning: Key 'a' not found in data during noisyfy_data.")

    # --- Placeholder for Image Noise/Cropping ---
    # If image noise/cropping is needed at this stage (before batching),
    # port the logic from data_utils.py::noisyfy_data here, operating on NumPy arrays.
    # Remember to handle both 'o' and 'o-m' if mirroring is used.
    # Example structure:
    # if 'o' in new_data and (img_noise_factor > 0 or img_random_shift):
    #     print("Applying image noise/cropping...")
    #     # ... NumPy implementation of cropping and noise addition ...
    #     # new_data['o'] = processed_o
    #     # if 'o-m' in new_data:
    #     #     new_data['o-m'] = processed_o_mirrored

    return new_data

# --- Batch Iterators (Yield NumPy arrays) ---

def make_batch_iterator(data, batch_size=32, seq_len=10, use_mirrored_data=True):
    """Generator yielding random batches (NumPy arrays) of specified seq_len."""
    seq_num = data['seq_num']
    weights = data['weights']
    num_sequences = len(seq_num)

    if num_sequences == 0:
        print("Warning: No sequences available for batch iteration.")
        return

    while True:
        batch_data = {k: [] for k in data.keys() if k not in ['seq_num', 'weights']}
        batch_count = 0
        while batch_count < batch_size:
            # Choose sequence segment based on weights
            seq_idx = np.random.choice(num_sequences, p=weights)

            # Determine start and end indices for this sequence segment in the flat array
            start_idx_global = 0 if seq_idx == 0 else seq_num[seq_idx - 1]
            end_idx_global = seq_num[seq_idx]
            segment_len = end_idx_global - start_idx_global

            if segment_len <= seq_len:
                # print(f"Warning: Sequence {seq_idx} is too short ({segment_len} <= {seq_len}). Skipping.")
                continue # Skip short sequences

            # Sample a valid start step within the chosen sequence segment
            max_start_offset = segment_len - seq_len
            start_offset = np.random.randint(0, max_start_offset) # Exclusive upper bound
            step_idx_global = start_idx_global + start_offset

            # Decide whether to use mirrored data for this sample
            use_mirror = use_mirrored_data and batch_count >= batch_size // 2

            # Extract data for the sequence
            for key in batch_data.keys():
                key_to_use = key
                # Select mirrored key if appropriate
                if use_mirror and key + '-m' in data:
                    key_to_use = key + '-m'
                elif key.endswith('-m') and not use_mirror: # Don't accidentally select mirror if not intended
                    continue # Skip mirrored keys if not use_mirror

                # Check if the key exists before slicing
                if key_to_use in data:
                     # Slice the data: [start_global : start_global + seq_len]
                     data_slice = data[key_to_use][step_idx_global : step_idx_global + seq_len]
                     # Ensure the slice has the expected length
                     if data_slice.shape[0] == seq_len:
                          batch_data[key].append(data_slice)
                     else:
                          # This should ideally not happen with correct index calculation
                          print(f"Warning: Slice error for key '{key_to_use}', seq {seq_idx}, start {step_idx_global}. Got shape {data_slice.shape}, expected {seq_len}. Skipping sample.")
                          # Need to discard other keys added for this sample
                          for k_discard in batch_data.keys():
                              if len(batch_data[k_discard]) > batch_count:
                                   batch_data[k_discard].pop()
                          batch_count -= 1 # Decrement count as sample failed
                          break # Break inner loop to retry sampling
                else:
                     print(f"Warning: Key '{key_to_use}' not found in data dictionary.")
                     # Handle missing key if necessary

            else: # Only increment if the inner loop completed without break
                 batch_count += 1


        # Stack the collected samples into a batch
        try:
            batch_np = {k: np.stack(v, axis=0) for k, v in batch_data.items() if v} # Stack along new batch dim
            if batch_np['s'].shape[0] == batch_size: # Check if we got a full batch
                 yield batch_np
            else:
                 print(f"Warning: Yielding incomplete batch (size {batch_np['s'].shape[0]}/{batch_size}).")
                 if batch_np['s'].shape[0] > 0: yield batch_np # Yield if not empty
                 # else: continue # Skip empty batch
        except ValueError as e:
             print(f"Error stacking batch: {e}. Skipping batch.")
             # Print shapes for debugging
             # for k, v_list in batch_data.items():
             #     if v_list: print(f"  Key {k}: {[item.shape for item in v_list]}")
             continue


def make_repeating_batch_iterator(data, epoch_len, batch_size=32, seq_len=10, use_mirrored_data=True):
    """Generator yielding batches (NumPy arrays) based on pre-sampled indices for one epoch, then repeats."""
    seq_num = data['seq_num']
    weights = data['weights']
    num_sequences = len(seq_num)

    if num_sequences == 0:
        print("Warning: No sequences available for repeating batch iteration.")
        return

    # --- Pre-sample indices for the entire epoch ---
    print(f"Pre-sampling {epoch_len} batches for repeating iterator...")
    epoch_indices = []
    for _ in range(epoch_len * batch_size):
        while True: # Keep sampling until a valid index is found
            seq_idx = np.random.choice(num_sequences, p=weights)
            start_idx_global = 0 if seq_idx == 0 else seq_num[seq_idx - 1]
            end_idx_global = seq_num[seq_idx]
            segment_len = end_idx_global - start_idx_global

            if segment_len > seq_len:
                max_start_offset = segment_len - seq_len
                start_offset = np.random.randint(0, max_start_offset)
                step_idx_global = start_idx_global + start_offset
                epoch_indices.append(step_idx_global)
                break # Valid index found
            # else: sequence too short, re-sample

    epoch_indices = np.array(epoch_indices).reshape(epoch_len, batch_size)
    print("Pre-sampling done.")

    # --- Yield batches based on pre-sampled indices ---
    while True: # Loop indefinitely over epochs
        for i in range(epoch_len): # Iterate through one epoch
            batch_indices_global = epoch_indices[i] # Indices for the current batch [batch_size]
            batch_data = {k: [] for k in data.keys() if k not in ['seq_num', 'weights']}

            for batch_idx, step_idx_global in enumerate(batch_indices_global):
                use_mirror = use_mirrored_data and batch_idx >= batch_size // 2

                for key in batch_data.keys():
                    key_to_use = key
                    if use_mirror and key + '-m' in data:
                        key_to_use = key + '-m'
                    elif key.endswith('-m') and not use_mirror:
                        continue

                    if key_to_use in data:
                        data_slice = data[key_to_use][step_idx_global : step_idx_global + seq_len]
                        if data_slice.shape[0] == seq_len:
                             batch_data[key].append(data_slice)
                        else: # Should not happen with pre-sampled valid indices
                             print(f"Error: Pre-sampled index {step_idx_global} resulted in incorrect slice shape for key {key_to_use}. Expected {seq_len}, got {data_slice.shape[0]}.")
                             # Handle error - potentially discard sample and log
                             for k_discard in batch_data.keys():
                                 if len(batch_data[k_discard]) > batch_idx: batch_data[k_discard].pop()
                             # This indicates an issue in pre-sampling logic if it occurs.
                             break # Break inner key loop
                else: # Only continue if inner loop didn't break
                     continue
                break # Break outer batch_idx loop if inner loop broke

            # Stack and yield if batch seems complete
            try:
                # Check if all lists have the expected batch size before stacking
                list_len = len(batch_data['s']) # Use 's' as reference
                if list_len == batch_size:
                    batch_np = {k: np.stack(v, axis=0) for k, v in batch_data.items() if v}
                    yield batch_np
                else:
                    # This indicates an error during batch assembly from pre-sampled indices
                    print(f"Warning: Assembled batch size {list_len} != expected {batch_size} in repeating iterator. Skipping batch.")
            except ValueError as e:
                 print(f"Error stacking repeating batch: {e}. Skipping batch.")
                 continue


# --- make_complete_batch_iterator, make_batch_iterator_for_evaluation ---
# These seem less critical for the main training loop, keep as NumPy for now.
# Review logic if used. make_complete_batch_iterator might be memory intensive.

def make_complete_batch_iterator(data, batch_size=1000, seq_len=10):
    """Generator yielding batches covering all possible start steps (NumPy arrays)."""
    seq_num = data['seq_num']
    num_sequences = len(seq_num)
    print("Warning: make_complete_batch_iterator can be memory intensive.")

    all_indices = []
    for seq_idx in range(num_sequences):
        start_idx_global = 0 if seq_idx == 0 else seq_num[seq_idx - 1]
        end_idx_global = seq_num[seq_idx]
        segment_len = end_idx_global - start_idx_global
        if segment_len > seq_len:
            max_start_offset = segment_len - seq_len
            for start_offset in range(max_start_offset):
                all_indices.append(start_idx_global + start_offset)

    num_total_samples = len(all_indices)
    print(f"Total possible samples for complete iterator: {num_total_samples}")
    idx_ptr = 0
    while idx_ptr < num_total_samples:
        batch_indices_global = all_indices[idx_ptr : idx_ptr + batch_size]
        idx_ptr += len(batch_indices_global)

        batch_data = {k: [] for k in data.keys() if k not in ['seq_num', 'weights']}
        for step_idx_global in batch_indices_global:
             for key in batch_data.keys():
                 # Not handling mirroring here, assumes base data only
                 if key in data:
                     data_slice = data[key][step_idx_global : step_idx_global + seq_len]
                     if data_slice.shape[0] == seq_len:
                          batch_data[key].append(data_slice)
                     else: # Should not happen
                          print(f"Error: Slice error in complete iterator at index {step_idx_global}.")
                          break # Break key loop
             else: # Continue if inner loop didn't break
                  continue
             break # Break outer loop if inner loop broke

        # Stack and yield
        try:
            if len(batch_data['s']) == len(batch_indices_global): # Check if batch is complete
                batch_np = {k: np.stack(v, axis=0) for k, v in batch_data.items() if v}
                yield batch_np
            else:
                 print("Warning: Incomplete batch in complete iterator.")
        except ValueError as e:
             print(f"Error stacking complete batch: {e}. Skipping.")
             continue


def make_batch_iterator_for_evaluation(data, start_step, trajectory_idx, batch_size=1, seq_len=10):
    """Yields a specific sequence segment for evaluation (NumPy arrays)."""
    seq_num = data['seq_num']
    num_sequences = len(seq_num)

    if trajectory_idx < 0 or trajectory_idx >= num_sequences:
        raise ValueError(f"Invalid trajectory_idx: {trajectory_idx}")

    # Find global start index for the sequence
    start_idx_global = 0 if trajectory_idx == 0 else seq_num[trajectory_idx - 1]
    end_idx_global = seq_num[trajectory_idx]
    segment_len = end_idx_global - start_idx_global

    # Adjust start_step to be relative to the global index
    eval_start_idx_global = start_idx_global + start_step

    if start_step < 0 or eval_start_idx_global + seq_len > end_idx_global:
        raise ValueError(f"Invalid start_step {start_step} or seq_len {seq_len} for trajectory {trajectory_idx} (len {segment_len}).")

    # This iterator yields only one batch
    batch_data = {k: [] for k in data.keys() if k not in ['seq_num', 'weights']}
    for _ in range(batch_size): # Repeat the same segment batch_size times
        for key in batch_data.keys():
            # Not handling mirroring here
            if key in data:
                data_slice = data[key][eval_start_idx_global : eval_start_idx_global + seq_len]
                if data_slice.shape[0] == seq_len:
                     batch_data[key].append(data_slice)
                else: # Should not happen
                     raise RuntimeError("Slice error in evaluation iterator.")

    # Stack and yield once
    try:
        batch_np = {k: np.stack(v, axis=0) for k, v in batch_data.items() if v}
        yield batch_np
    except ValueError as e:
         print(f"Error stacking evaluation batch: {e}.")
         raise


# --- Plotting functions (Keep as NumPy/Matplotlib) ---
def plot_observation_check(data, means, stds):
    # ... (keep original plotting logic, ensure inputs are NumPy) ...
    print("Plotting observation check...")
    observations = data['o'] # NumPy array
    means_o = means['o']     # NumPy array
    stds_o = stds['o']       # NumPy array

    plt.ion()
    num_to_plot = min(5, len(observations)) # Plot only a few
    for i in range(num_to_plot):
        o = observations[i:i+1] # Keep batch dim for broadcasting with stats
        norm_o = (o - means_o) / (stds_o + 1e-8) # Add epsilon

        print(f"\n--- Observation {i} ---")
        for d in range(o.shape[-1]): # Iterate through channels
            plt.figure(f"Obs {i} - Chan {d} (Norm)")
            plt.clf()
            plt.imshow(norm_o[0, :, :, d], interpolation='nearest', cmap='coolwarm', vmin=-3, vmax=3)
            plt.title(f"Obs {i} - Chan {d} (Normalized)")
            plt.colorbar()
            print(f'  Chan {d} (Raw Range): ({np.min(o[0, :, :, d]):.2f} - {np.max(o[0, :, :, d]):.2f})')
            show_pause(0.1)

        # Plot composite images (assuming 6 channels = RGB + DiffRGB)
        if o.shape[-1] == 6:
            plt.figure(f"Obs {i} - RGB")
            plt.clf()
            plt.imshow(np.clip(o[0, :, :, :3] / 255.0, 0.0, 1.0), interpolation='nearest')
            plt.title(f"Obs {i} - RGB")
            show_pause(0.1)

            plt.figure(f"Obs {i} - Diff RGB")
            plt.clf()
            # Visualize difference image centered around 0.5
            plt.imshow(np.clip(o[0, :, :, 3:6] / 255.0 / 2.0 + 0.5, 0.0, 1.0), interpolation='nearest')
            plt.title(f"Obs {i} - Diff RGB")
            show_pause(0.1)

        plt.pause(1) # Pause longer between observations
    plt.ioff()
    print("Finished observation check.")


def plot_video(data):
    # ... (keep original plotting logic, ensure input is NumPy) ...
    print("Plotting video...")
    observations = data['o'] # NumPy array
    plt.ion()
    plt.figure("Video")
    num_frames = observations.shape[0]
    for i, o in enumerate(observations):
        plt.clf()
        # Assuming 6 channels, plot first 3 (RGB)
        if o.shape[-1] >= 3:
            plt.imshow(np.clip(o[:, :, :3] / 255.0, 0.0, 1.0), interpolation='nearest')
            plt.title(f"Frame {i}/{num_frames}")
        plt.pause(0.05)
    plt.ioff()
    print("Finished plotting video.")


def plot_sequences(data, means, stds, state_step_sizes):
    # ... (keep original plotting logic, ensure inputs are NumPy) ...
    print("Plotting sequences...")
    seq_num = data['seq_num']
    s_data = data['s']
    o_data = data['o']

    # Plot example observations at sequence boundaries
    plt.figure('Sequence Boundary Observations')
    num_seq = len(seq_num)
    num_rows = min(num_seq, 5)
    num_cols = 4
    last_idx = 0
    for k, i in enumerate(seq_num):
        if k >= num_rows: break
        indices_to_plot = [last_idx, last_idx + 1, i - 2, i - 1] # Start and end of segment
        for plot_j, data_idx in enumerate(indices_to_plot):
            if data_idx < 0 or data_idx >= o_data.shape[0]: continue
            ax = plt.subplot(num_rows, num_cols, k * num_cols + plot_j + 1)
            if o_data.shape[-1] >= 3:
                ax.imshow(np.clip(o_data[data_idx, :, :, :3] / 255.0, 0.0, 1.0), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Seq {k}, Idx {data_idx}", fontsize=8)
        last_idx = i
    plt.tight_layout()

    # Plot trajectories
    plt.figure('Trajectories')
    last_idx = 0
    for k, i in enumerate(seq_num):
        segment = s_data[last_idx:i]
        if len(segment) > 0:
            plt.plot(segment[:, 0], segment[:, 1], label=f"Seq {k}")
            # Optionally plot quivers (can be slow)
            # plt.quiver(segment[:, 0], segment[:, 1], np.cos(segment[:, 2]), np.sin(segment[:, 2]), scale=50, width=0.002)
        last_idx = i
    plt.gca().set_aspect('equal')
    plt.legend(fontsize='small')
    plt.title("Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Plot normalized state components
    plt.figure('Normalized State Components')
    if 's' in means and 's' in stds:
        norm_states = (s_data - means['s']) / (stds['s'] + 1e-8)
        state_labels = ['x', 'y', 'theta', 'v', 'theta_dot']
        for d in range(min(s_data.shape[-1], len(state_labels))):
            plt.plot(norm_states[:, d], label=f'Norm {state_labels[d]}')
        # Add vertical lines for sequence boundaries
        for i in seq_num[:-1]: plt.axvline(i, color='k', linestyle='--', linewidth=0.5)
        plt.legend(fontsize='small')
        plt.title("Normalized State Components")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Value")
    else:
        plt.text(0.5, 0.5, "Stats missing for normalization", ha='center', va='center')


    # Plot scaled state components
    plt.figure('Scaled State Components')
    if len(state_step_sizes) == s_data.shape[-1]:
        scaled_states = s_data / (state_step_sizes + 1e-8)
        state_labels = ['x', 'y', 'theta', 'v', 'theta_dot']
        for d in range(min(s_data.shape[-1], len(state_labels))):
            plt.plot(scaled_states[:, d], label=f'Scaled {state_labels[d]}')
            print(f'Dim {d} ({state_labels[d]}): Step Size = {state_step_sizes[d]:.4f}')
        for i in seq_num[:-1]: plt.axvline(i, color='k', linestyle='--', linewidth=0.5)
        plt.legend(fontsize='small')
        plt.title("State Components Scaled by Step Size")
        plt.xlabel("Time Step")
        plt.ylabel("Scaled Value")
    else:
        plt.text(0.5, 0.5, "State step sizes missing or mismatched", ha='center', va='center')

    print("Finished plotting sequences.")
    plt.show()


def add_mirrored_data(data):
    """Adds mirrored versions ('o-m', 's-m') to the data dictionary."""
    print("Adding mirrored data...")
    # Mirror observations horizontally
    if 'o' in data:
        data['o-m'] = data['o'][:, :, ::-1, :] # Flip width dimension (axis 2)
    # Mirror state: invert y, theta, theta_dot
    if 's' in data:
        s_orig = data['s']
        data['s-m'] = np.concatenate([
            s_orig[..., 0:1],      # keep x
           -s_orig[..., 1:2],      # invert y
           wrap_angle(-s_orig[..., 2:3]), # invert and wrap angle
            s_orig[..., 3:4],      # keep forward vel
           -s_orig[..., 4:5]       # invert angular vel
        ], axis=-1).astype(s_orig.dtype)
    print("Finished adding mirrored data.")
    return data

if __name__ == '__main__':
    # Example usage
    try:
        # Load all sequences (or specify a list e.g., [0, 1, 2])
        # Set cache_file=None to force reloading from raw data
        kitti_data = load_kitti_sequences(sequence_list=None, cache_file='../data/kitti_processed.npz')

        # Compute statistics (returns numpy)
        means_np, stds_np, steps_np, mins_np, maxs_np = compute_statistics(kitti_data)

        # Split data
        data_split = split_data(kitti_data, ratio=0.8)

        # Create a batch iterator (yields numpy batches)
        train_iterator = make_repeating_batch_iterator(data_split['train'], epoch_len=10, batch_size=4, seq_len=20)

        # Get and print shape of one batch
        one_batch_np = next(train_iterator)
        print("\n--- Example Batch ---")
        for k, v in one_batch_np.items():
            print(f"  Batch key '{k}' shape: {v.shape}, dtype: {v.dtype}")

        # Plotting examples (optional)
        # plot_sequences(kitti_data, means_np, stds_np, steps_np)
        # plot_observation_check(one_batch_np, means_np, stds_np) # Pass batch and stats
        # plot_video(kitti_data['o'][:100]) # Plot first 100 frames

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the KITTI raw data is correctly placed relative to the script,")
        print("or that the cache file path is correct.")
    except Exception as e:
        print(f"\nAn error occurred during the KITTI data utils example:")
        import traceback
        traceback.print_exc()
