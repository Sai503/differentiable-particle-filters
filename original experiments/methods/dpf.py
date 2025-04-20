# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


#desired updates for mbot:
# update measurement model for standard rgb and not rgbd
# mlp (linear layers) for lidar encoding
# concatenate lidar and rgb encoding
# update motion model to use lidar and rgb encoding

# Assuming these utils handle torch tensors correctly or return numpy arrays
# that we will convert.
from utils.data_utils import wrap_angle, compute_statistics, split_data, make_batch_iterator, make_repeating_batch_iterator
from utils.method_utils import atan2, compute_sq_distance
from utils.plotting_utils import plot_maze, show_pause

# Helper function to move batch data to the target device
def move_batch_to_device(batch, device):
    """Moves all tensor values in a dictionary to the specified device."""
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to tensors on the device
            # Be mindful of the required dtype (float32 is common)
            try:
                batch_device[key] = torch.from_numpy(value).float().to(device)
            except TypeError: # Handle potential non-numeric numpy arrays if necessary
                 batch_device[key] = value # Keep as is if conversion fails
        else:
            batch_device[key] = value # Keep non-tensor/numpy types as they are
    return batch_device

class DPF(nn.Module):
    def __init__(self, init_with_true_state, learn_odom, use_proposer,
                 propose_ratio, proposer_keep_ratio, min_obs_likelihood,
                 dropout_keep_prob=1.0):
        """
        PyTorch version of the particle filter.
        (Note: some hyperparameters and details have been re–interpreted.)
        """
        super(DPF, self).__init__()
        self.init_with_true_state = init_with_true_state
        self.learn_odom = learn_odom
        # Only use proposer if not initializing with true state.
        self.use_proposer = use_proposer and (not init_with_true_state)
        self.propose_ratio = propose_ratio if not self.init_with_true_state else 0.0
        self.min_obs_likelihood = min_obs_likelihood
        self.state_dim = 3
        self.dropout_keep_prob = dropout_keep_prob  # (keep probability)
        self.proposer_keep_ratio = proposer_keep_ratio

        # “Placeholder” for number of particles will be set later.
        self.num_particles = None
        # Later in training we set self.particle_std

        # --- Store statistics as tensors on the correct device later ---
        self.means_t = None
        self.stds_t = None
        self.state_step_sizes_t = None
        self.state_mins_t = None
        self.state_maxs_t = None
        self.device = None # Will be set in fit/predict

        # Build the learnable modules.
        self.build_modules()

    def build_modules(self):
        # --- Encoder: a convolutional network ---
        # We assume input images are 24x24 with 3 channels.
        # Note: In PyTorch image tensors are [B, C, H, W].
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),  # 24
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 6 -> 3
            nn.ReLU()
        )
        self.encoder_flatten = nn.Flatten()
        # Calculate flattened size: 64 channels * 3 height * 3 width
        encoder_flat_size = 64 * 3 * 3
        self.encoder_dropout = nn.Dropout(p=1 - self.dropout_keep_prob)
        self.encoder_linear = nn.Linear(encoder_flat_size, 128) # Corrected input size

        # --- Observation likelihood estimator ---
        # Input: concatenation of the 128–dim image encoding and 4–dim transformed state.
        self.obs_like_estimator = nn.Sequential(
            nn.Linear(128 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # --- Motion noise generator ---
        # Input: concatenation of normalized action (3 dims) and a random 3–vector.
        self.mo_noise_generator = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.state_dim)
        )

        # --- Odometry transition model (if learning it) ---
        if self.learn_odom:
            # Input: concatenation of a 4–dim transformed state and 3–dim action = 7 dims.
            self.mo_transition_model = nn.Sequential(
                nn.Linear(7, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.state_dim)
            )

        # --- Particle proposer (if used) ---
        if self.use_proposer:
            self.particle_proposer = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(p=1 - self.proposer_keep_ratio),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Tanh()
            )

    def forward_encoder(self, o):
        """
        Encodes images.
        o: torch.Tensor of shape [B, H, W, C] or [B, C, H, W].
           Expected to be on the correct device already.
        """
        # Ensure correct shape: [B, C, H, W]
        if o.dim() == 4 and o.size(-1) == 4:
            # Make sure permutation happens on the correct device
            o = o.permute(0, 3, 1, 2).contiguous() # Add contiguous for efficiency
        # Ensure float type
        if o.dtype != torch.float32:
             o = o.float()

        x = self.encoder_conv(o)
        x = self.encoder_flatten(x)
        x = self.encoder_dropout(x)
        x = self.encoder_linear(x)
        return F.relu(x)

    def measurement_update(self, encoding, particles):
    
        # Computes the observation likelihood for each particle.
        # encoding: [B, 128] (on device)
        # particles: [B, N, 3] (on device)
        # Uses self.means_t and self.stds_t which should be tensors on the device.
    
        if self.means_t is None or self.stds_t is None:
             raise ValueError("Statistics (means_t, stds_t) not initialized or not on device.")

        particle_input = self.transform_particles_as_input(particles)  # [B, N, 4] (on device)
        # Tile encoding to match number of particles.
        encoding_input = encoding.unsqueeze(1).expand(-1, particles.size(1), -1)
        inp = torch.cat([encoding_input, particle_input], dim=-1)  # [B, N, 132]
        B, N, _ = inp.shape
        inp_flat = inp.view(B * N, -1)
        out = self.obs_like_estimator(inp_flat).view(B, N)
        # Scale the output between min_obs_likelihood and 1.
        return out * (1 - self.min_obs_likelihood) + self.min_obs_likelihood

    def transform_particles_as_input(self, particles):
        """
        Normalizes particle poses and augments with cosine and sine of orientation.
        particles: [B, N, 3] (on device)
        Uses self.means_t['s'] and self.stds_t['s'] which should be tensors on the device.
        """
        # means_t['s'] has shape [1, 1, 3] (on device)
        # stds_t['s'] has shape [1, 1, 3] (on device)

        # Slice means and stds to get the x, y components (shape [1, 1, 2])
        means_xy = self.means_t['s'][:, :, :2]
        stds_xy = self.stds_t['s'][:, :, :2]

        # Slice particles to get x, y components (shape [B, N, 2])
        particles_xy = particles[:, :, :2]

        # Normalize position. Broadcasting works: [B, N, 2] op [1, 1, 2] -> [B, N, 2]
        # Add epsilon to stds_xy to prevent division by zero
        norm_pos = (particles_xy - means_xy) / (stds_xy + 1e-8)

        # Get orientation components
        cos_theta = torch.cos(particles[:, :, 2:3])
        sin_theta = torch.sin(particles[:, :, 2:3])

        # Concatenate: [B, N, 2] + [B, N, 1] + [B, N, 1] -> [B, N, 4]
        return torch.cat([norm_pos, cos_theta, sin_theta], dim=-1)


    def propose_particles(self, encoding, num_particles):
        """
        Proposes new particles from an image encoding using a learned proposer network.

        Args:
            encoding (torch.Tensor): Image encodings, shape [B, encoding_dim] (on device).
            num_particles (int): Number of particles to propose per batch item.
        Uses self.state_mins_t and self.state_maxs_t which should be tensors on the device.

        Returns:
            torch.Tensor: Proposed particle states, shape [B, num_particles, 3] (on device).
        """
        if self.state_mins_t is None or self.state_maxs_t is None:
             raise ValueError("State bounds (state_mins_t, state_maxs_t) not initialized or not on device.")

        B = encoding.size(0)
        device = encoding.device # Get device from input tensor
        dtype = encoding.dtype

        # --- Handle empty batch or zero particles ---
        if B == 0 or num_particles == 0:
            return torch.empty((B, num_particles, 3), dtype=dtype, device=device)

        # --- Ensure particle_proposer exists ---
        if not hasattr(self, 'particle_proposer') or self.particle_proposer is None:
            raise RuntimeError("Particle proposer module (self.particle_proposer) is not defined or initialized.")

        # --- Duplicate encoding for each particle ---
        encoding_dup = encoding.unsqueeze(1).expand(-1, num_particles, -1)

        # --- Apply proposer network ---
        inp_flat = encoding_dup.reshape(B * num_particles, -1)
        proposed_raw = self.particle_proposer(inp_flat) # Output shape: [B * num_particles, 4]
        proposed_raw = proposed_raw.view(B, num_particles, 4)

        # --- Transform the outputs to valid state values ---
        # Use the pre-converted tensor versions of state bounds
        state_mins = self.state_mins_t
        state_maxs = self.state_maxs_t

        # Ensure state bounds have at least 2 elements for x and y
        if len(state_mins) < 2 or len(state_maxs) < 2:
             raise ValueError(f"state_mins_t/state_maxs_t must have at least 2 elements for x, y. Got lengths {len(state_mins)}, {len(state_maxs)}")

        # Scale and shift the first output (index 0) for x-coordinate
        range_x = state_maxs[0] - state_mins[0]
        mid_x = (state_maxs[0] + state_mins[0]) / 2.0
        part0 = proposed_raw[:, :, 0:1] * (range_x / 2.0) + mid_x

        # Scale and shift the second output (index 1) for y-coordinate
        range_y = state_maxs[1] - state_mins[1]
        mid_y = (state_maxs[1] + state_mins[1]) / 2.0
        part1 = proposed_raw[:, :, 1:2] * (range_y / 2.0) + mid_y

        # Compute angle (theta) using atan2 from the third and fourth outputs (indices 2, 3)
        # Use torch.atan2 which expects (y, x)
        part2 = torch.atan2(proposed_raw[:, :, 2:3], proposed_raw[:, :, 3:4])

        # --- Concatenate results ---
        proposed_particles = torch.cat([part0, part1, part2], dim=-1) # Shape: [B, num_particles, 3]

        return proposed_particles


    def motion_update(self, actions, particles, stop_sampling_gradient=False):
        # """
        # Moves particles according to actions with added learned noise.
        # actions: [B, 3] (on device)
        # particles: [B, N, 3] (on device)
        # Uses self.stds_t, self.means_t, self.state_step_sizes_t (tensors on device).
        # """
        if self.stds_t is None or self.means_t is None or self.state_step_sizes_t is None:
             raise ValueError("Statistics (stds_t, means_t, state_step_sizes_t) not initialized or not on device.")

        # Expand actions to match particles.
        actions_exp = actions.unsqueeze(1)  # [B, 1, 3]
        std_a = self.stds_t['a'] # Use tensor version

        # Ensure std_a is on the same device and dtype as actions
        # std_a should already be a tensor on the correct device from fit()
        # if std_a.device != actions.device or std_a.dtype != actions.dtype:
        #     std_a = std_a.to(device=actions.device, dtype=actions.dtype)

        action_input = actions_exp / (std_a + 1e-8) # Add epsilon
        B, N, _ = particles.shape
        action_input = action_input.expand(-1, N, -1)

        # Add random noise (create on the correct device)
        random_input = torch.randn_like(action_input) # Creates tensor on same device as action_input
        inp = torch.cat([action_input, random_input], dim=-1)  # [B, N, 6]
        inp_flat = inp.view(B * N, -1)
        delta = self.mo_noise_generator(inp_flat).view(B, N, -1)

        if stop_sampling_gradient:
            delta = delta.detach()
        # Center the noise delta
        delta = delta - delta.mean(dim=1, keepdim=True)
        noisy_actions = actions_exp + delta  # [B, N, 3]

        if self.learn_odom:
            # Use learned odometry model.
            state_input = self.transform_particles_as_input(particles)  # [B, N, 4] (on device)
            action_input_odom = noisy_actions / (std_a + 1e-8) # Use tensor std_a
            inp2 = torch.cat([state_input, action_input_odom], dim=-1)  # [B, N, 7]
            inp2_flat = inp2.view(B * N, -1)
            state_delta = self.mo_transition_model(inp2_flat).view(B, N, -1)

            # Use tensor version of state_step_sizes
            state_step_sizes = self.state_step_sizes_t
            # Ensure state_step_sizes is on the correct device/dtype
            # state_step_sizes should already be a tensor on the correct device from fit()
            # if state_step_sizes.device != state_delta.device or state_step_sizes.dtype != state_delta.dtype:
            #      state_step_sizes = state_step_sizes.to(device=state_delta.device, dtype=state_delta.dtype)

            state_delta = state_delta * state_step_sizes.unsqueeze(0).unsqueeze(0) # Ensure broadcasting [1, 1, 3]
            new_states = particles + state_delta
            # wrap_angle should handle tensors
            new_states[:, :, 2:3] = wrap_angle(new_states[:, :, 2:3])
            moved_particles = new_states
        else:
            # Standard odometry model (using PyTorch operations)
            theta = particles[:, :, 2:3]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            # Element-wise multiplication for applying rotation
            delta_x = noisy_actions[:, :, 0:1] * cos_theta + noisy_actions[:, :, 1:2] * sin_theta
            delta_y = noisy_actions[:, :, 0:1] * sin_theta - noisy_actions[:, :, 1:2] * cos_theta # Check sign convention if needed
            new_x = particles[:, :, 0:1] + delta_x
            new_y = particles[:, :, 1:2] + delta_y
            # wrap_angle should handle tensors
            new_theta = wrap_angle(particles[:, :, 2:3] + noisy_actions[:, :, 2:3])
            moved_particles = torch.cat([new_x, new_y, new_theta], dim=-1)
        return moved_particles

    def particles_to_state(self, particle_list, particle_probs_list):
        """
        Computes the weighted mean (position and orientation) of particles.
        particle_list: [B, T, N, 3] (on device)
        particle_probs_list: [B, T, N] (on device)
        Returns tensor on the same device.
        """
        # Weighted mean for x,y.
        mean_position = torch.sum(particle_probs_list.unsqueeze(-1) * particle_list[..., :2], dim=2)
        # For orientation, use weighted average on sin and cos.
        cos_sum = torch.sum(particle_probs_list.unsqueeze(-1) * torch.cos(particle_list[..., 2:3]), dim=2)
        sin_sum = torch.sum(particle_probs_list.unsqueeze(-1) * torch.sin(particle_list[..., 2:3]), dim=2)
        # Use torch.atan2 which expects (y, x)
        mean_orientation = torch.atan2(sin_sum, cos_sum)
        return torch.cat([mean_position, mean_orientation], dim=-1)

    def connect_modules(self, batch):
        """
        Builds the full particle filtering process.
        batch: dict with keys 'o', 'a', 's', already moved to the target device.
        Uses self.means_t, self.stds_t, self.state_mins_t, self.state_maxs_t, self.state_step_sizes_t
        """
        # Assume batch tensors are already on the correct device
        o_batch = batch['o']
        a_batch = batch['a']
        s_batch = batch['s']
        device = s_batch.device # Get device from batch tensors
        dtype = s_batch.dtype

        # print(f"connect_modules received batch on device: {device}") # Debug print

        B, T = o_batch.shape[0], o_batch.shape[1]

        # Flatten time dimension to process images through encoder.
        # Ensure input 'o' is float and has correct channel dimension [B*T, C, H, W]
        o_flat = o_batch.reshape(B * T, *o_batch.shape[2:]) # Reshape first
        if o_flat.dim() == 4 and o_flat.size(-1) == 3: # Check if channels are last
             o_flat = o_flat.permute(0, 3, 1, 2).contiguous()
        if o_flat.dtype != torch.float32:
             o_flat = o_flat.float()

        # Normalize observations before encoding
        o_flat_norm = (o_flat - self.means_t['o'].unsqueeze(0)) / (self.stds_t['o'].unsqueeze(0) + 1e-8)
        encodings = self.forward_encoder(o_flat_norm).view(B, T, -1)  # [B, T, 128] (on device)

        # Initialize particles.
        if self.init_with_true_state:
            # Use ground truth state from batch (already on device)
            initial_particles = s_batch[:, 0].unsqueeze(1).expand(B, self.num_particles, -1)
        else:
            if self.use_proposer:
                # Propose particles using the first encoding (on device)
                initial_particles = self.propose_particles(encodings[:, 0], self.num_particles)
            else:
                # Random uniform initialization (on device)
                state_mins = self.state_mins_t
                state_maxs = self.state_maxs_t
                parts = []
                for d in range(self.state_dim):
                    # Create random tensor directly on the device
                    rand_tensor = torch.rand(B, self.num_particles, 1, device=device, dtype=dtype)
                    parts.append(rand_tensor * (state_maxs[d] - state_mins[d]) + state_mins[d])
                initial_particles = torch.cat(parts, dim=-1)

        # Initial probabilities (on device)
        initial_particle_probs = torch.ones(B, self.num_particles, device=device, dtype=dtype) / self.num_particles

        # Lists to store history (will contain tensors on device).
        particle_list = [initial_particles]
        particle_probs_list = [initial_particle_probs]

        particles = initial_particles
        particle_probs = initial_particle_probs

        # Process each time step.
        for i in range(1, T):
            # Determine numbers of proposed/resampled particles.
            num_proposed_float = (self.propose_ratio ** float(i)) * float(self.num_particles)
            num_proposed = int(round(num_proposed_float))
            num_resampled = self.num_particles - num_proposed

            combined_particles_list = []
            combined_probs_list = []

            # --- Resampling Step (if needed) ---
            if num_resampled > 0:
                # Systematic resampling (on device)
                markers = torch.linspace(0, 1 - 1/num_resampled, steps=num_resampled,
                                           device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
                random_offset = torch.rand(B, 1, device=device, dtype=dtype) * (1/num_resampled)
                markers = markers + random_offset
                cum_probs = torch.cumsum(particle_probs, dim=1)

                # Efficient batch-wise searchsorted
                indices = torch.searchsorted(cum_probs, markers, right=True)
                # Clamp indices to be within valid range [0, N-1]
                indices = torch.clamp(indices, 0, self.num_particles - 1)

                # Gather resampled particles. Use batch_gather equivalent
                # indices need shape [B, num_resampled, 1] for gather
                indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.state_dim)
                standard_particles = torch.gather(particles, 1, indices_expanded)

                # Detach resampled particles from computation graph (as in original TF code)
                standard_particles = standard_particles.detach()

                # Motion update for resampled particles (on device)
                standard_particles = self.motion_update(a_batch[:, i], standard_particles)

                # Measurement update for resampled particles (on device)
                meas = self.measurement_update(encodings[:, i], standard_particles)
                # Initial weight is uniform for resampled, then multiplied by measurement likelihood
                standard_particle_probs = torch.ones(B, num_resampled, device=device, dtype=dtype) * meas

                combined_particles_list.append(standard_particles)
                combined_probs_list.append(standard_particle_probs)


            # --- Proposing Step (if needed) ---
            if num_proposed > 0:
                # Propose new particles (on device)
                proposed_particles = self.propose_particles(encodings[:, i], num_proposed)
                # Measurement update for proposed particles (on device)
                meas_proposed = self.measurement_update(encodings[:, i], proposed_particles)
                # Initial weight is uniform for proposed, then multiplied by measurement likelihood
                proposed_particle_probs = torch.ones(B, num_proposed, device=device, dtype=dtype) * meas_proposed

                combined_particles_list.append(proposed_particles)
                combined_probs_list.append(proposed_particle_probs)

            # --- Combine Resampled and Proposed ---
            if not combined_particles_list: # Should not happen if num_particles > 0
                 # If num_particles is 0, return empty lists? Or handle earlier.
                 # If num_resampled=0 and num_proposed=0, this means num_particles=0.
                 # Let's assume num_particles > 0.
                 if self.num_particles > 0:
                     raise ValueError("No particles generated in loop, but num_particles > 0.")
                 else: # Handle num_particles = 0 case gracefully
                     particles = torch.empty((B, 0, self.state_dim), device=device, dtype=dtype)
                     particle_probs = torch.empty((B, 0), device=device, dtype=dtype)

            else:
                particles = torch.cat(combined_particles_list, dim=1)
                particle_probs = torch.cat(combined_probs_list, dim=1)

                # Normalize probabilities (add epsilon for stability)
                prob_sum = particle_probs.sum(dim=1, keepdim=True)
                particle_probs = particle_probs / (prob_sum + 1e-16)

            # Append current step results (tensors on device)
            particle_list.append(particles)
            particle_probs_list.append(particle_probs)

        # Stack results along time dimension T
        # Resulting tensors will be on the device
        particle_list_stacked = torch.stack(particle_list, dim=1) # [B, T, N, 3]
        particle_probs_list_stacked = torch.stack(particle_probs_list, dim=1) # [B, T, N]

        # Compute final state estimate (on device)
        pred_states = self.particles_to_state(particle_list_stacked, particle_probs_list_stacked)

        # Store outputs (tensors on device) for later use if needed (e.g., in loss).
        self.pred_states = pred_states
        self.particle_list = particle_list_stacked
        self.particle_probs_list = particle_probs_list_stacked

        # Return values needed for E2E loss (on device)
        # Note: Original returned last step particles/probs, here returning full history
        return particle_list_stacked, particle_probs_list_stacked, encodings # Match E2E loss needs

    # === Training and prediction routines ===

    def compile_training_stages(self, learning_rate, plot_task):
        """
        Creates a dictionary of training stages.
        Assumes self.means_t, self.stds_t etc. are already set and on the correct device.
        Loss functions expect batches already moved to the device.
        """
        train_stages = {}
        device = self.device # Get device from model
        dtype = next(self.parameters()).dtype # Get dtype from model parameters

        # --- Helper function for plotting (to be passed in dict) ---
        # These lambda functions capture self and other necessary variables
        # They expect the batch to be already on the correct device
        def plot_motion_wrapper(epoch, batch):
            if epoch % 10 == 0:
                self.eval() # Set to eval mode for plotting consistency
                with torch.no_grad():
                    s_batch = batch['s']
                    a_batch = batch['a']
                    initial_state_expanded = s_batch[:, 0:1].expand(-1, self.num_particles, -1)
                    motion_samples = self.motion_update(a_batch[:, 1], initial_state_expanded)
                # Move batch and samples to CPU for plotting function
                batch_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
                self.plot_motion_model(batch_cpu, motion_samples.cpu(), plot_task)

        def plot_measurement_wrapper(epoch, batch):
             if epoch % 10 == 0:
                self.eval()
                with torch.no_grad():
                    s_batch = batch['s']
                    o_batch = batch['o']
                    B = s_batch.size(0)
                    if B <= 1: return # Need B > 1 for this plot

                    test_particles = s_batch[:, 0].unsqueeze(1).expand(B, B, -1)
                    o_first = o_batch[:, 0]
                    # Normalize obs before encoding
                    o_first_norm = (o_first - self.means_t['o'].unsqueeze(0)) / (self.stds_t['o'].unsqueeze(0) + 1e-8)
                    encoding = self.forward_encoder(o_first_norm)

                    encoding_expanded = encoding.unsqueeze(1).expand(B, B, -1)
                    encoding_flat = encoding_expanded.reshape(B * B, -1)
                    test_particles_flat = test_particles.reshape(B * B, 1, -1)
                    measurement_out_flat = self.measurement_update(encoding_flat, test_particles_flat)
                    measurement_out = measurement_out_flat.view(B, B)
                self.plot_measurement_model(measurement_out.cpu()) # Move to CPU

        def plot_proposer_wrapper(epoch, batch):
             if epoch % 10 == 0:
                self.eval()
                with torch.no_grad():
                    o_batch = batch['o']
                    # Normalize obs before encoding
                    o_first_norm = (o_batch[:, 0] - self.means_t['o'].unsqueeze(0)) / (self.stds_t['o'].unsqueeze(0) + 1e-8)
                    encoding = self.forward_encoder(o_first_norm)
                    proposed_particles = self.propose_particles(encoding, self.num_particles)
                # Move batch and samples to CPU for plotting function
                batch_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
                self.plot_particle_proposer(batch_cpu, proposed_particles.cpu(), plot_task)

        def plot_e2e_wrapper(epoch, batch):
             if epoch % 1 == 0: # Plot every epoch for E2E
                self.eval()
                # No need for gradients here, predict handles no_grad
                # Predict requires a batch dict, not just the device batch
                # Need to get a batch from a non-repeating iterator ideally
                # For simplicity, we'll plot using the passed batch (which might be from repeating iterator)
                # Move batch to CPU first for the plotting function
                batch_cpu = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
                self.plot_particle_filter(batch_cpu, plot_task, num_particles_plot=500) # Use CPU batch

        # Stage: Train Odometry Model.
        if self.learn_odom:
            def train_odom_fn(batch):
                # batch is assumed to be on the correct device
                s_batch = batch['s']
                a_batch = batch['a']
                # Expand first state to match particle count (use a reasonable number for training)
                num_train_samples = 1 # Only need one sample if stop_sampling_gradient=True
                initial_state_expanded = s_batch[:, 0:1].expand(-1, num_train_samples, -1)
                # Motion update expects particles on device
                motion_samples = self.motion_update(a_batch[:, 1],
                                                    initial_state_expanded,
                                                    stop_sampling_gradient=True) # Detach noise source
                # compute_sq_distance expects tensors on device, uses self.state_step_sizes_t
                sq_distance = compute_sq_distance(motion_samples, s_batch[:, 1:2], self.state_step_sizes_t)
                return sq_distance.mean() # Mean over batch and particles
            optimizer_odom = torch.optim.Adam(self.mo_transition_model.parameters(), lr=learning_rate)
            train_stages['train_odom'] = {
                'loss_fn': train_odom_fn,
                'optimizer': optimizer_odom,
                'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
                'monitor_losses': ['loss'], # Primary loss is the only one here
                'validation_loss': 'loss',
                'plot': plot_motion_wrapper
                }

        # Stage: Train Motion Noise Generator.
        def train_motion_sampling_fn(batch):
            # batch is assumed to be on the correct device
            s_batch = batch['s']
            a_batch = batch['a']
            # Use self.num_particles for training noise generator
            initial_state_expanded = s_batch[:, 0:1].expand(-1, self.num_particles, -1)
            # Motion update (gradients flow through noise generator)
            motion_samples = self.motion_update(a_batch[:, 1], initial_state_expanded)

            std_val = 0.01 # Fixed standard deviation for loss kernel
            # compute_sq_distance expects tensors on device, uses self.state_step_sizes_t
            sq_distance = compute_sq_distance(motion_samples, s_batch[:, 1:2], self.state_step_sizes_t) # [B, N]
            # Gaussian kernel activation (ensure constants are tensors on device)
            norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
            # Ensure num_particles is float for division
            num_particles_float = float(self.num_particles)
            activations = (1 / num_particles_float) * norm_const * \
                          torch.exp(-sq_distance / (2.0 * std_val ** 2)) # [B, N]
            # Negative log likelihood loss (sum over particles, mean over batch)
            loss = (-torch.log(1e-16 + torch.sum(activations, dim=1))).mean()
            return loss
        optimizer_motion = torch.optim.Adam(self.mo_noise_generator.parameters(), lr=learning_rate)
        train_stages['train_motion_sampling'] = {
            'loss_fn': train_motion_sampling_fn,
            'optimizer': optimizer_motion,
            'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
            'monitor_losses': ['loss'],
            'validation_loss': 'loss',
            'plot': plot_motion_wrapper # Can reuse motion plot
            }

        # Stage: Train Measurement Model.
        def train_measurement_model_fn(batch):
            # batch is assumed to be on the correct device
            s_batch = batch['s']
            o_batch = batch['o']
            B = s_batch.size(0)
            if B <= 1: return torch.tensor(0.0, device=device) # Loss requires B > 1

            # Create a test set by pairing the first state of each sequence with every other.
            # [B, 1, 3] -> [B, B, 3] (repeating along dim 1)
            test_particles = s_batch[:, 0].unsqueeze(1).expand(B, B, -1)
            # Encode the first observation image (on device)
            o_first = o_batch[:, 0] # [B, H, W, C] or [B, C, H, W]
            # Normalize obs before encoding
            # print(o_first.shape)
            # print(self.means_t['o'].shape)
            o_first_norm = (o_first - self.means_t['o']) / (self.stds_t['o'] + 1e-8)
            # print(o_first_norm.shape)
            encoding = self.forward_encoder(o_first_norm) # [B, 128]
            # print(encoding.shape)

            # Expand encoding to match particle structure: [B, 1, 128] -> [B, B, 128]
            encoding_expanded = encoding.unsqueeze(1).expand(B, B, -1)

            # Calculate measurement likelihood (expects encoding [B*B, 128], particles [B*B, 1, 3])
            # Reshape needed for measurement_update if it doesn't handle the extra batch dim
            # Let's reshape inputs before calling measurement_update for simplicity
            encoding_flat = encoding_expanded.reshape(B * B, -1)
            test_particles_flat = test_particles.reshape(B * B, 1, -1) # Add particle dim

            # measurement_update expects [Batch, NumParticles, Dim]
            # Here Batch = B*B, NumParticles = 1
            measurement_out_flat = self.measurement_update(encoding_flat, test_particles_flat) # [B*B, 1]
            measurement_out = measurement_out_flat.view(B, B) # Reshape back to [B, B]

            correct_samples = torch.diag(measurement_out) # [B]
            # Off–diagonals (use mask for clarity)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            incorrect_samples = measurement_out[mask] # Flattened off-diagonal elements [B*(B-1)]

            # Heuristic loss from original TF code
            loss_correct = -torch.log(correct_samples + 1e-16).sum() / B
            num_incorrect = B * (B - 1)
            if num_incorrect > 0:
                 loss_incorrect = -torch.log(1.0 - incorrect_samples + 1e-16).sum() / num_incorrect
            else:
                 loss_incorrect = torch.tensor(0.0, device=device, dtype=dtype)

            loss = loss_correct + loss_incorrect
            return loss

        params_measurement = list(self.encoder_conv.parameters()) + list(self.encoder_linear.parameters()) + \
                             list(self.obs_like_estimator.parameters())
        optimizer_measurement = torch.optim.Adam(params_measurement, lr=learning_rate)
        train_stages['train_measurement_model'] = {
            'loss_fn': train_measurement_model_fn,
            'optimizer': optimizer_measurement,
            'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
            'monitor_losses': ['loss'],
            'validation_loss': 'loss',
            'plot': plot_measurement_wrapper
            }


        # Stage: Train Particle Proposer.
        if self.use_proposer:
            def train_particle_proposer_fn(batch):
                # batch is assumed to be on the correct device
                s_batch = batch['s']
                o_batch = batch['o']
                # Stop gradients into the encoder
                with torch.no_grad():
                    # Normalize obs before encoding
                    o_first_norm = (o_batch[:, 0] - self.means_t['o'].unsqueeze(0)) / (self.stds_t['o'].unsqueeze(0) + 1e-8)
                    encoding = self.forward_encoder(o_first_norm) # [B, 128]

                # Propose particles (on device)
                proposed_particles = self.propose_particles(encoding, self.num_particles) # [B, N, 3]

                std_val = 0.2 # Fixed std dev for loss kernel
                # compute_sq_distance expects tensors on device, uses self.state_step_sizes_t
                # Compare proposed particles to the *true* first state
                sq_distance = compute_sq_distance(proposed_particles,
                                                  s_batch[:, 0:1].expand(-1, self.num_particles, -1), # Expand true state
                                                  self.state_step_sizes_t) # [B, N]

                # Gaussian kernel activation (ensure constants are tensors on device)
                norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
                num_particles_float = float(self.num_particles)
                activations = (1 / num_particles_float) * norm_const * \
                              torch.exp(-sq_distance / (2.0 * std_val ** 2)) # [B, N]
                # Negative log likelihood loss (sum over particles, mean over batch)
                loss = (-torch.log(1e-16 + torch.sum(activations, dim=1))).mean()
                return loss
            optimizer_proposer = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)
            train_stages['train_particle_proposer'] = {
                'loss_fn': train_particle_proposer_fn,
                'optimizer': optimizer_proposer,
                'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
                'monitor_losses': ['loss'],
                'validation_loss': 'loss',
                'plot': plot_proposer_wrapper
                }

        # Stage: End-to-End Training.
        def train_e2e_fn(batch):
            # batch is assumed to be on the correct device
            s_batch = batch['s'] # Ground truth states [B, T, 3]
            # Run the full filter (returns tensors on device)
            # connect_modules now returns stacked lists
            particle_list, particle_probs_list, _ = \
                self.connect_modules(batch)
            # particle_list: [B, T, N, 3], particle_probs_list: [B, T, N]

            std_val = self.particle_std  # Use externally set particle_std
            # compute_sq_distance expects tensors on device, uses self.state_step_sizes_t
            # Compare particle cloud at each step T to true state at that step T
            # Expand true state s_batch to match particle dimensions: [B, T, 3] -> [B, T, 1, 3]
            true_state_expanded = s_batch.unsqueeze(2)
            sq_dist = compute_sq_distance(particle_list, true_state_expanded, self.state_step_sizes_t) # [B, T, N]

            # Gaussian kernel activation (ensure constants are tensors on device)
            norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
            # Weight activations by particle probabilities
            activations = particle_probs_list * norm_const * \
                          torch.exp(-sq_dist / (2.0 * std_val ** 2)) # [B, T, N]

            # Negative log likelihood loss (MLE loss from original)
            # Sum over particles N, then mean over time T and batch B
            log_likelihood_per_step = torch.log(1e-16 + torch.sum(activations, dim=2)) # [B, T]
            mle_loss = -log_likelihood_per_step.mean() # Mean over B and T

            # Additionally monitor the MSE on the last time step prediction
            # Use the stored self.pred_states from connect_modules
            pred_states = self.pred_states # [B, T, 3]
            # compute_sq_distance uses state_step_sizes for weighting
            mse_last_step = compute_sq_distance(pred_states[:, -1:], s_batch[:, -1:], self.state_step_sizes_t).mean()

            # Return primary loss for optimization, and secondary for monitoring
            return mle_loss, mse_last_step

        optimizer_e2e = torch.optim.Adam(self.parameters(), lr=learning_rate) # Optimize all parameters
        train_stages['train_e2e'] = {
            'loss_fn': train_e2e_fn,
            'optimizer': optimizer_e2e,
            'batch_iterator_names': {'train': 'train', 'val': 'val'},
            'monitor_losses': ['mle', 'mse_last'], # Match original names
            'validation_loss': 'mse_last', # Match original validation loss
            'plot': plot_e2e_wrapper
            }

        return train_stages

    def predict(self, batch, num_particles, return_particles=False):
        """
        Runs the particle filter on a batch. Moves data to device.
        Assumes model is already on the correct device.
        """
        self.eval() # Set model to evaluation mode
        if self.device is None:
             raise ValueError("Device not set. Call fit() or manually set self.device and statistics.")
        if self.num_particles is None or self.num_particles != num_particles:
             # print(f"Warning: Overriding model num_particles ({self.num_particles}) with prediction num_particles ({num_particles})")
             self.num_particles = num_particles # Allow overriding for prediction

        # Move input batch to the same device as the model
        batch_device = move_batch_to_device(batch, self.device)

        with torch.no_grad(): # Disable gradient calculation for prediction
            # Run the filter. connect_modules uses self.<stat>_t tensors
            particle_list, particle_probs_list, _ = \
                self.connect_modules(batch_device)
            # pred_states is stored in self.pred_states by connect_modules
            pred_states = self.pred_states # [B, T, 3] (on device)

        if return_particles:
            # Return results (potentially moving back to CPU if needed downstream)
            # For now, return tensors on the device they were computed on.
            return pred_states, particle_list, particle_probs_list
        else:
            return pred_states

    def load_model(self, model_path, model_file='best_validation.pth',
                   statistics_file='statistics.npz', device='cpu'):
        """
        Loads saved parameters and statistics, moving model to the specified device.
        """
        # Load statistics first
        stats_path = os.path.join(model_path, statistics_file)
        if os.path.exists(stats_path):
             stats = np.load(stats_path, allow_pickle=True)
             stats = dict(stats)
             # Convert loaded numpy stats to tensors on the target device
             self.means = stats['means'].item() # Keep original numpy dict if needed elsewhere
             self.stds = stats['stds'].item()
             self.state_step_sizes = stats['state_step_sizes']
             self.state_mins = stats['state_mins']
             self.state_maxs = stats['state_maxs']

             self._stats_to_tensors(device) # Helper to create tensor versions
             print(f"Loaded statistics from {stats_path}")
        else:
             print(f"Warning: Statistics file not found at {stats_path}. Cannot initialize self.*_t tensors.")
             # You might want to raise an error here depending on requirements

        # Load model state dict
        state_dict_path = os.path.join(model_path, model_file)
        if os.path.exists(state_dict_path):
             state_dict = torch.load(state_dict_path, map_location=device) # Load directly to target device
             self.load_state_dict(state_dict)
             self.to(device) # Ensure model is on the device
             self.device = device # Store the device
             print(f"Loaded model state_dict from {state_dict_path} to device {device}")
        else:
             print(f"Warning: Model state_dict file not found at {state_dict_path}.")
             # Raise error or handle appropriately

    def _stats_to_tensors(self, device):
        """Converts loaded numpy statistics to tensors on the specified device."""
        dtype = torch.float32 # Default dtype
        # Determine number of observation channels C from the data if possible,
        # or hardcode if known. From error [..., 4], C=4 seems likely.
        # Let's try to infer from the loaded numpy array shape, falling back to 4.
        try:
            # Attempt to get C from the last dimension of the squeezed numpy array
            num_obs_channels = np.asarray(self.means['o']).squeeze().shape[-1]
            if num_obs_channels <= 1: # Squeeze might remove last dim if C=1
                 num_obs_channels = np.asarray(self.means['o']).squeeze().size
            print(f"DEBUG: Inferred num_obs_channels = {num_obs_channels}")
            if num_obs_channels not in [3, 4]: # Plausibility check
                 print(f"Warning: Inferred num_obs_channels ({num_obs_channels}) seems unusual. Double-check data.")
                 # Fallback or raise error? Let's fallback to 4 based on error msg.
                 # num_obs_channels = 4
        except Exception as e:
            print(f"Warning: Could not infer num_obs_channels from self.means['o']. Defaulting to 4. Error: {e}")
            num_obs_channels = 4 # Default based on previous error [..., 4]

        # --- Convert means ---
        self.means_t = {}
        for k, v_np in self.means.items():
            # Squeeze potential extra dimensions from numpy array before converting
            v_squeezed_np = np.asarray(v_np).squeeze()
            v_tensor = torch.tensor(v_squeezed_np, dtype=dtype, device=device)

            if k == 's': # State: Expect [1, 1, 3]
                if v_tensor.numel() != 3:
                    raise ValueError(f"Expected means['s'] to have 3 elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                self.means_t[k] = v_tensor.view(1, 1, 3)
            elif k == 'o': # Observation: Expect [C]
                if v_tensor.numel() != num_obs_channels:
                     raise ValueError(f"Expected means['o'] to have {num_obs_channels} elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                # Reshape to [C] for direct broadcasting with [B, H, W, C]
                self.means_t[k] = v_tensor.view(num_obs_channels)
            else: # Others (e.g., 'a'): Keep original squeezed tensor shape
                self.means_t[k] = v_tensor

        # --- Convert stds ---
        self.stds_t = {}
        for k, v_np in self.stds.items():
            # Squeeze potential extra dimensions from numpy array before converting
            v_squeezed_np = np.asarray(v_np).squeeze()
            v_tensor = torch.tensor(v_squeezed_np, dtype=dtype, device=device)

            if k == 's': # State: Expect [1, 1, 3]
                if v_tensor.numel() != 3:
                    raise ValueError(f"Expected stds['s'] to have 3 elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                self.stds_t[k] = v_tensor.view(1, 1, 3)
            elif k == 'a': # Action: Expect [1, 1, 3] for motion_update broadcasting
                 if v_tensor.numel() != 3:
                      raise ValueError(f"Expected stds['a'] to have 3 elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                 self.stds_t[k] = v_tensor.view(1, 1, 3) # Use view to force the shape [1, 1, 3]
            elif k == 'o': # Observation: Expect [C]
                if v_tensor.numel() != num_obs_channels:
                     raise ValueError(f"Expected stds['o'] to have {num_obs_channels} elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                # Reshape to [C] for direct broadcasting with [B, H, W, C]
                self.stds_t[k] = v_tensor.view(num_obs_channels)
            else: # Others: Keep original squeezed tensor shape
                self.stds_t[k] = v_tensor

        # Convert others (these are usually 1D arrays)
        self.state_step_sizes_t = torch.tensor(self.state_step_sizes, dtype=dtype, device=device)
        self.state_mins_t = torch.tensor(self.state_mins, dtype=dtype, device=device)
        self.state_maxs_t = torch.tensor(self.state_maxs, dtype=dtype, device=device)
        print(f"Converted statistics to tensors on device {device}")
        # Add debug prints to confirm shapes
        if 'o' in self.means_t: print(f"DEBUG: Final self.means_t['o'] shape: {self.means_t['o'].shape}") # Should print torch.Size([4])
        if 'o' in self.stds_t: print(f"DEBUG: Final self.stds_t['o'] shape: {self.stds_t['o'].shape}") # Should print torch.Size([4])
        if 'a' in self.stds_t: print(f"DEBUG: Final self.stds_t['a'] shape: {self.stds_t['a'].shape}") # Should print torch.Size([1, 1, 3])



    def fit(self, data, model_path, train_individually, train_e2e, split_ratio,
            seq_len, batch_size, epoch_length, num_epochs, patience,
            learning_rate, dropout_keep_ratio, num_particles, particle_std,
            plot_task=None, plot=False):
        """
        Full training loop. Handles device placement.
        """
        # --- Determine Device ---
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("CUDA not available. Using CPU.")
        self.device = device # Store device in model instance

        # --- Move Model to Device ---
        self.to(device)
        dtype = next(self.parameters()).dtype # Get model's dtype (usually float32)
        print(f"Model moved to device: {device}")

        # --- Set parameters ---
        self.particle_std = particle_std
        self.num_particles = num_particles
        # Update dropout probability based on keep_ratio
        if hasattr(self, 'encoder_dropout'):
             self.encoder_dropout.p = 1.0 - dropout_keep_ratio
        if hasattr(self, 'particle_proposer'):
             # Find dropout layer in proposer sequence if necessary
             for layer in self.particle_proposer:
                 if isinstance(layer, nn.Dropout):
                     layer.p = 1.0 - self.proposer_keep_ratio # Use proposer_keep_ratio here
                     break


        # --- Preprocess data and compute statistics ---
        # Assume data_utils return numpy arrays
        data_split = split_data(data, ratio=split_ratio)
        means, stds, state_step_sizes, state_mins, state_maxs = compute_statistics(data_split['train'])

        # Store original numpy stats and convert to tensors on the device
        self.means, self.stds = means, stds
        self.state_step_sizes, self.state_mins, self.state_maxs = state_step_sizes, state_mins, state_maxs
        self._stats_to_tensors(device) # Create tensor versions on the correct device

        # --- Create Batch Iterators ---
        # These iterators yield numpy arrays or tensors (handle in loop)
        epoch_lengths = {'train': epoch_length, 'val': epoch_length * 2} # Steps per epoch
        # Use repeating iterators for training/validation during epochs
        batch_iterators = {
            'train': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=seq_len),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
            'train1': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=2), # For 1-step losses
            'val1': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=2),
            # Add non-repeating iterators for plotting specific examples if needed
            'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len),
        }

        # --- Compile training stages ---
        # Pass only necessary args; stats are now attributes (self.*_t)
        train_stages = self.compile_training_stages(learning_rate, plot_task)

        # --- Save statistics ---
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # Save the original numpy statistics
        np.savez(os.path.join(model_path, 'statistics.npz'),
                 means=self.means, stds=self.stds, state_step_sizes=self.state_step_sizes,
                 state_mins=self.state_mins, state_maxs=self.state_maxs)
        print(f"Saved statistics to {os.path.join(model_path, 'statistics.npz')}")

        # --- Define curriculum ---
        curriculum = []
        if train_individually:
            if self.learn_odom and 'train_odom' in train_stages:
                curriculum.append('train_odom')
            if 'train_motion_sampling' in train_stages:
                 curriculum.append('train_motion_sampling')
            if 'train_measurement_model' in train_stages:
                 curriculum.append('train_measurement_model')
            if self.use_proposer and 'train_particle_proposer' in train_stages:
                curriculum.append('train_particle_proposer')
        if train_e2e and 'train_e2e' in train_stages:
            curriculum.append('train_e2e')

        if not curriculum:
             print("Warning: No training stages selected in the curriculum.")
             return None

        # Initialize log structure based on monitor_losses defined in compile_training_stages
        log = {}
        for stage_name in curriculum:
            stage_info = train_stages[stage_name]
            log[stage_name] = {'train': {}, 'val': {}}
            for loss_key in stage_info['monitor_losses']:
                 log[stage_name]['train'][loss_key] = {'mean': [], 'se': []} # Match original structure
                 log[stage_name]['val'][loss_key] = {'mean': [], 'se': []}

        best_overall_val_loss = float('inf') # Track best loss across all stages for final save

        # --- Training Loop ---
        for stage_name in curriculum:
            print(f"\n--- Starting Training Stage: {stage_name} ---")
            stage_info = train_stages[stage_name]
            best_stage_val_loss = float('inf')
            best_stage_epoch = 0
            epoch = 0
            optimizer = stage_info['optimizer']
            loss_fn = stage_info['loss_fn']
            iter_names = stage_info.get('batch_iterator_names', {})
            train_iter_name = iter_names.get('train', 'train')
            val_iter_name = iter_names.get('val', 'val')
            monitor_keys = stage_info['monitor_losses'] # e.g., ['loss'] or ['mle', 'mse_last']
            validation_loss_key = stage_info['validation_loss'] # e.g., 'loss' or 'mse_last'
            plot_fn = stage_info.get('plot', None)

            # --- Load pre-trained model if available ---
            # Check if a model specifically saved for *this stage* exists
            # This logic might need adjustment depending on whether stages build sequentially
            # For now, we just load the overall best model before starting a stage
            save_path_overall = os.path.join(model_path, 'best_validation.pth')
            if os.path.exists(save_path_overall):
                 try:
                     self.load_state_dict(torch.load(save_path_overall, map_location=device))
                     print(f"  Loaded best model state from: {save_path_overall}")
                 except Exception as e:
                     print(f"  Error loading previous best model state: {e}. Starting stage with current parameters.")
            else:
                 print(f"  No previous best model found. Starting stage {stage_name} with current parameters.")


            while epoch < num_epochs and (epoch - best_stage_epoch) < patience:
                # Dictionary to store losses for the current epoch
                epoch_loss_lists = {'train': {k: [] for k in monitor_keys}, 'val': {k: [] for k in monitor_keys}}

                for phase in ['train', 'val']:
                    is_train = phase == 'train'
                    self.train(is_train) # Set train/eval mode

                    iterator = batch_iterators[train_iter_name if is_train else val_iter_name]
                    num_steps = epoch_lengths[phase]

                    for step in range(num_steps):
                        batch = next(iterator)
                        # Move batch data to the device
                        batch_device = move_batch_to_device(batch, device)

                        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

                        with torch.set_grad_enabled(is_train):
                            # Loss function returns tuple: (primary_loss, monitor1, monitor2, ...)
                            output = loss_fn(batch_device)
                            if isinstance(output, tuple):
                                primary_loss = output[0]
                                monitor_values = output # Keep all returned values
                            else:
                                primary_loss = output
                                monitor_values = (output,) # Make it a tuple

                            if is_train and torch.is_tensor(primary_loss) and primary_loss.requires_grad:
                                primary_loss.backward()
                                # Optional: Gradient clipping
                                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                                optimizer.step()

                        # Store monitor values from the output tuple
                        for i, key in enumerate(monitor_keys):
                            if i < len(monitor_values):
                                epoch_loss_lists[phase][key].append(monitor_values[i].item())
                            else:
                                epoch_loss_lists[phase][key].append(0.0) # Should not happen if monitor_keys matches output

                # Calculate average and SE for epoch logs (matching original format)
                txt = "" # For printing epoch summary
                current_val_loss = float('inf') # Reset for this epoch
                for key in monitor_keys:
                    txt += f'{key}: '
                    for phase in ['train', 'val']:
                        losses = epoch_loss_lists[phase][key]
                        if losses:
                            mean_loss = np.mean(losses)
                            se_loss = np.std(losses, ddof=1) / np.sqrt(len(losses)) if len(losses) > 1 else 0.0
                            log[stage_name][phase][key]['mean'].append(mean_loss)
                            log[stage_name][phase][key]['se'].append(se_loss)
                            txt += f'{mean_loss:.4f}+-{se_loss:.4f}/'
                            # Get the validation loss for comparison
                            if phase == 'val' and key == validation_loss_key:
                                current_val_loss = mean_loss
                        else:
                            # Append NaN or handle appropriately if no data for phase/key
                            log[stage_name][phase][key]['mean'].append(np.nan)
                            log[stage_name][phase][key]['se'].append(np.nan)
                            txt += f'N/A/'
                    txt = txt[:-1] + ' -- ' # Replace last '/' with separator


                # Check for improvement and save best model
                save_path_stage = os.path.join(model_path, f'best_val_{stage_name}.pth') # Stage-specific best (optional)

                if current_val_loss < best_stage_val_loss:
                    best_stage_val_loss = current_val_loss
                    best_stage_epoch = epoch
                    # Save stage-specific best (optional, might overwrite if stages are repeated)
                    # torch.save(self.state_dict(), save_path_stage)
                    # Always check against overall best and save that
                    if current_val_loss < best_overall_val_loss:
                         best_overall_val_loss = current_val_loss
                         torch.save(self.state_dict(), save_path_overall)
                         print(f"  * Overall: New best validation loss: {current_val_loss:.4f}. Saved model to {save_path_overall}")
                    txt_prefix = f"epoch {epoch:3d} >> "
                else:
                    txt_prefix = f"epoch {epoch:3d} == "

                # Print epoch summary
                print(f"{txt_prefix}Stage: {stage_name} -- {txt}Patience: {epoch - best_stage_epoch}/{patience}")


                # Optional plotting
                if plot and plot_fn:
                     try:
                         # Get a batch from a non-repeating iterator for consistent plotting
                         plot_batch = next(batch_iterators['val_ex'])
                         plot_batch_device = move_batch_to_device(plot_batch, device)
                         plot_fn(epoch, plot_batch_device) # Pass device batch to wrapper
                     except StopIteration:
                         print("Warning: Plotting iterator exhausted.")
                         # Reset iterator if needed:
                         batch_iterators['val_ex'] = make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len)
                     except Exception as e:
                         print(f"Error during plotting: {e}")


                epoch += 1

            # End of stage: Restore the overall best model found so far
            if os.path.exists(save_path_overall):
                 print(f"--- End of Stage {stage_name}. Restoring overall best model ({save_path_overall}) ---")
                 self.load_state_dict(torch.load(save_path_overall, map_location=device))
            else:
                 print(f"--- End of Stage {stage_name}. No best model saved yet. ---")


        print("\n--- Training Finished ---")
        # Final load of the best model
        if os.path.exists(save_path_overall):
             print(f"Loading final best model from {save_path_overall}")
             self.load_state_dict(torch.load(save_path_overall, map_location=device))

        return log # Return training log

    # === Plotting functions (adapted to PyTorch, matching dpf_old.py style) ===

    def plot_motion_model(self, batch_cpu, motion_samples_cpu, task):
        """Plots motion model samples vs ground truth."""
        # batch_cpu: dict of numpy arrays from CPU
        # motion_samples_cpu: numpy array [B, N, 3] from CPU
        plt.figure('Motion Model')
        plt.clf() # Clear figure
        ax = plt.gca()
        plot_maze(task, ax=ax) # Pass axis

        num_to_plot = min(motion_samples_cpu.shape[0], 10)
        for i in range(num_to_plot):
            s_motion = motion_samples_cpu[i] # Samples for example i [N, 3]
            s_true = batch_cpu['s'][i]      # True states for example i [T, 3] (T=2 here)

            # Plot particle cloud after motion update (blue, thin)
            ax.quiver(s_motion[:, 0], s_motion[:, 1],
                       np.cos(s_motion[:, 2]), np.sin(s_motion[:, 2]),
                       color='blue', width=0.001, scale=100, alpha=0.5)
            # Plot initial state (t=0) (black, thick)
            ax.quiver(s_true[0, 0], s_true[0, 1], np.cos(s_true[0, 2]), np.sin(s_true[0, 2]),
                       color='black', scale=50, width=0.003, label='Start (t=0)' if i == 0 else "")
            # Plot target state (t=1) (red, thick)
            ax.quiver(s_true[1, 0], s_true[1, 1], np.cos(s_true[1, 2]), np.sin(s_true[1, 2]),
                       color='red', scale=50, width=0.003, label='Target (t=1)' if i == 0 else "")

        ax.set_aspect('equal')
        ax.set_title('Motion Model (Blue: Samples)')
        ax.legend(fontsize='small')
        show_pause(pause=0.01)

    def plot_measurement_model(self, measurement_model_out_cpu):
        """Plots the measurement likelihood matrix."""
        # measurement_model_out_cpu: numpy array [B, B] from CPU
        plt.figure('Measurement Model Output')
        plt.clf() # Clear figure
        ax = plt.gca()
        im = ax.imshow(measurement_model_out_cpu, interpolation="nearest", cmap="coolwarm")
        ax.set_title('Measurement Likelihood (Obs vs State)')
        ax.set_xlabel('Particle State Index (from Batch)')
        ax.set_ylabel('Observation Index (from Batch)')
        plt.colorbar(im, ax=ax)
        show_pause(pause=0.01)

    def plot_particle_proposer(self, batch_cpu, proposed_particles_cpu, task):
        """Plots proposed particles vs true initial state."""
        # batch_cpu: dict of numpy arrays from CPU
        # proposed_particles_cpu: numpy array [B, N, 3] from CPU
        plt.figure('Particle Proposer')
        plt.clf() # Clear figure
        ax = plt.gca()
        plot_maze(task, ax=ax)

        num_to_plot = min(proposed_particles_cpu.shape[0], 10)
        for i in range(num_to_plot):
            s_proposed = proposed_particles_cpu[i] # Proposed particles [N, 3]
            s_true_initial = batch_cpu['s'][i, 0] # True state at t=0 [3]
            color = np.random.uniform(0.0, 1.0, 3) # Generate random color

            # Plot proposed particle cloud (random color, thin)
            ax.quiver(s_proposed[:, 0], s_proposed[:, 1],
                       np.cos(s_proposed[:, 2]), np.sin(s_proposed[:, 2]),
                       color=color, width=0.001, scale=100, alpha=0.5)
            # Plot true initial state (t=0) (same random color, thick)
            ax.quiver(s_true_initial[0], s_true_initial[1],
                       np.cos(s_true_initial[2]), np.sin(s_true_initial[2]),
                       color=color, scale=50, width=0.003, label=f'True State {i}' if i < 3 else "") # Label first few

        ax.set_aspect('equal')
        ax.set_title('Particle Proposer (Colors: Proposed Clouds + True State)')
        ax.legend(fontsize='small')
        show_pause(pause=0.01)

    def plot_particle_filter(self, batch_cpu, task, num_particles_plot=500):
        """Plots the filter evolution for one example from the batch."""
        print(f"Plotting particle filter for task {task}...")
        # Ensure model is on the correct device for prediction
        if self.device is None:
             print("Warning: Device not set, attempting prediction on CPU.")
             device = torch.device('cpu')
             self.to(device) # Move model temporarily if needed
             self._stats_to_tensors(device) # Ensure stats are tensors on CPU
        else:
             device = self.device

        # Run prediction on the first example of the batch
        # Need to convert numpy batch back to tensors for predict
        batch_first_tensor = {k: torch.from_numpy(v[0:1]).float() if isinstance(v, np.ndarray) else v for k, v in batch_cpu.items()}

        pred, particle_list, particle_probs_list = self.predict(batch_first_tensor, num_particles=num_particles_plot, return_particles=True)

        # Move results to CPU for plotting
        pred_cpu = pred[0].detach().cpu().numpy() # Remove batch dim [T, 3]
        particles_cpu = particle_list[0].detach().cpu().numpy() # [T, N, 3]
        probs_cpu = particle_probs_list[0].detach().cpu().numpy() # [T, N]
        true_states_cpu = batch_cpu['s'][0] # True states for the first example [T, 3]

        T = particles_cpu.shape[0] # Number of time steps T
        N = particles_cpu.shape[1] # Number of particles N

        # Plotting parameters from dpf_old.py
        head_scale = 1.5
        quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 40., 'width': 0.003,
                       'headlength': 5 * head_scale, 'headwidth': 3 * head_scale,
                       'headaxislength': 4.5 * head_scale}
        marker_kwargs = {'markersize': 4.5, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}
        color_list = plt.cm.tab10(np.linspace(0, 1, 10))
        # Use a consistent color for the PF prediction line (e.g., color_list[1] or [2])
        pf_color = color_list[1] # Example: Blue

        num_rows = 4
        num_cols = 5
        num_plots = num_rows * num_cols
        num_steps_to_plot = min(T, num_plots -1) # Reserve last plot for trajectory

        fig = plt.figure(f"Particle Filter Evolution (Example 0, N={N})", figsize=[12, 9.6])
        fig.clf() # Clear figure

        for i in range(num_steps_to_plot): # Plot each step up to num_plots - 1
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.cla()
            plot_maze(task, margin=5, linewidth=0.5, ax=ax) # Pass axis to plot_maze

            s_particle = particles_cpu[i] # [N, 3]
            s_probs = probs_cpu[i] # [N]

            # Clim from dpf_old.py
            clim = [0.0, 2.0 / N]

            # Plot particles, colored by probability
            ax.quiver(s_particle[:, 0], s_particle[:, 1],
                      np.cos(s_particle[:, 2]), np.sin(s_particle[:, 2]),
                      s_probs, cmap='viridis_r', clim=clim,
                      alpha=1.0, **quiv_kwargs) # Use exact kwargs

            # Plot true state at step i
            current_state = true_states_cpu[i]
            ax.quiver(current_state[0], current_state[1],
                      np.cos(current_state[2]), np.sin(current_state[2]),
                      color="red", **quiv_kwargs) # Use exact kwargs
            # Plot true state marker
            ax.plot(current_state[0], current_state[1], 'or', **marker_kwargs) # Use exact kwargs

            ax.set_title(f'Step {i}')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        # Final plot shows full trajectory
        ax = fig.add_subplot(num_rows, num_cols, num_plots) # Use last subplot
        ax.cla()
        plot_maze(task, margin=5, linewidth=0.5, ax=ax)
        # Plot true trajectory
        ax.plot(true_states_cpu[:T, 0], true_states_cpu[:T, 1], '-', linewidth=0.6, color='red', label='True Traj.')
        # Plot predicted trajectory
        ax.plot(pred_cpu[:T, 0], pred_cpu[:T, 1], '-', linewidth=0.6, color=pf_color, label='PF Traj.')
        # Mark start points
        ax.plot(true_states_cpu[0, 0], true_states_cpu[0, 1], '.', color='red', markersize=3)
        ax.plot(pred_cpu[0, 0], pred_cpu[0, 1], '.', color=pf_color, markersize=3)
        ax.set_title(f'Full Trajectory (T={T})')
        ax.legend(fontsize='small', loc='best') # Use 'best' location
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')


        plt.tight_layout(pad=0.5) # Adjust subplot spacing
        show_pause(pause=0.01)

