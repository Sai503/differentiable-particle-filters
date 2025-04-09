import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

# Assuming these utils handle torch tensors correctly or return numpy arrays
# that we will convert.
from utils.data_utils_kitti import (wrap_angle, compute_statistics, split_data,
                                    make_batch_iterator, make_repeating_batch_iterator,
                                    rotation_matrix, load_data_for_stats) # Use KITTI specific utils
from utils.method_utils import atan2, compute_sq_distance
from utils.plotting_utils import plot_maze, show_pause # plot_maze might not be relevant for KITTI

# Helper function to move batch data to the target device (same as before)
def move_batch_to_device(batch, device):
    """Moves all tensor values in a dictionary to the specified device."""
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            try:
                # Convert numpy arrays to tensors on the device (float32 default)
                batch_device[key] = torch.from_numpy(value).float().to(device)
            except TypeError: # Handle non-numeric numpy arrays if necessary
                 batch_device[key] = value # Keep as is
        else:
            batch_device[key] = value # Keep non-tensor/numpy types
    return batch_device

# A learnable noise scaler (same as before)
class NoiseScaler(nn.Module):
    def __init__(self):
        super(NoiseScaler, self).__init__()
        self.scaler = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        # Multiply input x by exp(10 * scaler)
        # Using torch.exp for differentiability
        return x * torch.exp(10 * self.scaler)


class DPF(nn.Module):
    def __init__(self, init_with_true_state, learn_odom, use_proposer,
                 propose_ratio, proposer_keep_ratio, min_obs_likelihood,
                 learn_gaussian_mle, dropout_keep_prob=1.0): # Added dropout_keep_prob for consistency
        """
        PyTorch version of the Deep Particle Filter for KITTI.
        """
        super(DPF, self).__init__()
        self.init_with_true_state = init_with_true_state
        self.learn_odom = learn_odom
        self.use_proposer = use_proposer and (not init_with_true_state)
        self.propose_ratio = propose_ratio if not init_with_true_state else 0.0
        self.min_obs_likelihood = min_obs_likelihood
        self.learn_gaussian_mle = learn_gaussian_mle
        self.dropout_keep_prob = dropout_keep_prob # For encoder dropout
        self.proposer_keep_ratio = proposer_keep_ratio # For proposer dropout

        # KITTI specific dimensions
        self.state_dim = 5 # (x, y, theta, v, theta_dot)
        self.action_dim = 3 # (Not explicitly used in this motion model?)
        self.observation_dim = 6 # (From KITTI data utils)

        # Placeholders for training parameters
        self.num_particles = None
        self.particle_std = None # Used in E2E loss kernel

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
        # --- MEASUREMENT MODEL ---
        # Encoder: a convolutional network to encode input “images.”
        # Input shape: [B, observation_dim, H, W] (e.g., [B, 6, 50, 150])
        # TODO: Make compute_encoder_output_size dynamic instead of hardcoding
        self._encoder_output_flat_size = self._compute_encoder_output_size_dynamic()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.observation_dim, 16, kernel_size=7, stride=1, padding=3), # 50x150 -> 50x150
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=(1,2), padding=2), # 50x150 -> 50x75
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=(1,2), padding=2), # 50x75 -> 50x38
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=(2,2), padding=2), # 50x38 -> 25x19
            nn.ReLU(),
        )
        self.encoder_flatten = nn.Flatten()
        # Dropout is applied in forward_encoder
        self.encoder_dropout = nn.Dropout(p=1.0 - self.dropout_keep_prob)
        self.encoder_linear = nn.Linear(self._encoder_output_flat_size, 128)


        # Observation likelihood estimator: maps concatenated (encoding, transformed state) to a probability.
        # Input: encoding (128) + transformed state (2 dims: norm_v, norm_theta_dot) = 130
        self.obs_like_estimator = nn.Sequential(
            nn.Linear(128 + 2, 128), # Adjusted input size
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output in (0, 1)
        )

        # --- MOTION NOISE GENERATOR ---
        # Input depends on learn_gaussian_mle
        # If MLE: input is normalized (v, theta_dot) -> 2 dims
        # If not MLE: input is random noise -> 2 dims (for v, theta_dot noise)
        mo_noise_input_dim = 2
        # Output depends on learn_gaussian_mle
        # If MLE: output is (mean_v, log_std_v, mean_theta_dot, log_std_theta_dot) -> 4 dims
        # If not MLE: output is (noise_v, noise_theta_dot) -> 2 dims
        mlp_out = 4 if self.learn_gaussian_mle else 2
        self.mo_noise_generator = nn.Sequential(
            nn.Linear(mo_noise_input_dim, 32), # Adjusted input size
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, mlp_out)
        )

        # --- ODOMETRY MODEL (Optional, seems less common for KITTI PF?) ---
        # Original TF code didn't seem to use learn_odom=True for KITTI examples.
        # If needed, define it here. Input would likely be state + action.
        if self.learn_odom:
             # Example: Input state (5) + action (3?) = 8
             odom_input_dim = self.state_dim # + self.action_dim (if actions are used)
             self.mo_transition_model = nn.Sequential(
                 nn.Linear(odom_input_dim, 128),
                 nn.ReLU(),
                 nn.Linear(128, 128),
                 nn.ReLU(),
                 nn.Linear(128, 128),
                 nn.ReLU(),
                 nn.Linear(128, self.state_dim)
             )
        else:
             self.mo_transition_model = None


        # --- PARTICLE PROPOSER ---
        if self.use_proposer:
            self.particle_proposer = nn.Sequential(
                nn.Linear(128, 128), # Input is encoding
                nn.ReLU(),
                nn.Dropout(p=1.0 - self.proposer_keep_ratio), # Use proposer_keep_ratio
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 4), # Output: x_norm, y_norm, cos(theta)_norm, sin(theta)_norm
                nn.Tanh() # Output in [-1, 1]
            )
        else:
            self.particle_proposer = None

        # Learnable noise scalers (Not used in the provided KITTI motion model logic)
        # self.noise_scaler1 = NoiseScaler()
        # self.noise_scaler2 = NoiseScaler()

    def _compute_encoder_output_size_dynamic(self, H=50, W=150):
        """Helper to compute the flattened size after conv layers dynamically."""
        # Create a dummy input tensor (on CPU is fine for shape calculation)
        # Shape: [B, C, H, W]
        dummy_input = torch.zeros(1, self.observation_dim, H, W)
        with torch.no_grad():
            dummy_output = self.encoder_conv(dummy_input)
        return int(np.prod(dummy_output.shape[1:])) # Multiply C_out * H_out * W_out

    def forward_encoder(self, o):
        """
        Encodes observations.
        o: torch.Tensor of shape [B, C, H, W] (e.g., [B, 6, 50, 150]).
           Expected to be normalized and on the correct device already.
        """
        # Ensure float type
        if o.dtype != torch.float32:
             o = o.float()

        x = self.encoder_conv(o)
        x = self.encoder_flatten(x)
        x = self.encoder_dropout(x) # Apply dropout
        x = self.encoder_linear(x)
        return F.relu(x) # Apply final ReLU

    def measurement_update(self, encoding, particles):
        """
        Compute the likelihood for each particle given the observation encoding.
        encoding: [B, 128] (on device)
        particles: [B, N, state_dim] (on device)
        Uses self.means_t['s'] and self.stds_t['s'] (tensors on device).
        """
        if self.means_t is None or self.stds_t is None:
             raise ValueError("Statistics (means_t, stds_t) not initialized or not on device.")

        particle_input = self.transform_particles_as_input(particles) # [B, N, 2] (on device)
        B, N = particles.shape[0], particles.shape[1]

        # Tile encoding for each particle.
        encoding_input = encoding.unsqueeze(1).expand(B, N, -1) # [B, N, 128]

        # Concatenate encoding and transformed particle state
        inp = torch.cat([encoding_input, particle_input], dim=-1) # [B, N, 130]
        inp_flat = inp.reshape(B * N, -1)

        # Pass through likelihood estimator
        out = self.obs_like_estimator(inp_flat).reshape(B, N)

        # Scale the output
        obs_likelihood = out * (1.0 - self.min_obs_likelihood) + self.min_obs_likelihood
        return obs_likelihood

    def transform_particles_as_input(self, particles):
        """
        Normalizes particle states and selects relevant dimensions for measurement model.
        particles: [B, N, state_dim] (on device)
        Uses self.means_t['s'] and self.stds_t['s'] (tensors on device).
        Returns: [B, N, 2] (normalized v, theta_dot)
        """
        # means_t['s'] shape [1, 1, 5], stds_t['s'] shape [1, 1, 5]
        # Add epsilon for stability
        norm_particles = (particles - self.means_t['s']) / (self.stds_t['s'] + 1e-8)
        # Select dimensions 3 (v) and 4 (theta_dot)
        return norm_particles[..., 3:5] # [B, N, 2]

    def propose_particles(self, encoding, num_particles):
        """
        Propose new particles based on the observation encoding.
        encoding: [B, 128] (on device)
        Uses self.state_mins_t, self.state_maxs_t (tensors on device).
        Returns: [B, num_particles, 3] (x, y, theta) - Note: Proposes only pose part.
        """
        if self.state_mins_t is None or self.state_maxs_t is None:
             raise ValueError("State bounds (state_mins_t, state_maxs_t) not initialized or not on device.")
        if self.particle_proposer is None:
             raise RuntimeError("Particle proposer module is not defined.")

        B = encoding.shape[0]
        device = encoding.device
        dtype = encoding.dtype

        if B == 0 or num_particles == 0:
            return torch.empty((B, num_particles, 3), dtype=dtype, device=device)

        # Duplicate encoding for each particle
        encoding_dup = encoding.unsqueeze(1).expand(B, num_particles, -1) # [B, N, 128]
        encoding_flat = encoding_dup.reshape(B * num_particles, -1)

        # Apply proposer network
        proposed_raw = self.particle_proposer(encoding_flat).reshape(B, num_particles, 4) # [B, N, 4]

        # --- Transform the outputs (tanh: -1 to 1) to valid state values ---
        state_mins = self.state_mins_t # Tensor [5] on device
        state_maxs = self.state_maxs_t # Tensor [5] on device

        # Scale and shift for x-coordinate (dim 0)
        range_x = state_maxs[0] - state_mins[0]
        mid_x = (state_maxs[0] + state_mins[0]) / 2.0
        part0 = proposed_raw[:, :, 0:1] * (range_x / 2.0) + mid_x

        # Scale and shift for y-coordinate (dim 1)
        range_y = state_maxs[1] - state_mins[1]
        mid_y = (state_maxs[1] + state_mins[1]) / 2.0
        part1 = proposed_raw[:, :, 1:2] * (range_y / 2.0) + mid_y

        # Compute angle (theta, dim 2) using atan2 from outputs 2 and 3
        # Assuming output[2] is sin component, output[3] is cos component
        part2 = atan2(proposed_raw[:, :, 2:3], proposed_raw[:, :, 3:4]) # atan2(y, x)

        # Concatenate: [B, N, 1] + [B, N, 1] + [B, N, 1] -> [B, N, 3]
        proposed_pose = torch.cat([part0, part1, part2], dim=-1)

        # NOTE: This proposer only outputs pose (x, y, theta).
        # The remaining state dims (v, theta_dot) need initialization.
        # Let's initialize them randomly similar to uniform initialization.
        # Or perhaps sample from a prior? For simplicity, let's use random uniform.
        v_rand = torch.rand(B, num_particles, 1, device=device, dtype=dtype) * \
                 (state_maxs[3] - state_mins[3]) + state_mins[3]
        theta_dot_rand = torch.rand(B, num_particles, 1, device=device, dtype=dtype) * \
                         (state_maxs[4] - state_mins[4]) + state_mins[4]

        proposed_particles = torch.cat([proposed_pose, v_rand, theta_dot_rand], dim=-1) # [B, N, 5]

        return proposed_particles


    def motion_update(self, particles, stop_sampling_gradient=False):
        """
        Move particles according to constant velocity model with learned noise.
        particles: [B, N, state_dim] (on device)
        Uses self.means_t['s'], self.stds_t['s'], self.state_step_sizes_t (tensors on device).
        Returns:
            moved_particles: [B, N, state_dim]
            delta (if learn_gaussian_mle): [B, N, 4] (mean_v, log_std_v, mean_tdot, log_std_tdot)
            noise (if not learn_gaussian_mle): [B, N, 2] (noise_v, noise_tdot)
        """
        if self.means_t is None or self.stds_t is None or self.state_step_sizes_t is None:
             raise ValueError("Statistics (means_t, stds_t, state_step_sizes_t) not initialized or not on device.")

        time_step = 0.103 # Typical KITTI time step
        B, N, _ = particles.shape
        device = particles.device
        dtype = particles.dtype

        # State components
        x = particles[:, :, 0:1]
        y = particles[:, :, 1:2]
        theta = particles[:, :, 2:3]
        v = particles[:, :, 3:4]
        theta_dot = particles[:, :, 4:5]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # --- Apply Learned Noise ---
        if self.learn_gaussian_mle:
            # Input to noise generator: normalized current velocity components
            norm_v = (v - self.means_t['s'][:, :, 3:4]) / (self.stds_t['s'][:, :, 3:4] + 1e-8)
            norm_theta_dot = (theta_dot - self.means_t['s'][:, :, 4:5]) / (self.stds_t['s'][:, :, 4:5] + 1e-8)
            action_input = torch.cat([norm_v, norm_theta_dot], dim=-1) # [B, N, 2]

            # Predict noise parameters (mean and log_std for v and theta_dot)
            delta = self.mo_noise_generator(action_input.reshape(B * N, -1)).reshape(B, N, 4) # [B, N, 4]

            if stop_sampling_gradient:
                delta = delta.detach()

            # Extract mean and std (ensure std is positive)
            mean_v_noise = delta[:, :, 0:1]
            std_v_noise = torch.exp(delta[:, :, 1:2]) # log_std -> std
            mean_tdot_noise = delta[:, :, 2:3]
            std_tdot_noise = torch.exp(delta[:, :, 3:4]) # log_std -> std

            # Sample noise using reparameterization trick
            epsilon_v = torch.randn_like(mean_v_noise)
            epsilon_tdot = torch.randn_like(mean_tdot_noise)

            # Scale noise by state_step_sizes (using dims 3 and 4)
            sampled_v_noise = (mean_v_noise + epsilon_v * std_v_noise) * self.state_step_sizes_t[3]
            sampled_tdot_noise = (mean_tdot_noise + epsilon_tdot * std_tdot_noise) * self.state_step_sizes_t[4]

            # Apply noise to velocity components
            new_v = v + sampled_v_noise
            new_theta_dot = theta_dot + sampled_tdot_noise
            return_noise = delta # Return the predicted parameters for loss calculation

        else: # Kernel-based loss approach (simpler noise addition)
            # Input to noise generator: random noise
            rand_in = torch.randn(B, N, 2, device=device, dtype=dtype) # [B, N, 2]

            # Predict noise values directly
            noise = self.mo_noise_generator(rand_in.reshape(B * N, -1)).reshape(B, N, 2) # [B, N, 2]

            if stop_sampling_gradient:
                noise = noise.detach()

            # Center the noise (optional, but done in original)
            noise = noise - noise.mean(dim=1, keepdim=True)

            # Scale noise by state_step_sizes (using dims 3 and 4)
            scaled_noise_v = noise[:, :, 0:1] * self.state_step_sizes_t[3]
            scaled_noise_tdot = noise[:, :, 1:2] * self.state_step_sizes_t[4]

            # Apply noise to velocity components
            new_v = v + scaled_noise_v
            new_theta_dot = theta_dot + scaled_noise_tdot
            return_noise = noise # Return the generated noise

        # --- Apply Motion Model (Constant Velocity) ---
        # Use the *original* velocities (v, theta_dot) for deterministic motion part
        delta_x = cos_theta * v * time_step
        delta_y = sin_theta * v * time_step
        delta_theta = theta_dot * time_step

        new_x = x + delta_x
        new_y = y + delta_y
        new_theta = wrap_angle(theta + delta_theta) # wrap_angle should handle tensors

        # Combine new pose and noisy velocities
        moved_particles = torch.cat([new_x, new_y, new_theta, new_v, new_theta_dot], dim=-1)

        return moved_particles, return_noise


    def particles_to_state(self, particle_list, particle_probs_list):
        """
        Computes the weighted mean of particles (full state).
        particle_list: [B, T, N, 5] (on device)
        particle_probs_list: [B, T, N] (on device)
        Returns: [B, T, 5] (on device)
        """
        probs_expanded = particle_probs_list.unsqueeze(-1) # [B, T, N, 1]

        # Weighted mean for x, y (dims 0, 1)
        mean_position = torch.sum(probs_expanded * particle_list[..., :2], dim=2) # [B, T, 2]

        # Weighted mean for orientation (dim 2) using atan2 on sin/cos
        cos_theta = torch.cos(particle_list[..., 2:3])
        sin_theta = torch.sin(particle_list[..., 2:3])
        cos_sum = torch.sum(probs_expanded * cos_theta, dim=2) # [B, T, 1]
        sin_sum = torch.sum(probs_expanded * sin_theta, dim=2) # [B, T, 1]
        mean_orientation = atan2(sin_sum, cos_sum) # atan2(y, x) -> [B, T, 1]

        # Weighted mean for v, theta_dot (dims 3, 4)
        mean_velocity = torch.sum(probs_expanded * particle_list[..., 3:5], dim=2) # [B, T, 2]

        # Concatenate results
        return torch.cat([mean_position, mean_orientation, mean_velocity], dim=-1) # [B, T, 5]


    def connect_modules(self, batch):
        """
        Builds the full particle filtering process for one batch.
        batch: dict with keys 'o', 'a', 's', already moved to the target device.
               'a' (actions) might not be used in this KITTI motion model.
        Uses self.*_t statistics tensors (on device).
        """
        # Get data from batch (already on device)
        o_batch = batch['o'] # Expected shape [B, T, H, W, C=6] or [B, T, C=6, H, W]
        # a_batch = batch['a'] # Actions might not be needed
        s_batch = batch['s'] # True states [B, T, 5]
        device = s_batch.device
        dtype = s_batch.dtype

        B, T = o_batch.shape[0], o_batch.shape[1]
        H, W = o_batch.shape[2], o_batch.shape[3] # Assuming [B, T, H, W, C] or [B, T, C, H, W]

        # --- 1. Process Observations ---
        # Ensure observation tensor is [B, T, C, H, W] and normalized
        if o_batch.dim() == 5 and o_batch.shape[-1] == self.observation_dim:
            o_batch = o_batch.permute(0, 1, 4, 2, 3).contiguous() # [B, T, C, H, W]
        if o_batch.dtype != torch.float32:
            o_batch = o_batch.float()

        # Normalize observations using self.means_t['o'], self.stds_t['o']
        if self.means_t is None or self.stds_t is None:
             raise ValueError("Statistics (means_t, stds_t) not initialized or not on device.")
        # means['o'], stds['o'] should have shape [1, 1, C, 1, 1] or broadcast correctly
        norm_o_batch = (o_batch - self.means_t['o']) / (self.stds_t['o'] + 1e-8)

        # Flatten time dimension and encode
        norm_o_flat = norm_o_batch.reshape(B * T, self.observation_dim, H, W)
        encodings = self.forward_encoder(norm_o_flat).view(B, T, -1)  # [B, T, 128] (on device)

        # --- 2. Initialize Particles ---
        if self.init_with_true_state:
            # Use ground truth state from batch (already on device)
            initial_particles = s_batch[:, 0:1, :].expand(B, self.num_particles, self.state_dim) # [B, N, 5]
        else:
            if self.use_proposer:
                # Propose particles using the first encoding (on device)
                initial_particles = self.propose_particles(encodings[:, 0], self.num_particles) # [B, N, 5]
            else:
                # Random uniform initialization (on device)
                state_mins = self.state_mins_t # [5]
                state_maxs = self.state_maxs_t # [5]
                parts = []
                for d in range(self.state_dim):
                    rand_tensor = torch.rand(B, self.num_particles, 1, device=device, dtype=dtype)
                    parts.append(rand_tensor * (state_maxs[d] - state_mins[d]) + state_mins[d])
                initial_particles = torch.cat(parts, dim=-1) # [B, N, 5]

        # Initial probabilities (uniform, on device)
        initial_particle_probs = torch.ones(B, self.num_particles, device=device, dtype=dtype) / self.num_particles

        # Lists to store history (will contain tensors on device).
        particle_list = [initial_particles]
        particle_probs_list = [initial_particle_probs]
        # Store noise/delta if needed for loss
        motion_noise_info_list = [torch.zeros(B, self.num_particles, 4 if self.learn_gaussian_mle else 2, device=device, dtype=dtype)] # Placeholder for t=0

        particles = initial_particles
        particle_probs = initial_particle_probs

        # --- 3. Loop through Time Steps ---
        for i in range(1, T):
            # Determine numbers of proposed/resampled particles.
            # Note: Original KITTI code might have used fixed ratio or different schedule.
            # Using the same decaying proposer ratio as dpf.py for now.
            num_proposed_float = (self.propose_ratio ** float(i)) * float(self.num_particles)
            num_proposed = int(round(num_proposed_float))
            num_resampled = self.num_particles - num_proposed

            current_particles_list = []
            current_probs_list = []
            current_noise_info_list = []

            # --- Resampling Step (if needed) ---
            if num_resampled > 0:
                # Systematic resampling (on device) - using the efficient version
                markers = torch.linspace(0, 1 - 1/num_resampled, steps=num_resampled,
                                           device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
                random_offset = torch.rand(B, 1, device=device, dtype=dtype) * (1/num_resampled)
                markers = markers + random_offset
                cum_probs = torch.cumsum(particle_probs, dim=1)

                indices = torch.searchsorted(cum_probs, markers, right=True)
                indices = torch.clamp(indices, 0, self.num_particles - 1) # [B, num_resampled]

                # Gather resampled particles
                indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.state_dim) # [B, num_resampled, 5]
                resampled_particles = torch.gather(particles, 1, indices_expanded)

                # Detach resampled particles from graph (standard PF practice)
                resampled_particles = resampled_particles.detach()

                # Motion update for resampled particles (on device)
                # Actions a_batch[:, i] are not used in this motion model
                moved_resampled, noise_info_resampled = self.motion_update(resampled_particles)

                # Measurement update for moved resampled particles (on device)
                meas_resampled = self.measurement_update(encodings[:, i], moved_resampled)
                # Initial weight is uniform, then multiplied by measurement likelihood
                probs_resampled = torch.ones(B, num_resampled, device=device, dtype=dtype) * meas_resampled

                current_particles_list.append(moved_resampled)
                current_probs_list.append(probs_resampled)
                current_noise_info_list.append(noise_info_resampled)


            # --- Proposing Step (if needed) ---
            if num_proposed > 0 and self.use_proposer:
                # Propose new particles (on device)
                proposed_particles = self.propose_particles(encodings[:, i], num_proposed) # [B, num_proposed, 5]

                # Measurement update for proposed particles (on device)
                meas_proposed = self.measurement_update(encodings[:, i], proposed_particles)
                # Initial weight is uniform, then multiplied by measurement likelihood
                probs_proposed = torch.ones(B, num_proposed, device=device, dtype=dtype) * meas_proposed

                current_particles_list.append(proposed_particles)
                current_probs_list.append(probs_proposed)
                # Noise info for proposed particles (placeholder or zero?)
                # Let's use zeros as they didn't go through motion update noise generation here.
                noise_info_proposed = torch.zeros(B, num_proposed, noise_info_resampled.shape[-1], device=device, dtype=dtype)
                current_noise_info_list.append(noise_info_proposed)

            # --- Combine Resampled and Proposed ---
            if not current_particles_list:
                 raise ValueError("No particles generated in loop (num_resampled=0 and num_proposed=0 or proposer disabled).")

            particles = torch.cat(current_particles_list, dim=1) # [B, N, 5]
            particle_probs = torch.cat(current_probs_list, dim=1) # [B, N]
            motion_noise_info = torch.cat(current_noise_info_list, dim=1) # [B, N, 4 or 2]

            # Normalize probabilities (add epsilon for stability)
            particle_probs = particle_probs / (particle_probs.sum(dim=1, keepdim=True) + 1e-16)

            # Append current step results (tensors on device)
            particle_list.append(particles)
            particle_probs_list.append(particle_probs)
            motion_noise_info_list.append(motion_noise_info)

        # Stack results along time dimension T
        particle_list_stacked = torch.stack(particle_list, dim=1) # [B, T, N, 5]
        particle_probs_list_stacked = torch.stack(particle_probs_list, dim=1) # [B, T, N]
        motion_noise_info_stacked = torch.stack(motion_noise_info_list, dim=1) # [B, T, N, 4 or 2]

        # Compute final state estimate (on device)
        pred_states = self.particles_to_state(particle_list_stacked, particle_probs_list_stacked) # [B, T, 5]

        # Store outputs (tensors on device) for potential use in loss or prediction
        self.pred_states = pred_states
        self.particle_list = particle_list_stacked
        self.particle_probs_list = particle_probs_list_stacked
        self.motion_noise_info = motion_noise_info_stacked # Store noise info

        # Return values needed for loss calculations
        return particle_list_stacked, particle_probs_list_stacked, encodings, motion_noise_info_stacked


    # === Training and prediction routines (Adapted from dpf.py) ===

    def compile_training_stages(self, learning_rate, plot_task=None):
        """
        Creates a dictionary of training stages for KITTI.
        Loss functions expect batches already moved to the device.
        """
        train_stages = {}
        device = self.device
        dtype = next(self.parameters()).dtype

        # --- Stage: Train Motion Noise Generator ---
        if self.learn_gaussian_mle:
            # Gaussian MLE Loss for Motion Sampling
            def train_motion_sampling_mle_fn(batch):
                # batch on device, seq_len=2 needed (state_t, state_t+1)
                s_batch = batch['s'] # [B, 2, 5]
                # Run motion update on the first state to get noise predictions
                # Use num_particles=1 for this loss? Or average over N? Let's use N.
                initial_state_expanded = s_batch[:, 0:1, :].expand(-1, self.num_particles, -1) # [B, N, 5]
                # Need noise predictions (delta) without sampling noise affecting gradients
                # Run motion_update just to get delta (mean, log_std)
                _, delta = self.motion_update(initial_state_expanded, stop_sampling_gradient=False) # [B, N, 4]

                # True change in velocity components (target for noise)
                true_delta_v = s_batch[:, 1:2, 3:4] - s_batch[:, 0:1, 3:4] # [B, 1, 1]
                true_delta_tdot = s_batch[:, 1:2, 4:5] - s_batch[:, 0:1, 4:5] # [B, 1, 1]

                # Scale true deltas by state_step_sizes (inverse of scaling applied to noise)
                # state_step_sizes_t should be on device
                target_v_noise = true_delta_v / (self.state_step_sizes_t[3] + 1e-8) # [B, 1, 1]
                target_tdot_noise = true_delta_tdot / (self.state_step_sizes_t[4] + 1e-8) # [B, 1, 1]

                # Predicted noise parameters (average over particles N?)
                # Let's average the predictions over N first
                delta_mean_n = delta.mean(dim=1) # [B, 4]
                pred_mean_v = delta_mean_n[:, 0:1]     # [B, 1]
                pred_log_std_v = delta_mean_n[:, 1:2]  # [B, 1]
                pred_mean_tdot = delta_mean_n[:, 2:3]  # [B, 1]
                pred_log_std_tdot = delta_mean_n[:, 3:4] # [B, 1]

                # Negative Log Likelihood (Gaussian)
                # Loss = 0.5 * [ log(2*pi*std^2) + ((target - mean)/std)^2 ]
                # Loss = 0.5 * [ log(2*pi) + 2*log_std + ((target - mean)/exp(log_std))^2 ]
                # Loss = log_std + 0.5 * ((target - mean) * exp(-log_std))^2 + const
                loss_v = pred_log_std_v + 0.5 * ((target_v_noise - pred_mean_v) * torch.exp(-pred_log_std_v))**2
                loss_tdot = pred_log_std_tdot + 0.5 * ((target_tdot_noise - pred_mean_tdot) * torch.exp(-pred_log_std_tdot))**2

                # Mean loss over batch B
                total_loss = (loss_v + loss_tdot).mean()
                return total_loss

            optimizer_motion = torch.optim.Adam(self.mo_noise_generator.parameters(), lr=learning_rate)
            train_stages['train_motion_sampling'] = {
                'loss_fn': train_motion_sampling_mle_fn,
                'optimizer': optimizer_motion,
                'batch_iterator_names': {'train': 'train2', 'val': 'val2'} # Needs seq_len=2
            }
        else:
            # Kernel-based Loss for Motion Sampling (like dpf.py)
            def train_motion_sampling_kernel_fn(batch):
                # batch on device, seq_len=2 needed
                s_batch = batch['s'] # [B, 2, 5]
                initial_state_expanded = s_batch[:, 0:1, :].expand(-1, self.num_particles, -1) # [B, N, 5]

                # Motion update (gradients flow through noise generator)
                # Actions a_batch[:, 1] are not used
                motion_samples, _ = self.motion_update(initial_state_expanded) # [B, N, 5]

                # Compare only velocity components (dims 3, 4) using compute_sq_distance?
                # compute_sq_distance needs state_step_sizes. Let's use only dims 3, 4.
                # Target state: s_batch[:, 1:2, :] -> [B, 1, 5]
                # Need to adapt compute_sq_distance or calculate manually.
                # Manual calculation for dims 3, 4:
                diff_v = (motion_samples[..., 3:4] - s_batch[:, 1:2, 3:4]) / (self.state_step_sizes_t[3] + 1e-8)
                diff_tdot = (motion_samples[..., 4:5] - s_batch[:, 1:2, 4:5]) / (self.state_step_sizes_t[4] + 1e-8)
                sq_distance = diff_v**2 + diff_tdot**2 # [B, N, 1]
                sq_distance = sq_distance.squeeze(-1) # [B, N]

                std_val = 0.01 # Fixed standard deviation for loss kernel
                norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
                activations = (1.0 / self.num_particles) * norm_const * \
                              torch.exp(-sq_distance / (2.0 * std_val ** 2)) # [B, N]

                # Negative log likelihood loss (sum over particles, mean over batch)
                loss = (-torch.log(1e-16 + torch.sum(activations, dim=1))).mean()
                return loss

            optimizer_motion = torch.optim.Adam(self.mo_noise_generator.parameters(), lr=learning_rate)
            train_stages['train_motion_sampling'] = {
                'loss_fn': train_motion_sampling_kernel_fn,
                'optimizer': optimizer_motion,
                'batch_iterator_names': {'train': 'train2', 'val': 'val2'} # Needs seq_len=2
            }


        # --- Stage: Train Measurement Model ---
        # Uses contrastive loss similar to dpf.py
        def train_measurement_model_fn(batch):
            # batch on device, seq_len=1 needed (just first step)
            s_batch = batch['s'] # [B, 1, 5]
            o_batch = batch['o'] # [B, 1, H, W, C] or [B, 1, C, H, W]
            B = s_batch.size(0)
            if B <= 1: return torch.tensor(0.0, device=device) # Loss requires B > 1

            # --- Prepare data for contrastive loss ---
            # Use state at t=0: s_batch[:, 0, :] -> [B, 5]
            # Create pairs: each state with every observation encoding
            # Test particles: true state s_i repeated B times -> [B, B, 5]
            test_particles = s_batch[:, 0, :].unsqueeze(1).expand(B, B, -1) # [B, B, 5]

            # Encode the first observation image (on device)
            o_first = o_batch[:, 0] # [B, H, W, C] or [B, C, H, W]
            # Normalize observation
            if o_first.dim() == 4 and o_first.shape[-1] == self.observation_dim:
                o_first = o_first.permute(0, 3, 1, 2).contiguous()
            if o_first.dtype != torch.float32: o_first = o_first.float()
            norm_o_first = (o_first - self.means_t['o'].squeeze(1)) / (self.stds_t['o'].squeeze(1) + 1e-8) # Adjust stats shape

            encoding = self.forward_encoder(norm_o_first) # [B, 128]

            # Expand encoding: encoding_j repeated B times -> [B, B, 128]
            encoding_expanded = encoding.unsqueeze(0).expand(B, B, -1) # [B, B, 128]

            # --- Calculate measurement likelihood for all pairs ---
            # measurement_update expects [Batch, NumParticles, Dim]
            # Reshape inputs: Batch=B*B, NumParticles=1
            encoding_flat = encoding_expanded.reshape(B * B, -1)
            # Need to add NumParticles dimension to particles: [B, B, 5] -> [B*B, 1, 5]
            test_particles_flat = test_particles.reshape(B * B, 1, -1)

            measurement_out_flat = self.measurement_update(encoding_flat, test_particles_flat) # [B*B, 1]
            measurement_out = measurement_out_flat.view(B, B) # Reshape back to [B, B]

            # --- Contrastive Loss ---
            correct_samples = torch.diag(measurement_out) # [B] (likelihood of o_i with s_i)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            incorrect_samples = measurement_out[mask] # Flattened off-diagonal elements [B*(B-1)] (likelihood of o_i with s_j, i!=j)

            loss_correct = -torch.log(correct_samples + 1e-16).mean()
            loss_incorrect = -torch.log(1.0 - incorrect_samples + 1e-16).mean()
            loss = loss_correct + loss_incorrect
            return loss

        params_measurement = list(self.encoder_conv.parameters()) + list(self.encoder_linear.parameters()) + \
                             list(self.obs_like_estimator.parameters())
        optimizer_measurement = torch.optim.Adam(params_measurement, lr=learning_rate)
        train_stages['train_measurement_model'] = {
            'loss_fn': train_measurement_model_fn,
            'optimizer': optimizer_measurement,
            'batch_iterator_names': {'train': 'train1', 'val': 'val1'} # Needs seq_len=1
        }

        # --- Stage: Train Particle Proposer ---
        if self.use_proposer:
            # Kernel-based loss similar to dpf.py
            def train_particle_proposer_fn(batch):
                # batch on device, seq_len=1 needed
                s_batch = batch['s'] # [B, 1, 5]
                o_batch = batch['o'] # [B, 1, ...]
                B = s_batch.size(0)

                # Encode the first observation image (on device)
                o_first = o_batch[:, 0]
                # Normalize observation
                if o_first.dim() == 4 and o_first.shape[-1] == self.observation_dim:
                    o_first = o_first.permute(0, 3, 1, 2).contiguous()
                if o_first.dtype != torch.float32: o_first = o_first.float()
                norm_o_first = (o_first - self.means_t['o'].squeeze(1)) / (self.stds_t['o'].squeeze(1) + 1e-8)

                # Stop gradients into the encoder
                with torch.no_grad():
                    encoding = self.forward_encoder(norm_o_first) # [B, 128]

                # Propose particles (on device)
                proposed_particles = self.propose_particles(encoding, self.num_particles) # [B, N, 5]

                # Compare proposed particles to the *true* first state (pose part only?)
                # Proposer outputs pose + random v/tdot. Loss should focus on pose.
                # Let's compare only x, y, theta (dims 0, 1, 2)
                true_state_pose = s_batch[:, 0:1, 0:3].expand(-1, self.num_particles, -1) # [B, N, 3]
                proposed_pose = proposed_particles[..., 0:3] # [B, N, 3]

                # Use compute_sq_distance, providing only step sizes for x, y, theta
                step_sizes_pose = self.state_step_sizes_t[0:3] # [3]
                sq_distance = compute_sq_distance(proposed_pose, true_state_pose, step_sizes_pose) # [B, N]

                std_val = 0.2 # Fixed std dev for loss kernel
                norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
                activations = (1.0 / self.num_particles) * norm_const * \
                              torch.exp(-sq_distance / (2.0 * std_val ** 2)) # [B, N]

                # Negative log likelihood loss (sum over particles, mean over batch)
                loss = (-torch.log(1e-16 + torch.sum(activations, dim=1))).mean()
                return loss

            optimizer_proposer = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)
            train_stages['train_particle_proposer'] = {
                'loss_fn': train_particle_proposer_fn,
                'optimizer': optimizer_proposer,
                'batch_iterator_names': {'train': 'train1', 'val': 'val1'} # Needs seq_len=1
            }


        # --- Stage: End-to-End Training ---
        # Kernel-based loss similar to dpf.py
        def train_e2e_fn(batch):
            # batch is assumed to be on the correct device, full sequence length
            s_batch = batch['s'] # Ground truth states [B, T, 5]

            # Run the full filter (returns tensors on device)
            particle_list, particle_probs_list, _, _ = self.connect_modules(batch)
            # particle_list: [B, T, N, 5], particle_probs_list: [B, T, N]

            std_val = self.particle_std  # Use externally set particle_std

            # Compare particle cloud at each step T to true state at that step T
            # Expand true state s_batch to match particle dimensions: [B, T, 5] -> [B, T, 1, 5]
            true_state_expanded = s_batch.unsqueeze(2)
            # Use full state_step_sizes_t for distance calculation
            sq_dist = compute_sq_distance(particle_list, true_state_expanded, self.state_step_sizes_t) # [B, T, N]

            # Gaussian kernel activation (ensure constants are tensors on device)
            norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
            # Weight activations by particle probabilities
            activations = particle_probs_list * norm_const * \
                          torch.exp(-sq_dist / (2.0 * std_val ** 2)) # [B, T, N]

            # Negative log likelihood loss
            log_likelihood_per_step = torch.log(1e-16 + torch.sum(activations, dim=2)) # [B, T]
            loss = -log_likelihood_per_step.mean() # Mean over B and T

            # Additionally monitor the MSE on the last time step prediction (full state)
            pred_states = self.pred_states # [B, T, 5] (stored by connect_modules)
            mse_last_step = F.mse_loss(pred_states[:, -1], s_batch[:, -1])

            return loss, mse_last_step

        optimizer_e2e = torch.optim.Adam(self.parameters(), lr=learning_rate) # Optimize all parameters
        train_stages['train_e2e'] = {
            'loss_fn': train_e2e_fn,
            'optimizer': optimizer_e2e,
            'batch_iterator_names': {'train': 'train', 'val': 'val'} # Needs full seq_len
        }

        return train_stages


    def predict(self, batch, num_particles, return_particles=False):
        """
        Runs the particle filter on a batch for prediction. Moves data to device.
        Assumes model is already on the correct device.
        """
        self.eval() # Set model to evaluation mode
        if self.device is None:
             raise ValueError("Device not set. Call fit() or load_model() first.")
        if self.num_particles != num_particles:
             print(f"Warning: Overriding model num_particles ({self.num_particles}) with prediction num_particles ({num_particles})")
             self.num_particles = num_particles # Allow overriding for prediction

        # Move input batch to the same device as the model
        batch_device = move_batch_to_device(batch, self.device)

        with torch.no_grad(): # Disable gradient calculation for prediction
            # Run the filter. connect_modules uses self.*_t tensors and stores results
            particle_list, particle_probs_list, _, _ = self.connect_modules(batch_device)
            pred_states = self.pred_states # [B, T, 5] (on device)

        # Return results, optionally moving to CPU numpy arrays
        if return_particles:
            pred_states_np = pred_states.cpu().numpy()
            particle_list_np = particle_list.cpu().numpy()
            particle_probs_list_np = particle_probs_list.cpu().numpy()
            return pred_states_np, particle_list_np, particle_probs_list_np
        else:
            pred_states_np = pred_states.cpu().numpy()
            return pred_states_np


    def load_model(self, model_path, model_file='best_validation.pth',
                   statistics_file='statistics.npz', device='cpu'):
        """
        Loads saved parameters and statistics, moving model to the specified device.
        """
        # Determine target device
        if isinstance(device, str):
             device = torch.device(device)
        self.device = device # Store the device

        # Load statistics first
        stats_path = os.path.join(model_path, statistics_file)
        if os.path.exists(stats_path):
             # Load numpy stats (allow_pickle needed for dictionaries)
             stats = np.load(stats_path, allow_pickle=True)
             self.means = stats['means'].item() # .item() retrieves dict from 0-d array
             self.stds = stats['stds'].item()
             self.state_step_sizes = stats['state_step_sizes']
             self.state_mins = stats['state_mins']
             self.state_maxs = stats['state_maxs']
             stats.close() # Close the file

             # Convert loaded numpy stats to tensors on the target device
             self._stats_to_tensors(device)
             print(f"Loaded and converted statistics from {stats_path} to device {device}")
        else:
             print(f"Warning: Statistics file not found at {stats_path}. Cannot initialize self.*_t tensors.")
             # Depending on usage, might need to raise an error or handle default stats

        # Load model state dict
        state_dict_path = os.path.join(model_path, model_file)
        if os.path.exists(state_dict_path):
             state_dict = torch.load(state_dict_path, map_location=device) # Load directly to target device
             self.load_state_dict(state_dict)
             self.to(device) # Ensure model parameters are on the device
             print(f"Loaded model state_dict from {state_dict_path} to device {device}")
        else:
             print(f"Warning: Model state_dict file not found at {state_dict_path}.")
             # Raise error or handle appropriately (e.g., random initialization)

        self.eval() # Set to evaluation mode after loading


    def _stats_to_tensors(self, device):
        """Converts loaded numpy statistics to tensors on the specified device."""
        dtype = torch.float32 # Default dtype

        # Convert means and stds (dictionaries of arrays)
        # Add necessary unsqueezed dimensions for broadcasting
        # means['o'], stds['o'] -> [1, 1, C, 1, 1] for [B, T, C, H, W] data
        # means['s'], stds['s'] -> [1, 1, D] for [B, N, D] particles
        self.means_t = {}
        self.stds_t = {}
        for k, v in self.means.items():
            tensor = torch.tensor(v, dtype=dtype, device=device)
            if k == 'o': # Observation stats
                # Assuming original shape was [C, H, W] or similar, needs C dim
                # Add B, T, H, W dims for broadcasting: [1, 1, C, 1, 1]
                if tensor.ndim == 3: # Assume C, H, W
                     tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, 1, C, 1, 1]
                elif tensor.ndim == 1: # Assume C
                     tensor = tensor.reshape(1, 1, -1, 1, 1) # [1, 1, C, 1, 1]
                else:
                     print(f"Warning: Unexpected shape for means['{k}']: {v.shape}. Assuming [1, 1, C, 1, 1].")
                     tensor = tensor.reshape(1, 1, -1, 1, 1)
            elif k == 's': # State stats
                # Add B, N dims for broadcasting: [1, 1, D]
                tensor = tensor.reshape(1, 1, -1) # [1, 1, D]
            # Add other keys ('a'?) if necessary
            self.means_t[k] = tensor

        for k, v in self.stds.items():
            tensor = torch.tensor(v, dtype=dtype, device=device)
            # Add epsilon during conversion for stability if stds might be zero
            # tensor = torch.tensor(v + 1e-8, dtype=dtype, device=device)
            if k == 'o':
                 if tensor.ndim == 3: tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                 elif tensor.ndim == 1: tensor = tensor.reshape(1, 1, -1, 1, 1)
                 else:
                     print(f"Warning: Unexpected shape for stds['{k}']: {v.shape}. Assuming [1, 1, C, 1, 1].")
                     tensor = tensor.reshape(1, 1, -1, 1, 1)
            elif k == 's':
                tensor = tensor.reshape(1, 1, -1)
            self.stds_t[k] = tensor

        # Convert others (state_step_sizes, state_mins, state_maxs are 1D arrays)
        self.state_step_sizes_t = torch.tensor(self.state_step_sizes, dtype=dtype, device=device)
        self.state_mins_t = torch.tensor(self.state_mins, dtype=dtype, device=device)
        self.state_maxs_t = torch.tensor(self.state_maxs, dtype=dtype, device=device)
        print(f"Converted statistics to tensors on device {device}")


    def fit(self, data, model_path, train_individually, train_e2e, split_ratio,
            seq_len, batch_size, epoch_length, num_epochs, patience,
            learning_rate, dropout_keep_prob, # Use dropout_keep_prob here
            num_particles, particle_std,
            plot_task=None, plot=False): # Removed learn_gaussian_mle (already in __init__)
        """
        Full training loop for KITTI DPF. Handles device placement.
        """
        # --- Determine Device ---
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("CUDA not available. Using CPU.")
        self.device = device

        # --- Move Model to Device ---
        self.to(device)
        dtype = next(self.parameters()).dtype
        print(f"Model moved to device: {device}")

        # --- Set parameters ---
        self.particle_std = particle_std
        self.num_particles = num_particles
        # Update dropout probabilities based on keep_prob/ratio
        if hasattr(self, 'encoder_dropout'):
             self.encoder_dropout.p = 1.0 - dropout_keep_prob # Use dropout_keep_prob
        if hasattr(self, 'particle_proposer'):
             # Find dropout layer in proposer sequence
             for layer in self.particle_proposer:
                 if isinstance(layer, nn.Dropout):
                     layer.p = 1.0 - self.proposer_keep_ratio # Use proposer_keep_ratio
                     break

        # --- Preprocess data and compute statistics ---
        # Use KITTI specific compute_statistics
        print("Computing statistics...")
        # Need to load data first if 'data' is just a path/config
        # Assuming 'data' is the loaded data dictionary/structure
        means, stds, state_step_sizes, state_mins, state_maxs = compute_statistics(data)
        print("Statistics computed.")

        # Store original numpy stats and convert to tensors on the device
        self.means, self.stds = means, stds
        self.state_step_sizes, self.state_mins, self.state_maxs = state_step_sizes, state_mins, state_maxs
        self._stats_to_tensors(device) # Create tensor versions on the correct device

        # --- Split Data ---
        # Use KITTI specific split_data
        print("Splitting data...")
        data_split = split_data(data, ratio=split_ratio)
        print("Data split.")

        # --- Create Batch Iterators ---
        # Use KITTI specific iterators
        epoch_lengths = {'train': epoch_length, 'val': int(epoch_length * 0.2) + 1} # Adjust val length?
        print("Creating batch iterators...")
        batch_iterators = {
            'train': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=seq_len),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
            'train1': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=1), # For 1-step losses
            'val1': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=1),
            'train2': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=2), # For 2-step losses
            'val2': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=2),
            # Add non-repeating iterators if needed for final evaluation or plotting
            'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len),
        }
        print("Iterators created.")

        # --- Compile training stages ---
        train_stages = self.compile_training_stages(learning_rate, plot_task)

        # --- Save statistics ---
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        stats_save_path = os.path.join(model_path, 'statistics.npz')
        np.savez(stats_save_path,
                 means=self.means, stds=self.stds, state_step_sizes=self.state_step_sizes,
                 state_mins=self.state_mins, state_maxs=self.state_maxs)
        print(f"Saved statistics to {stats_save_path}")

        # --- Define curriculum ---
        curriculum = []
        # Add stages based on flags and model structure
        if train_individually:
            if 'train_motion_sampling' in train_stages:
                 curriculum.append('train_motion_sampling')
            if 'train_measurement_model' in train_stages:
                 curriculum.append('train_measurement_model')
            if self.use_proposer and 'train_particle_proposer' in train_stages:
                curriculum.append('train_particle_proposer')
            # Add odom stage if self.learn_odom is True and implemented
            # if self.learn_odom and 'train_odom' in train_stages:
            #     curriculum.append('train_odom')
        if train_e2e and 'train_e2e' in train_stages:
            curriculum.append('train_e2e')

        if not curriculum:
             print("Warning: No training stages selected in the curriculum.")
             return None

        log = {stage: {'train': {'loss': [], 'monitor': []}, 'val': {'loss': [], 'monitor': []}} for stage in curriculum}
        best_overall_val_loss = float('inf')

        # --- Training Loop (adapted from dpf.py) ---
        for stage in curriculum:
            print(f"\n--- Starting Training Stage: {stage} ---")
            best_stage_val_loss = float('inf')
            best_stage_epoch = 0
            epoch = 0
            optimizer = train_stages[stage]['optimizer']
            loss_fn = train_stages[stage]['loss_fn']
            iter_names = train_stages[stage].get('batch_iterator_names', {})
            train_iter_name = iter_names.get('train', 'train')
            val_iter_name = iter_names.get('val', 'val')

            # Adjust num_epochs/patience per stage? Or use global ones? Using global for now.
            current_patience = 0
            while epoch < num_epochs and current_patience < patience:
                epoch_log = {'train': {'loss': [], 'monitor': []}, 'val': {'loss': [], 'monitor': []}}
                start_time = datetime.now()

                # --- Training Phase ---
                self.train() # Set model to training mode
                iterator = batch_iterators[train_iter_name]
                num_steps = epoch_lengths['train']
                for step in range(num_steps):
                    batch = next(iterator)
                    batch_device = move_batch_to_device(batch, device)

                    optimizer.zero_grad(set_to_none=True)

                    # Loss function might return one (loss) or two (loss, monitor) values
                    output = loss_fn(batch_device)
                    if isinstance(output, tuple):
                        loss_val, monitor_val = output
                    else:
                        loss_val = output
                        monitor_val = torch.tensor(0.0) # Placeholder

                    if torch.is_tensor(loss_val) and loss_val.requires_grad:
                        loss_val.backward()
                        # Optional: Gradient clipping
                        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_log['train']['loss'].append(loss_val.item())
                        epoch_log['train']['monitor'].append(monitor_val.item())
                    else: # Handle cases where loss might be zero or non-tensor
                        epoch_log['train']['loss'].append(loss_val.item() if torch.is_tensor(loss_val) else float(loss_val))
                        epoch_log['train']['monitor'].append(monitor_val.item())


                # --- Validation Phase ---
                self.eval() # Set model to evaluation mode
                iterator = batch_iterators[val_iter_name]
                num_steps = epoch_lengths['val']
                with torch.no_grad():
                    for step in range(num_steps):
                        batch = next(iterator)
                        batch_device = move_batch_to_device(batch, device)

                        output = loss_fn(batch_device)
                        if isinstance(output, tuple):
                            loss_val, monitor_val = output
                        else:
                            loss_val = output
                            monitor_val = torch.tensor(0.0)

                        epoch_log['val']['loss'].append(loss_val.item() if torch.is_tensor(loss_val) else float(loss_val))
                        epoch_log['val']['monitor'].append(monitor_val.item())

                # --- Epoch Summary & Saving ---
                epoch_time = (datetime.now() - start_time).total_seconds()
                avg_train_loss = np.mean(epoch_log['train']['loss']) if epoch_log['train']['loss'] else 0
                avg_val_loss = np.mean(epoch_log['val']['loss']) if epoch_log['val']['loss'] else 0
                avg_train_monitor = np.mean(epoch_log['train']['monitor']) if epoch_log['train']['monitor'] else 0
                avg_val_monitor = np.mean(epoch_log['val']['monitor']) if epoch_log['val']['monitor'] else 0

                log[stage]['train']['loss'].append(avg_train_loss)
                log[stage]['val']['loss'].append(avg_val_loss)
                log[stage]['train']['monitor'].append(avg_train_monitor)
                log[stage]['val']['monitor'].append(avg_val_monitor)

                save_path_stage = os.path.join(model_path, f'best_val_{stage}.pth')
                save_path_overall = os.path.join(model_path, 'best_validation.pth')

                if avg_val_loss < best_stage_val_loss:
                    best_stage_val_loss = avg_val_loss
                    best_stage_epoch = epoch
                    current_patience = 0 # Reset patience
                    torch.save(self.state_dict(), save_path_stage)
                    print(f"  * Stage {stage}: New best validation loss: {avg_val_loss:.4f}. Saved model.")
                    txt_prefix = f"epoch {epoch:3d} >> "
                    # Check overall best
                    if avg_val_loss < best_overall_val_loss:
                         best_overall_val_loss = avg_val_loss
                         torch.save(self.state_dict(), save_path_overall)
                         print(f"    * Overall: New best validation loss. Saved model.")
                else:
                    current_patience += 1
                    txt_prefix = f"epoch {epoch:3d} == "

                print(f"{txt_prefix}Stage: {stage} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                      f"| Train Mon: {avg_train_monitor:.4f} | Val Mon: {avg_val_monitor:.4f} "
                      f"| Patience: {current_patience}/{patience} | Time: {epoch_time:.2f}s")

                # Optional plotting (needs adaptation for KITTI)
                if plot and epoch % 5 == 0: # Plot every 5 epochs example
                     try:
                         plot_batch = next(batch_iterators['val_ex']) # Use non-repeating iterator
                         self.plot_particle_filter(plot_batch, task=plot_task) # Pass batch and task
                     except Exception as e:
                         print(f"Plotting failed: {e}")

                epoch += 1
            # --- End of Stage ---
            # Restore the best model for this stage before proceeding? Or keep the last?
            # Let's restore the best stage model.
            if os.path.exists(save_path_stage):
                 print(f"--- End of Stage {stage}. Restoring best model from epoch {best_stage_epoch} ---")
                 self.load_state_dict(torch.load(save_path_stage, map_location=device))
            else:
                 print(f"--- End of Stage {stage}. No best model saved. Continuing with current model. ---")


        print("\n--- Training Finished ---")
        # Optionally, load the overall best model found across all stages
        if os.path.exists(save_path_overall):
             print(f"Loading overall best model from {save_path_overall}")
             self.load_state_dict(torch.load(save_path_overall, map_location=device))

        return log


    # === Plotting functions (adapted for KITTI, ensure CPU/NumPy) ===

    def plot_motion_model(self, batch, motion_samples, task=None): # Task might not be relevant
        # Ensure data is on CPU as numpy arrays
        batch_cpu = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
        # motion_samples might be (moved_particles, noise_info) tuple
        if isinstance(motion_samples, tuple):
             s_motion_samples = motion_samples[0].detach().cpu().numpy()
        else:
             s_motion_samples = motion_samples.detach().cpu().numpy()

        state_step_sizes = self.state_step_sizes # Use numpy version for plotting scales

        plt.figure('Motion Model (Velocities)')
        plt.clf()
        num_to_plot = min(s_motion_samples.shape[0], 5) # Plot fewer examples
        for i in range(num_to_plot):
            # Plot particle cloud velocities (v, theta_dot) after motion update
            plt.scatter(s_motion_samples[i, :, 3] / state_step_sizes[3],
                        s_motion_samples[i, :, 4] / state_step_sizes[4],
                        color='blue', s=1, alpha=0.5, label='Samples' if i==0 else "")
            # Plot initial state velocity (t=0)
            batch_s_i = batch_cpu['s'][i] # Get ith item from cpu batch
            plt.scatter(batch_s_i[0, 3] / state_step_sizes[3],
                        batch_s_i[0, 4] / state_step_sizes[4],
                        color='black', s=10, marker='x', label='Start (t=0)' if i==0 else "")
            # Plot target state velocity (t=1)
            plt.scatter(batch_s_i[1, 3] / state_step_sizes[3],
                        batch_s_i[1, 4] / state_step_sizes[4],
                        color='red', s=10, marker='o', label='Target (t=1)' if i==0 else "")
            # Optional: Plot line connecting start and target
            plt.plot([batch_s_i[0, 3] / state_step_sizes[3], batch_s_i[1, 3] / state_step_sizes[3]],
                     [batch_s_i[0, 4] / state_step_sizes[4], batch_s_i[1, 4] / state_step_sizes[4]],
                     color='gray', linestyle='--', linewidth=0.5)

        # Set reasonable limits based on expected velocity ranges?
        # plt.xlim([0, 200]) # Original limits might be too large
        # plt.ylim([-50, 50])
        plt.xlabel('Normalized Translational Velocity (v)')
        plt.ylabel('Normalized Angular Velocity (theta_dot)')
        plt.title('Motion Model Noise (Velocities)')
        plt.legend()
        plt.grid(True)
        # plt.gca().set_aspect('equal') # Aspect ratio might not make sense here
        show_pause(pause=0.01)

    def plot_measurement_model(self, measurement_model_out):
        # Ensure input is on CPU as numpy array
        # measurement_model_out is likely the [B, B] matrix from the loss fn
        out_np = measurement_model_out.detach().cpu().numpy()

        plt.figure('Measurement Model Output (Contrastive)')
        plt.clf()
        plt.imshow(out_np, interpolation="nearest", cmap="coolwarm", vmin=0.0, vmax=1.0)
        plt.title('Measurement Likelihood (Correct Pair = Diagonal)')
        plt.xlabel('Particle State Index (from Batch)')
        plt.ylabel('Observation Index (from Batch)')
        plt.colorbar()
        show_pause(pause=0.01)

        # Optional: Plot input velocities vs likelihood (like original) - needs batch access
        # This requires passing the batch to the plotting function or retrieving it.
        # Example (if batch was available):
        # plt.figure('Measurement Model Input vs Likelihood')
        # plt.clf()
        # batch_s_cpu = batch['s'].cpu().numpy() # Assuming batch is available and on device
        # # Plot likelihood of obs[0] vs state[i]
        # plt.scatter(batch_s_cpu[:, 0, 3], batch_s_cpu[:, 0, 4], # Velocities at t=0
        #             marker='o', c=out_np[0, :], # Likelihood from first row
        #             vmin=0, vmax=1.0, cmap='viridis_r')
        # plt.xlabel('Translational Velocity (v)')
        # plt.ylabel('Angular Velocity (theta_dot)')
        # plt.colorbar(label='Likelihood with Obs[0]')
        # plt.title('Measurement Input (Velocities) vs. Likelihood')
        # show_pause(pause=0.01)


    def plot_particle_proposer(self, batch, proposed_particles, task=None):
         # Ensure data is on CPU as numpy arrays
        batch_cpu = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
        s_samples = proposed_particles.detach().cpu().numpy() # [B, N, 5]

        plt.figure('Particle Proposer (Pose)')
        plt.clf()
        # plot_maze(task) # plot_maze might not be suitable for KITTI XY coordinates
        num_to_plot = min(s_samples.shape[0], 5)
        for i in range(num_to_plot):
            color = plt.cm.viridis(i / num_to_plot)
            # Plot proposed particles (pose part: x, y, theta)
            plt.quiver(s_samples[i, :, 0], s_samples[i, :, 1],
                       np.cos(s_samples[i, :, 2]), np.sin(s_samples[i, :, 2]),
                       color=color, width=0.001, scale=100, alpha=0.3, label='Proposed' if i==0 else "")
            # Plot true state (t=0) pose
            batch_s_i = batch_cpu['s'][i]
            plt.quiver(batch_s_i[0, 0], batch_s_i[0, 1], np.cos(batch_s_i[0, 2]), np.sin(batch_s_i[0, 2]),
                       color='red', scale=50, width=0.003, label='True State (t=0)' if i == 0 else "")

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Particle Proposer (Pose: x, y, theta)')
        plt.legend()
        plt.grid(True)
        show_pause(pause=0.01)


    def plot_particle_filter(self, batch, task=None, num_particles_plot=200):
        """Plots the filter evolution for one example from the batch."""
        print(f"Plotting particle filter evolution...")
        # Ensure model is on the correct device for prediction
        if self.device is None:
             print("Warning: Device not set, attempting prediction on CPU.")
             device = torch.device('cpu')
             self.to(device)
             self._stats_to_tensors(device) # Ensure stats are tensors on CPU
        else:
             device = self.device

        # Run prediction on the first example of the batch
        batch_first = {k: v[0:1] if torch.is_tensor(v) or isinstance(v, np.ndarray) else v for k, v in batch.items()}
        pred_np, particles_np, probs_np = self.predict(batch_first, num_particles=num_particles_plot, return_particles=True)

        # Data is already numpy from predict()
        pred_cpu = pred_np[0] # Remove batch dim -> [T, 5]
        particles_cpu = particles_np[0] # -> [T, N, 5]
        probs_cpu = probs_np[0] # -> [T, N]

        # Get true states for the first example (move if necessary)
        if torch.is_tensor(batch['s']):
             true_states_cpu = batch['s'][0].cpu().numpy() # -> [T, 5]
        else: # Assume numpy
             true_states_cpu = batch['s'][0]

        T = particles_cpu.shape[0] # Number of time steps T
        N = particles_cpu.shape[1] # Number of particles N

        # --- Plotting Trajectory (X-Y) ---
        plt.figure(f"Particle Filter Trajectory (Example 0)", figsize=[8, 8])
        plt.clf()
        ax_traj = plt.gca()
        # Plot true trajectory
        ax_traj.plot(true_states_cpu[:, 0], true_states_cpu[:, 1], '-', linewidth=1.5, color='red', label='True Trajectory')
        # Plot predicted trajectory (mean particle)
        ax_traj.plot(pred_cpu[:, 0], pred_cpu[:, 1], '--', linewidth=1.5, color='blue', label='PF Estimate Trajectory')
        # Mark start points
        ax_traj.plot(true_states_cpu[0, 0], true_states_cpu[0, 1], 'o', color='red', markersize=6, label='Start (True)')
        ax_traj.plot(pred_cpu[0, 0], pred_cpu[0, 1], 'x', color='blue', markersize=6, label='Start (PF)')
        # Mark end points
        ax_traj.plot(true_states_cpu[-1, 0], true_states_cpu[-1, 1], 's', color='red', markersize=6, label='End (True)')
        ax_traj.plot(pred_cpu[-1, 0], pred_cpu[-1, 1], '*', color='blue', markersize=8, label='End (PF)')

        # Optionally plot particle cloud at the last step
        last_particles = particles_cpu[-1] # [N, 5]
        last_probs = probs_cpu[-1] # [N]
        # clim_max = np.percentile(last_probs, 99) * 1.5 + 1e-9
        # ax_traj.scatter(last_particles[:, 0], last_particles[:, 1], s=5, c=last_probs, cmap='viridis_r', alpha=0.3, vmin=0, vmax=clim_max, label='Particles (End)')
        ax_traj.scatter(last_particles[:, 0], last_particles[:, 1], s=5, color='green', alpha=0.1, label='Particles (End)')


        ax_traj.set_xlabel("X coordinate (m)")
        ax_traj.set_ylabel("Y coordinate (m)")
        ax_traj.set_title(f'Trajectory Comparison (T={T})')
        ax_traj.legend(fontsize='small')
        ax_traj.grid(True)
        ax_traj.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        show_pause(pause=0.01)

        # --- Plotting State Evolution Over Time ---
        state_labels = ['X (m)', 'Y (m)', 'Theta (rad)', 'V (m/s)', 'Theta_dot (rad/s)']
        num_states_to_plot = self.state_dim
        plt.figure(f"Particle Filter State Evolution (Example 0)", figsize=[10, 8])
        plt.clf()
        time_steps = np.arange(T)
        for d in range(num_states_to_plot):
            ax_state = plt.subplot(num_states_to_plot, 1, d + 1)
            # Plot true state
            ax_state.plot(time_steps, true_states_cpu[:, d], '-', linewidth=1.5, color='red', label='True')
            # Plot predicted state (mean particle)
            ax_state.plot(time_steps, pred_cpu[:, d], '--', linewidth=1.5, color='blue', label='PF Estimate')

            # Optionally plot particle distribution (e.g., +/- 1 std dev or quantiles)
            # Calculate weighted standard deviation or quantiles for each time step
            mean = pred_cpu[:, d]
            variance = np.sum(probs_cpu * (particles_cpu[..., d] - mean[:, np.newaxis])**2, axis=1)
            std_dev = np.sqrt(variance)
            ax_state.fill_between(time_steps, mean - std_dev, mean + std_dev, color='blue', alpha=0.2, label='PF Std Dev')

            ax_state.set_ylabel(state_labels[d])
            ax_state.grid(True)
            if d == 0:
                ax_state.legend(fontsize='small', loc='upper right')
            if d == num_states_to_plot - 1:
                ax_state.set_xlabel("Time Step")
            else:
                ax_state.set_xticklabels([])

        plt.suptitle("State Evolution Comparison")
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap
        show_pause(pause=0.01)

