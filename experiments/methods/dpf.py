import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
            nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 6 -> 3
            nn.ReLU()
        )
        self.encoder_flatten = nn.Flatten()
        self.encoder_dropout = nn.Dropout(p=1 - self.dropout_keep_prob)
        # dropout output = 32x192
        self.encoder_linear = nn.Linear(192, 128)

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
        o: torch.Tensor of shape [B, 24, 24, 3] or [B, 3, 24, 24].
           Expected to be on the correct device already.
        """
        # Ensure correct shape: [B, 3, 24, 24]
        if o.dim() == 4 and o.size(-1) == 3:
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
        """
        Computes the observation likelihood for each particle.
        encoding: [B, 128] (on device)
        particles: [B, N, 3] (on device)
        Uses self.means_t and self.stds_t which should be tensors on the device.
        """
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
        part2 = atan2(proposed_raw[:, :, 2:3], proposed_raw[:, :, 3:4]) # atan2 should handle tensors

        # --- Concatenate results ---
        proposed_particles = torch.cat([part0, part1, part2], dim=-1) # Shape: [B, num_particles, 3]

        return proposed_particles


    def motion_update(self, actions, particles, stop_sampling_gradient=False):
        """
        Moves particles according to actions with added learned noise.
        actions: [B, 3] (on device)
        particles: [B, N, 3] (on device)
        Uses self.stds_t, self.means_t, self.state_step_sizes_t (tensors on device).
        """
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

            state_delta = state_delta * state_step_sizes  # elementwise scaling
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
        # atan2 should handle tensors
        mean_orientation = atan2(cos_sum, sin_sum) # Note: custom atan2 uses torch.atan2(y, x)
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

        encodings = self.forward_encoder(o_flat).view(B, T, -1)  # [B, T, 128] (on device)

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

            resampled_particles_list = []
            resampled_probs_list = []

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

                resampled_particles_list.append(standard_particles)
                resampled_probs_list.append(standard_particle_probs)


            # --- Proposing Step (if needed) ---
            if num_proposed > 0:
                # Propose new particles (on device)
                proposed_particles = self.propose_particles(encodings[:, i], num_proposed)
                # Measurement update for proposed particles (on device)
                meas_proposed = self.measurement_update(encodings[:, i], proposed_particles)
                # Initial weight is uniform for proposed, then multiplied by measurement likelihood
                proposed_particle_probs = torch.ones(B, num_proposed, device=device, dtype=dtype) * meas_proposed

                resampled_particles_list.append(proposed_particles)
                resampled_probs_list.append(proposed_particle_probs)

            # --- Combine Resampled and Proposed ---
            if not resampled_particles_list: # Should not happen if num_particles > 0
                 raise ValueError("No particles generated in loop.")

            particles = torch.cat(resampled_particles_list, dim=1)
            particle_probs = torch.cat(resampled_probs_list, dim=1)

            # Normalize probabilities (add epsilon for stability)
            particle_probs = particle_probs / (particle_probs.sum(dim=1, keepdim=True) + 1e-16)

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

        # Stage: Train Odometry Model.
        if self.learn_odom:
            def train_odom_fn(batch):
                # batch is assumed to be on the correct device
                s_batch = batch['s']
                a_batch = batch['a']
                # Expand first state to match particle count
                initial_state_expanded = s_batch[:, 0:1].expand(-1, self.num_particles, -1)
                # Motion update expects particles on device
                motion_samples = self.motion_update(a_batch[:, 1],
                                                    initial_state_expanded,
                                                    stop_sampling_gradient=True) # Detach noise source
                # compute_sq_distance expects tensors on device, uses self.state_step_sizes_t
                sq_distance = compute_sq_distance(motion_samples, s_batch[:, 1:2], self.state_step_sizes_t)
                return sq_distance.mean() # Mean over batch and particles
            optimizer_odom = torch.optim.Adam(self.mo_transition_model.parameters(), lr=learning_rate)
            train_stages['train_odom'] = {'loss_fn': train_odom_fn, 'optimizer': optimizer_odom, 'batch_iterator_names': {'train': 'train1', 'val': 'val1'}}

        # Stage: Train Motion Noise Generator.
        def train_motion_sampling_fn(batch):
            # batch is assumed to be on the correct device
            s_batch = batch['s']
            a_batch = batch['a']
            initial_state_expanded = s_batch[:, 0:1].expand(-1, self.num_particles, -1)
            # Motion update (gradients flow through noise generator)
            motion_samples = self.motion_update(a_batch[:, 1], initial_state_expanded)

            std_val = 0.01 # Fixed standard deviation for loss kernel
            # compute_sq_distance expects tensors on device, uses self.state_step_sizes_t
            sq_distance = compute_sq_distance(motion_samples, s_batch[:, 1:2], self.state_step_sizes_t) # [B, N]
            # Gaussian kernel activation (ensure constants are tensors on device)
            norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
            activations = (1 / self.num_particles) * norm_const * \
                          torch.exp(-sq_distance / (2.0 * std_val ** 2)) # [B, N]
            # Negative log likelihood loss (sum over particles, mean over batch)
            loss = (-torch.log(1e-16 + torch.sum(activations, dim=1))).mean()
            return loss
        optimizer_motion = torch.optim.Adam(self.mo_noise_generator.parameters(), lr=learning_rate)
        train_stages['train_motion_sampling'] = {'loss_fn': train_motion_sampling_fn, 'optimizer': optimizer_motion, 'batch_iterator_names': {'train': 'train1', 'val': 'val1'}}

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
            encoding = self.forward_encoder(o_first) # [B, 128]

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

            # Binary cross-entropy like loss
            # Maximize likelihood for correct pairs (diag), minimize for incorrect pairs (off-diag)
            loss_correct = -torch.log(correct_samples + 1e-16).mean() # Average over B
            loss_incorrect = -torch.log(1.0 - incorrect_samples + 1e-16).mean() # Average over B*(B-1)

            loss = loss_correct + loss_incorrect
            return loss

        params_measurement = list(self.encoder_conv.parameters()) + list(self.encoder_linear.parameters()) + \
                             list(self.obs_like_estimator.parameters())
        optimizer_measurement = torch.optim.Adam(params_measurement, lr=learning_rate)
        # Uses full sequence length for better context? Or just first step? Original used seq_len=2 ('train1')
        # Let's assume it needs pairs, so 'train1'/'val1' iterators.
        train_stages['train_measurement_model'] = {'loss_fn': train_measurement_model_fn, 'optimizer': optimizer_measurement, 'batch_iterator_names': {'train': 'train1', 'val': 'val1'}}


        # Stage: Train Particle Proposer.
        if self.use_proposer:
            def train_particle_proposer_fn(batch):
                # batch is assumed to be on the correct device
                s_batch = batch['s']
                o_batch = batch['o']
                # Stop gradients into the encoder
                with torch.no_grad():
                    encoding = self.forward_encoder(o_batch[:, 0]) # [B, 128]

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
                activations = (1 / self.num_particles) * norm_const * \
                              torch.exp(-sq_distance / (2.0 * std_val ** 2)) # [B, N]
                # Negative log likelihood loss (sum over particles, mean over batch)
                loss = (-torch.log(1e-16 + torch.sum(activations, dim=1))).mean()
                return loss
            optimizer_proposer = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)
            # Proposer uses first step image/state, so 'train1'/'val1' iterators seem appropriate.
            train_stages['train_particle_proposer'] = {'loss_fn': train_particle_proposer_fn, 'optimizer': optimizer_proposer, 'batch_iterator_names': {'train': 'train1', 'val': 'val1'}}

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

            # Negative log likelihood loss
            # Sum over particles N, then mean over time T and batch B
            log_likelihood_per_step = torch.log(1e-16 + torch.sum(activations, dim=2)) # [B, T]
            loss = -log_likelihood_per_step.mean() # Mean over B and T

            # Additionally monitor the MSE on the last time step prediction
            # Use the stored self.pred_states from connect_modules
            pred_states = self.pred_states # [B, T, 3]
            # compute_sq_distance uses state_step_sizes for weighting, so it's not exactly MSE unless step_sizes are 1
            # Let's compute simple MSE for monitoring
            mse_last_step = F.mse_loss(pred_states[:, -1], s_batch[:, -1])

            # Return primary loss for optimization, and secondary for monitoring
            return loss, mse_last_step

        optimizer_e2e = torch.optim.Adam(self.parameters(), lr=learning_rate) # Optimize all parameters
        # E2E uses full sequences, so 'train'/'val' iterators.
        train_stages['train_e2e'] = {'loss_fn': train_e2e_fn, 'optimizer': optimizer_e2e, 'batch_iterator_names': {'train': 'train', 'val': 'val'}}

        return train_stages

    def predict(self, batch, num_particles, return_particles=False):
        """
        Runs the particle filter on a batch. Moves data to device.
        Assumes model is already on the correct device.
        """
        self.eval() # Set model to evaluation mode
        if self.device is None:
             raise ValueError("Device not set. Call fit() or manually set self.device and statistics.")
        if self.num_particles != num_particles:
             print(f"Warning: Overriding model num_particles ({self.num_particles}) with prediction num_particles ({num_particles})")
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

        # Convert means and stds (which are dictionaries of arrays)
        self.means_t = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in self.means.items()}
        self.stds_t = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in self.stds.items()}

        # Convert others
        self.state_step_sizes_t = torch.tensor(self.state_step_sizes, dtype=dtype, device=device)
        self.state_mins_t = torch.tensor(self.state_mins, dtype=dtype, device=device)
        self.state_maxs_t = torch.tensor(self.state_maxs, dtype=dtype, device=device)
        print(f"Converted statistics to tensors on device {device}")

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
        if hasattr(self, 'particle_proposer') and hasattr(self.particle_proposer[-6], 'p'): # Assuming dropout is 3rd layer in proposer
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
        batch_iterators = {
            'train': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=seq_len),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
            'train1': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=2), # For 1-step losses
            'val1': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=2),
            # Add non-repeating iterators if needed for final evaluation or plotting
            # 'train_ex': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=seq_len),
            # 'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len),
        }

        # --- Compile training stages ---
        # Pass only necessary args; stats are now attributes (self.*_t)
        train_stages = self.compile_training_stages(learning_rate, plot_task)

        # --- Save statistics ---
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # Save the original numpy statistics
        np.savez(os.path.join(model_path, 'statistics'),
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

        log = {stage: {'train': {'loss': [], 'monitor': []}, 'val': {'loss': [], 'monitor': []}} for stage in curriculum}
        best_overall_val_loss = float('inf') # Track best loss across all stages for final save

        # --- Training Loop ---
        for stage in curriculum:
            print(f"\n--- Starting Training Stage: {stage} ---")
            best_stage_val_loss = float('inf')
            best_stage_epoch = 0
            epoch = 0
            optimizer = train_stages[stage]['optimizer']
            loss_fn = train_stages[stage]['loss_fn']
            # Get correct iterator names for this stage
            iter_names = train_stages[stage].get('batch_iterator_names', {})
            train_iter_name = iter_names.get('train', 'train') # Default to 'train'
            val_iter_name = iter_names.get('val', 'val')       # Default to 'val'

            # --- Load pre-trained model if available ---
            pretrained_model_path = os.path.join(model_path, f'best_val_{stage}.pth')
            if os.path.exists(pretrained_model_path):
                try:
                    self.load_state_dict(torch.load(pretrained_model_path, map_location=device))
                    print(f"  Loaded pre-trained model for {stage} from: {pretrained_model_path}")
                    # skip to next stage
                    continue
                except RuntimeError as e:
                    print(f"  Error loading pre-trained model for {stage}: {e}. Training from current state.")
            else:
                print(f"  No pre-trained model found for {stage}. Training from current state.")

            while epoch < num_epochs and (epoch - best_stage_epoch) < patience:
                epoch_log = {'train': {'loss': [], 'monitor': []}, 'val': {'loss': [], 'monitor': []}}
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
                            # Loss function might return one (loss) or two (loss, monitor) values
                            output = loss_fn(batch_device)
                            if isinstance(output, tuple):
                                loss_val, monitor_val = output
                            else:
                                loss_val = output
                                monitor_val = torch.tensor(0.0) # Placeholder if no monitor value

                            if is_train and torch.is_tensor(loss_val) and loss_val.requires_grad:
                                loss_val.backward()
                                # Optional: Gradient clipping
                                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                                optimizer.step()

                        epoch_log[phase]['loss'].append(loss_val.item())
                        epoch_log[phase]['monitor'].append(monitor_val.item())

                        # Print progress within epoch (optional)
                        # if is_train and (step + 1) % (num_steps // 5) == 0:
                        #     print(f"  Epoch {epoch} [{phase}] Step {step+1}/{num_steps} Loss: {loss_val.item():.4f}")


                # Calculate average losses for the epoch
                avg_train_loss = np.mean(epoch_log['train']['loss']) if epoch_log['train']['loss'] else 0 # Handle empty lists
                avg_val_loss = np.mean(epoch_log['val']['loss']) if epoch_log['val']['loss'] else 0
                avg_train_monitor = np.mean(epoch_log['train']['monitor']) if epoch_log['train']['monitor'] else 0
                avg_val_monitor = np.mean(epoch_log['val']['monitor']) if epoch_log['val']['monitor'] else 0

                # Log epoch results
                log[stage]['train']['loss'].append(avg_train_loss)
                log[stage]['val']['loss'].append(avg_val_loss)
                log[stage]['train']['monitor'].append(avg_train_monitor)
                log[stage]['val']['monitor'].append(avg_val_monitor)

                # Check for improvement and save best model for this stage
                save_path_stage = os.path.join(model_path, f'best_val_{stage}.pth')
                save_path_overall = os.path.join(model_path, 'best_validation.pth') # Overall best

                if avg_val_loss < best_stage_val_loss:
                    best_stage_val_loss = avg_val_loss
                    best_stage_epoch = epoch
                    torch.save(self.state_dict(), save_path_stage)
                    print(f"  * Stage {stage}: New best validation loss: {avg_val_loss:.4f}. Saved model to {save_path_stage}")
                    # Also check if this is the best overall model
                    if avg_val_loss < best_overall_val_loss:
                         best_overall_val_loss = avg_val_loss
                         torch.save(self.state_dict(), save_path_overall)
                         print(f"    * Overall: New best validation loss. Saved model to {save_path_overall}")
                    txt_prefix = f"epoch {epoch:3d} >> "
                else:
                    txt_prefix = f"epoch {epoch:3d} == "

                # Print epoch summary
                print(f"{txt_prefix}Stage: {stage} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                      f"| Train Mon: {avg_train_monitor:.4f} | Val Mon: {avg_val_monitor:.4f} "
                      f"| Patience: {epoch - best_stage_epoch}/{patience}")


                # Optional plotting (ensure plot functions handle device/cpu transfer)
                # if plot and 'plot' in train_stages[stage]:
                #     # You might need to fetch a specific batch for plotting
                #     # plot_batch = next(batch_iterators['val_ex']) # Example: use a non-repeating iterator
                #     # plot_batch_device = move_batch_to_device(plot_batch, device)
                #     # train_stages[stage]['plot'](epoch, plot_batch_device) # Pass device batch
                #     pass

                epoch += 1

            # End of stage: Restore the best model for this stage before proceeding
            if os.path.exists(save_path_stage):
                 print(f"--- End of Stage {stage}. Restoring best model from epoch {best_stage_epoch} ({save_path_stage}) ---")
                 self.load_state_dict(torch.load(save_path_stage, map_location=device))
            else:
                 print(f"--- End of Stage {stage}. No best model saved for this stage. ---")


        print("\n--- Training Finished ---")
        # Optionally, load the overall best model found across all stages
        if os.path.exists(save_path_overall):
             print(f"Loading overall best model from {save_path_overall}")
             self.load_state_dict(torch.load(save_path_overall, map_location=device))

        return log # Return training log
  
    # === Plotting functions (adapted to PyTorch) ===
    # IMPORTANT: Ensure tensors are moved to CPU and converted to NumPy for plotting

    def plot_motion_model(self, batch, motion_samples, task):
        # Ensure batch and motion_samples are on CPU as numpy arrays
        batch_cpu = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
        s_motion_samples = motion_samples.detach().cpu().numpy() # Detach before moving

        plt.figure('Motion Model')
        plt.clf()
        plot_maze(task) # Assumes plot_maze handles task string
        # Plot only a few examples from the batch
        num_to_plot = min(s_motion_samples.shape[0], 10)
        for i in range(num_to_plot):
            # Plot particle cloud after motion update
            plt.quiver(s_motion_samples[i, :, 0], s_motion_samples[i, :, 1],
                       np.cos(s_motion_samples[i, :, 2]), np.sin(s_motion_samples[i, :, 2]),
                       color='blue', width=0.001, scale=100, alpha=0.5) # Added alpha
            # Plot initial state (t=0)
            batch_s_i = batch_cpu['s'][i] # Get ith item from cpu batch
            plt.quiver(batch_s_i[0, 0], batch_s_i[0, 1], np.cos(batch_s_i[0, 2]), np.sin(batch_s_i[0, 2]),
                       color='black', scale=50, width=0.003)
            # Plot target state (t=1)
            plt.quiver(batch_s_i[1, 0], batch_s_i[1, 1], np.cos(batch_s_i[1, 2]), np.sin(batch_s_i[1, 2]),
                       color='red', scale=50, width=0.003)
        plt.gca().set_aspect('equal')
        plt.title('Motion Model (Black: Start, Red: Target, Blue: Samples)')
        show_pause(pause=0.01) # Use show_pause utility

    def plot_measurement_model(self, measurement_model_out):
        # Ensure input is on CPU as numpy array
        s_measurement_model_out = measurement_model_out.detach().cpu().numpy()

        plt.figure('Measurement Model Output')
        plt.clf()
        plt.imshow(s_measurement_model_out, interpolation="nearest", cmap="coolwarm")
        plt.title('Measurement Likelihood (Correct Pair = Diagonal)')
        plt.xlabel('Particle Index (from Batch)')
        plt.ylabel('Observation Index (from Batch)')
        plt.colorbar()
        show_pause(pause=0.01)

    def plot_particle_proposer(self, batch, proposed_particles, task):
         # Ensure batch and proposed_particles are on CPU as numpy arrays
        batch_cpu = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
        s_samples = proposed_particles.detach().cpu().numpy()

        plt.figure('Particle Proposer')
        plt.clf()
        plot_maze(task)
        num_to_plot = min(s_samples.shape[0], 10)
        for i in range(num_to_plot):
            color = plt.cm.viridis(i / num_to_plot) # Use colormap for different examples
            # Plot proposed particles
            plt.quiver(s_samples[i, :, 0], s_samples[i, :, 1],
                       np.cos(s_samples[i, :, 2]), np.sin(s_samples[i, :, 2]),
                       color=color, width=0.001, scale=100, alpha=0.5)
            # Plot true state (t=0)
            batch_s_i = batch_cpu['s'][i]
            plt.quiver(batch_s_i[0, 0], batch_s_i[0, 1], np.cos(batch_s_i[0, 2]), np.sin(batch_s_i[0, 2]),
                       color='red', scale=50, width=0.003, label='True State' if i == 0 else "") # Label once
        plt.gca().set_aspect('equal')
        plt.title('Particle Proposer (Red: True State, Colors: Proposed Clouds)')
        plt.legend()
        show_pause(pause=0.01)

    def plot_particle_filter(self, batch, task, num_particles_plot=500):
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
        batch_first = {k: v[0:1] for k, v in batch.items()} # Select first item, keep batch dim
        pred, particle_list, particle_probs_list = self.predict(batch_first, num_particles=num_particles_plot, return_particles=True)

        # Move results to CPU for plotting
        pred_cpu = pred[0].detach().cpu().numpy() # Remove batch dim
        particles_cpu = particle_list[0].detach().cpu().numpy()
        probs_cpu = particle_probs_list[0].detach().cpu().numpy()
        true_states_cpu = batch['s'][0].cpu().numpy() # Get true states for the first example

        T = particles_cpu.shape[0] # Number of time steps T
        N = particles_cpu.shape[1] # Number of particles N

        # Plotting parameters
        head_scale = 1.5
        quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 40., 'width': 0.003,
                       'headlength': 5 * head_scale, 'headwidth': 3 * head_scale,
                       'headaxislength': 4.5 * head_scale}
        marker_kwargs = {'markersize': 4.5, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}
        color_list = plt.cm.tab10(np.linspace(0, 1, 10))
        pf_color = color_list[1] # Color for PF prediction

        num_rows = 4
        num_cols = 5
        num_plots = num_rows * num_cols

        plt.figure(f"Particle Filter Evolution (Example 0)", figsize=[12, 9.6]) # Adjusted figsize
        plt.clf()

        for i in range(min(T, num_plots)): # Plot each step up to num_plots
            ax = plt.subplot(num_rows, num_cols, i + 1)
            plt.cla()
            plot_maze(task, margin=5, linewidth=0.5, ax=ax) # Pass axis to plot_maze

            if i < T: # Plot particles and true state for step i
                s_particle = particles_cpu[i] # [N, 3]
                s_probs = probs_cpu[i] # [N]

                # Normalize probabilities for color mapping if needed
                # clim_max = max(1e-9, 2.0 * np.mean(s_probs)) # Avoid zero max clim
                clim_max = np.percentile(s_probs, 99) * 1.5 + 1e-9 # Use percentile for robustness

                # Plot particles, colored by probability
                ax.quiver(s_particle[:, 0], s_particle[:, 1],
                          np.cos(s_particle[:, 2]), np.sin(s_particle[:, 2]),
                          s_probs, cmap='viridis_r', clim=[0, clim_max], # Adjusted clim
                          alpha=0.7, **quiv_kwargs) # Added alpha

                # Plot true state at step i
                current_state = true_states_cpu[i]
                ax.quiver(current_state[0], current_state[1],
                          np.cos(current_state[2]), np.sin(current_state[2]),
                          color="red", label='True State' if i == 0 else "", **quiv_kwargs)
                ax.plot(current_state[0], current_state[1], 'or', label='_nolegend_', **marker_kwargs)

                # Plot predicted state (mean particle) at step i
                pred_state = pred_cpu[i]
                ax.quiver(pred_state[0], pred_state[1],
                          np.cos(pred_state[2]), np.sin(pred_state[2]),
                          color=pf_color, label='PF Estimate' if i == 0 else "", **quiv_kwargs)
                ax.plot(pred_state[0], pred_state[1], 'o', color=pf_color, label='_nolegend_', **marker_kwargs)

                ax.set_title(f'Step {i}')

            # Add legend to the first plot
            if i == 0:
                 ax.legend(fontsize='small', loc='upper right')

            # Final plot shows full trajectory
            if i == num_plots - 1 or i == T - 1:
                 ax = plt.subplot(num_rows, num_cols, num_plots) # Use last subplot for trajectory
                 plt.cla()
                 plot_maze(task, margin=5, linewidth=0.5, ax=ax)
                 # Plot true trajectory
                 ax.plot(true_states_cpu[:, 0], true_states_cpu[:, 1], '-', linewidth=1.0, color='red', label='True Trajectory')
                 # Plot predicted trajectory
                 ax.plot(pred_cpu[:, 0], pred_cpu[:, 1], '-', linewidth=1.0, color=pf_color, label='PF Trajectory')
                 # Mark start points
                 ax.plot(true_states_cpu[0, 0], true_states_cpu[0, 1], 'o', color='red', markersize=5, label='_nolegend_')
                 ax.plot(pred_cpu[0, 0], pred_cpu[0, 1], 'o', color=pf_color, markersize=5, label='_nolegend_')
                 ax.set_title(f'Full Trajectory (T={T})')
                 ax.legend(fontsize='small', loc='upper right')


            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal')

        plt.tight_layout(pad=0.5) # Adjust subplot spacing
        # plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.95, wspace=0.05, hspace=0.15)
        show_pause(pause=0.01)

