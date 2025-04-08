import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.data_utils import wrap_angle, compute_staticstics, split_data, make_batch_iterator, make_repeating_batch_iterator
from utils.method_utils import atan2, compute_sq_distance
from utils.plotting_utils import plot_maze, show_pause


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

        # Build the learnable modules.
        self.build_modules()

    def build_modules(self):
        # --- Encoder: a convolutional network ---
        # We assume input images are 24x24 with 3 channels.
        # Note: In PyTorch image tensors are [B, C, H, W].
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 6 -> 3
            nn.ReLU()
        )
        self.encoder_flatten = nn.Flatten()
        self.encoder_dropout = nn.Dropout(p=1 - self.dropout_keep_prob)
        # 64 filters * 3x3 output = 576 features.
        self.encoder_linear = nn.Linear(576, 128)

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
        """
        # Ensure correct shape: [B, 3, 24, 24]
        if o.dim() == 4 and o.size(-1) == 3:
            o = o.permute(0, 3, 1, 2)
        x = self.encoder_conv(o)
        x = self.encoder_flatten(x)
        x = self.encoder_dropout(x)
        x = self.encoder_linear(x)
        return F.relu(x)

    def measurement_update(self, encoding, particles, means, stds):
        """
        Computes the observation likelihood for each particle.
        encoding: [B, 128]
        particles: [B, N, 3]
        """
        particle_input = self.transform_particles_as_input(particles, means, stds)  # [B, N, 4]
        # Tile encoding to match number of particles.
        encoding_input = encoding.unsqueeze(1).expand(-1, particles.size(1), -1)
        inp = torch.cat([encoding_input, particle_input], dim=-1)  # [B, N, 132]
        B, N, _ = inp.shape
        inp_flat = inp.view(B * N, -1)
        out = self.obs_like_estimator(inp_flat).view(B, N)
        # Scale the output between min_obs_likelihood and 1.
        return out * (1 - self.min_obs_likelihood) + self.min_obs_likelihood

    def transform_particles_as_input(self, particles, means, stds):
        """
        Normalizes particle poses and augments with cosine and sine of orientation.
        particles: [B, N, 3]
        means['s'] and stds['s'] are assumed to be tensors or convertible arrays with shape [1, 1, 3].
        """
        # means['s'] has shape [1, 1, 3]
        # stds['s'] has shape [1, 1, 3]

        # Slice means and stds to get the x, y components (shape [1, 1, 2])
        means_xy = means['s'][:, :, :2]
        stds_xy = stds['s'][:, :, :2]

        # Slice particles to get x, y components (shape [B, N, 2])
        particles_xy = particles[:, :, :2]

        # Normalize position. Broadcasting works: [B, N, 2] op [1, 1, 2] -> [B, N, 2]
        norm_pos = (particles_xy - means_xy) / stds_xy

        # Get orientation components
        cos_theta = torch.cos(particles[:, :, 2:3])
        sin_theta = torch.sin(particles[:, :, 2:3])

        # Concatenate: [B, N, 2] + [B, N, 1] + [B, N, 1] -> [B, N, 4]
        return torch.cat([norm_pos, cos_theta, sin_theta], dim=-1)


    def propose_particles(self, encoding, num_particles, state_mins, state_maxs):
        """
        Proposes new particles from an image encoding using a learned proposer network.

        Args:
            encoding (torch.Tensor): Image encodings, shape [B, encoding_dim].
            num_particles (int): Number of particles to propose per batch item.
            state_mins (list/np.array/torch.Tensor): Minimum values for state dimensions [x, y, theta].
            state_maxs (list/np.array/torch.Tensor): Maximum values for state dimensions [x, y, theta].

        Returns:
            torch.Tensor: Proposed particle states, shape [B, num_particles, 3].
        """
        B = encoding.size(0)
        device = encoding.device
        dtype = encoding.dtype

        # --- Handle empty batch or zero particles ---
        if B == 0 or num_particles == 0:
            # Return an empty tensor with the correct final dimension (state_dim = 3)
            return torch.empty((B, num_particles, 3), dtype=dtype, device=device) # Assuming state_dim is 3

        # --- Ensure particle_proposer exists ---
        if not hasattr(self, 'particle_proposer') or self.particle_proposer is None:
            raise RuntimeError("Particle proposer module (self.particle_proposer) is not defined or initialized.")

        # --- Duplicate encoding for each particle ---
        # tf.tile(encoding[:, tf.newaxis, :], [1, num_particles, 1]) -> torch equivalent
        # Add dimension: [B, encoding_dim] -> [B, 1, encoding_dim]
        # Expand: [B, 1, encoding_dim] -> [B, num_particles, encoding_dim]
        encoding_dup = encoding.unsqueeze(1).expand(-1, num_particles, -1)

        # --- Apply proposer network (equivalent to snt.BatchApply) ---
        # Reshape for MLP: [B, num_particles, encoding_dim] -> [B * num_particles, encoding_dim]
        inp_flat = encoding_dup.reshape(B * num_particles, -1)
        # Apply the proposer network
        proposed_raw = self.particle_proposer(inp_flat) # Expected output shape: [B * num_particles, 4]
        # Reshape back: [B * num_particles, 4] -> [B, num_particles, 4]
        proposed_raw = proposed_raw.view(B, num_particles, 4)

        # --- Transform the outputs to valid state values ---
        # Convert state bounds to tensors on the correct device and dtype
        if not torch.is_tensor(state_mins):
            state_mins = torch.tensor(state_mins, dtype=dtype, device=device)
        if not torch.is_tensor(state_maxs):
            state_maxs = torch.tensor(state_maxs, dtype=dtype, device=device)

        # Ensure state bounds have at least 2 elements for x and y
        if len(state_mins) < 2 or len(state_maxs) < 2:
            raise ValueError(f"state_mins/state_maxs must have at least 2 elements for x, y. Got lengths {len(state_mins)}, {len(state_maxs)}")

        # Scale and shift the first output (index 0) for x-coordinate
        range_x = state_maxs[0] - state_mins[0]
        mid_x = (state_maxs[0] + state_mins[0]) / 2.0
        part0 = proposed_raw[:, :, 0:1] * (range_x / 2.0) + mid_x

        # Scale and shift the second output (index 1) for y-coordinate
        range_y = state_maxs[1] - state_mins[1]
        mid_y = (state_maxs[1] + state_mins[1]) / 2.0
        part1 = proposed_raw[:, :, 1:2] * (range_y / 2.0) + mid_y

        # Compute angle (theta) using atan2 from the third and fourth outputs (indices 2, 3)
        # The provided method_utils.atan2(x, y) calls torch.atan2(y, x).
        # So, the first argument is x, the second is y.
        part2 = atan2(proposed_raw[:, :, 2:3], proposed_raw[:, :, 3:4])

        # --- Concatenate results ---
        # tf.concat([...], axis=2) -> torch.cat([...], dim=-1)
        proposed_particles = torch.cat([part0, part1, part2], dim=-1) # Shape: [B, num_particles, 3]

        return proposed_particles


    def motion_update(self, actions, particles, means, stds, state_step_sizes, stop_sampling_gradient=False):
        """
        Moves particles according to actions with added learned noise.
        actions: [B, 3]
        particles: [B, N, 3]
        state_step_sizes: list or array of length 3.
        """
        # Expand actions to match particles.
        # print(actions.shape)
        actions = torch.tensor(actions)
        actions_exp = actions.unsqueeze(1)  # [B, 1, 3]
        std_a = stds['a']
        if not torch.is_tensor(std_a):
            std_a = torch.tensor(std_a, dtype=actions.dtype, device=actions.device)
        action_input = actions_exp / std_a  # broadcasting over particle dim.
        B, N, _ = particles.shape
        action_input = action_input.expand(-1, N, -1)
        # Add random noise.
        random_input = torch.randn_like(action_input)
        inp = torch.cat([action_input, random_input], dim=-1)  # [B, N, 6]
        inp_flat = inp.view(B * N, -1)
        delta = self.mo_noise_generator(inp_flat).view(B, N, -1)
        if stop_sampling_gradient:
            delta = delta.detach()
        delta = delta - delta.mean(dim=1, keepdim=True)
        noisy_actions = actions_exp + delta  # [B, N, 3]

        if self.learn_odom:
            # Use learned odometry model.
            state_input = self.transform_particles_as_input(particles, means, stds)  # [B, N, 4]
            action_input_odom = noisy_actions / std_a
            inp2 = torch.cat([state_input, action_input_odom], dim=-1)  # [B, N, 7]
            inp2_flat = inp2.view(B * N, -1)
            state_delta = self.mo_transition_model(inp2_flat).view(B, N, -1)
            state_step_sizes = torch.tensor(state_step_sizes, dtype=state_delta.dtype, device=state_delta.device)
            state_delta = state_delta * state_step_sizes  # elementwise scaling
            new_states = particles + state_delta
            new_states[:, :, 2:3] = wrap_angle(new_states[:, :, 2:3])
            moved_particles = new_states
        else:
            theta = particles[:, :, 2:3]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            new_x = particles[:, :, 0:1] + (noisy_actions[:, :, 0:1] * cos_theta + noisy_actions[:, :, 1:2] * sin_theta)
            new_y = particles[:, :, 1:2] + (noisy_actions[:, :, 0:1] * sin_theta - noisy_actions[:, :, 1:2] * cos_theta)
            new_theta = wrap_angle(particles[:, :, 2:3] + noisy_actions[:, :, 2:3])
            moved_particles = torch.cat([new_x, new_y, new_theta], dim=-1)
        return moved_particles

    def particles_to_state(self, particle_list, particle_probs_list):
        """
        Computes the weighted mean (position and orientation) of particles.
        particle_list: [B, T, N, 3]
        particle_probs_list: [B, T, N]
        """
        # Weighted mean for x,y.
        mean_position = (particle_probs_list.unsqueeze(-1) * particle_list[..., :2]).sum(dim=2)
        # For orientation, use weighted average on sin and cos.
        cos_sum = (particle_probs_list.unsqueeze(-1) * torch.cos(particle_list[..., 2:3])).sum(dim=2)
        sin_sum = (particle_probs_list.unsqueeze(-1) * torch.sin(particle_list[..., 2:3])).sum(dim=2)
        mean_orientation = atan2(cos_sum, sin_sum)
        return torch.cat([mean_position, mean_orientation], dim=-1)

    def connect_modules(self, batch, means, stds, state_mins, state_maxs, state_step_sizes):
        """
        Builds the full particle filtering process.
        batch: dict with keys 'o', 'a', 's'
          - o: [B, T, 24, 24, 3] (or [B, T, 3, 24, 24])
          - a: [B, T, 3]
          - s: [B, T, 3]
        """
        print(batch['o'].shape)
        B, T = batch['o'].shape[0], batch['o'].shape[1]
        # Flatten time dimension to process images through encoder.
        # o = batch['o'].view(B * T, batch['o'].shape[-3], batch['o'].shape[-2], batch['o'].shape[-1])
        o = torch.tensor(batch['o'], dtype=torch.float32, device=batch['s'].device).view(
            B * T, batch['o'].shape[-3], batch['o'].shape[-2], batch['o'].shape[-1]
        )
        if o.size(1) != 3:
            o = o.permute(0, 3, 1, 2)
        encodings = self.forward_encoder(o).view(B, T, -1)  # [B, T, 128]

        # Initialize particles.
        if self.init_with_true_state:
            initial_particles = batch['s'][:, 0].unsqueeze(1).expand(B, self.num_particles, -1)
        else:
            if self.use_proposer:
                initial_particles = self.propose_particles(encodings[:, 0], self.num_particles, state_mins, state_maxs)
            else:
                state_mins_t = torch.tensor(state_mins, dtype=batch['s'].dtype, device=batch['s'].device)
                state_maxs_t = torch.tensor(state_maxs, dtype=batch['s'].dtype, device=batch['s'].device)
                parts = []
                for d in range(self.state_dim):
                    parts.append(torch.rand(B, self.num_particles, 1, device=batch['s'].device) *
                                 (state_maxs_t[d] - state_mins_t[d]) + state_mins_t[d])
                initial_particles = torch.cat(parts, dim=-1)

        initial_particle_probs = torch.ones(B, self.num_particles, device=batch['s'].device) / self.num_particles

        # Lists to store history.
        particle_list = initial_particles.unsqueeze(1)  # [B, 1, N, 3]
        particle_probs_list = initial_particle_probs.unsqueeze(1)  # [B, 1, N]

        particles = initial_particles
        particle_probs = initial_particle_probs

        # Process each time step.
        for i in range(1, T):
            # Determine numbers of proposed/resampled particles.
            # print("i: ", i)
            # print("self.propose_ratio: ", self.propose_ratio)
            # print("self.num_particles: ", self.num_particles)
            # print( round((self.propose_ratio ** float(i)) * float(self.num_particles)))
            num_proposed_float = round((self.propose_ratio ** float(i)) * float(self.num_particles))
            num_proposed = int(num_proposed_float)
            num_resampled = self.num_particles - num_proposed

            if self.propose_ratio < 1.0:
                # Systematic resampling.
                markers = torch.linspace(0, 1 - 1/num_resampled, steps=num_resampled,
                                           device=particle_probs.device).unsqueeze(0).expand(B, -1)
                random_offset = torch.rand(B, 1, device=particle_probs.device) * (1/num_resampled)
                markers = markers + random_offset
                cum_probs = torch.cumsum(particle_probs, dim=1)
                # For each marker, find the first index where cumulative probability exceeds it.
                samples = []
                for b in range(B):
                    indices = torch.searchsorted(cum_probs[b], markers[b])
                    samples.append(indices)
                samples = torch.stack(samples, dim=0)  # [B, num_resampled]
                # Gather resampled particles.
                standard_particles = torch.gather(particles, 1,
                                                  samples.unsqueeze(-1).expand(-1, -1, self.state_dim))
                standard_particle_probs = torch.ones(B, num_resampled, device=particle_probs.device)
                # (Detaching as in TF code.)
                standard_particles = standard_particles.detach()
                standard_particle_probs = standard_particle_probs.detach()
                # Motion update.
                standard_particles = self.motion_update(batch['a'][:, i], standard_particles, means, stds, state_step_sizes)
                meas = self.measurement_update(encodings[:, i], standard_particles, means, stds)
                standard_particle_probs = standard_particle_probs * meas

            if self.propose_ratio > 0.0:
                proposed_particles = self.propose_particles(encodings[:, i], num_proposed, state_mins, state_maxs)
                proposed_particle_probs = torch.ones(B, num_proposed, device=particle_probs.device)

            if self.propose_ratio == 1.0:
                particles = proposed_particles
                particle_probs = proposed_particle_probs
            elif self.propose_ratio == 0.0:
                particles = standard_particles
                particle_probs = standard_particle_probs
            else:
                standard_particle_probs = standard_particle_probs * (num_resampled / self.num_particles) / \
                    (standard_particle_probs.sum(dim=1, keepdim=True) + 1e-16)
                proposed_particle_probs = proposed_particle_probs * (num_proposed / self.num_particles) / \
                    (proposed_particle_probs.sum(dim=1, keepdim=True) + 1e-16)
                particles = torch.cat([standard_particles, proposed_particles], dim=1)
                particle_probs = torch.cat([standard_particle_probs, proposed_particle_probs], dim=1)

            particle_probs = particle_probs / (particle_probs.sum(dim=1, keepdim=True) + 1e-16)
            particle_list = torch.cat([particle_list, particles.unsqueeze(1)], dim=1)
            particle_probs_list = torch.cat([particle_probs_list, particle_probs.unsqueeze(1)], dim=1)

        pred_states = self.particles_to_state(particle_list, particle_probs_list)
        # Store outputs for later use.
        self.pred_states = pred_states
        self.particle_list = particle_list
        self.particle_probs_list = particle_probs_list
        return particles, particle_probs, encodings, particle_list, particle_probs_list

    # === Training and prediction routines ===
    # (Below we re–implement training loops without sessions or savers.
    #  Each training “stage” is defined by a loss function and its optimizer.)

    def compile_training_stages(self, means, stds, state_step_sizes, state_mins,
                                  state_maxs, learning_rate, plot_task):
        """
        Creates a dictionary of training stages.
        In PyTorch one usually writes one training loop; here we define several loss functions
        and corresponding optimizers so that you can, for example, alternate training the odometry model,
        measurement model, particle proposer, and the end-to–end filter.
        (Many details depend on how your batch dictionaries and statistics are defined.)
        """
        train_stages = {}

        # Stage: Train Odometry Model.
        if self.learn_odom:
            def train_odom_fn(batch):
                # Use time index 1 for action and the first state (duplicated) for particles.
                motion_samples = self.motion_update(batch['a'][:, 1],
                                                    batch['s'][:, :1].expand(-1, self.num_particles, -1),
                                                    means, stds, state_step_sizes,
                                                    stop_sampling_gradient=True)
                sq_distance = compute_sq_distance(motion_samples, batch['s'][:, 1:2], state_step_sizes)
                return sq_distance.mean()
            optimizer_odom = torch.optim.Adam(self.mo_transition_model.parameters(), lr=learning_rate)
            train_stages['train_odom'] = {'loss_fn': train_odom_fn, 'optimizer': optimizer_odom}

        # Stage: Train Motion Noise Generator.
        def train_motion_sampling_fn(batch):
            motion_samples = self.motion_update(batch['a'][:, 1],
                                                batch['s'][:, :1].expand(-1, self.num_particles, -1),
                                                means, stds, state_step_sizes)
            std_val = 0.01
            sq_distance = compute_sq_distance(motion_samples, batch['s'][:, 1:2], state_step_sizes)
            activations = (1 / self.num_particles) / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2)) * \
                          torch.exp(-sq_distance / (2.0 * std_val ** 2))
            loss = (-torch.log(1e-16 + torch.sum(activations, dim=-1))).mean()
            return loss
        optimizer_motion = torch.optim.Adam(self.mo_noise_generator.parameters(), lr=learning_rate)
        train_stages['train_motion_sampling'] = {'loss_fn': train_motion_sampling_fn, 'optimizer': optimizer_motion}

        # Stage: Train Measurement Model.
        def train_measurement_model_fn(batch):
            B = batch['s'].size(0)
            # Create a test set by pairing the first state of each sequence with every other.
            test_particles = batch['s'][:, 0].unsqueeze(1).expand(B, B, -1)  # [B, B, 3]
            encoding = self.forward_encoder(batch['o'][:, 0])
            measurement_out = self.measurement_update(encoding, test_particles, means, stds)
            correct_samples = torch.diag(measurement_out)
            # Off–diagonals.
            incorrect_samples = measurement_out - torch.diag(torch.diag(measurement_out))
            loss = ( -torch.log(correct_samples).sum()/B -
                     torch.log(1.0 - incorrect_samples).sum()/(B * (B - 1)) )
            return loss
        params_measurement = list(self.encoder_conv.parameters()) + list(self.encoder_linear.parameters()) + \
                             list(self.obs_like_estimator.parameters())
        optimizer_measurement = torch.optim.Adam(params_measurement, lr=learning_rate)
        train_stages['train_measurement_model'] = {'loss_fn': train_measurement_model_fn, 'optimizer': optimizer_measurement}

        # Stage: Train Particle Proposer.
        if self.use_proposer:
            def train_particle_proposer_fn(batch):
                # Stop gradients into the encoder.
                encoding = self.forward_encoder(batch['o'][:, 0]).detach()
                proposed_particles = self.propose_particles(encoding, self.num_particles, state_mins, state_maxs)
                std_val = 0.2
                sq_distance = compute_sq_distance(proposed_particles,
                                                  batch['s'][:, :1].expand(-1, self.num_particles, -1),
                                                  state_step_sizes)
                activations = (1 / self.num_particles) / torch.sqrt(2 * np.pi * std_val ** 2) * \
                              torch.exp(-sq_distance / (2.0 * std_val ** 2))
                loss = (-torch.log(1e-16 + torch.sum(activations, dim=-1))).mean()
                return loss
            optimizer_proposer = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)
            train_stages['train_particle_proposer'] = {'loss_fn': train_particle_proposer_fn, 'optimizer': optimizer_proposer}

        # Stage: End-to-End Training.
        def train_e2e_fn(batch):
            # Run the full filter.
            particles, particle_probs, encodings, particle_list, particle_probs_list = \
                self.connect_modules(batch, means, stds, state_mins, state_maxs, state_step_sizes)
            std_val = self.particle_std  # assumed to be set externally
            loss1 = compute_sq_distance(particle_list, batch['s'].unsqueeze(2), state_step_sizes)
            activations = particle_probs / torch.sqrt(2 * np.pi * std_val ** 2) * \
                          torch.exp(-loss1 / (2.0 * self.particle_std ** 2))
            loss = (-torch.log(1e-16 + torch.sum(activations, dim=2))).mean()
            # Additionally monitor the MSE on the last time step.
            pred = self.particles_to_state(particle_list, particle_probs_list)
            loss2 = compute_sq_distance(pred[:, -1], batch['s'][:, -1], state_step_sizes).mean()
            return loss, loss2
        optimizer_e2e = torch.optim.Adam(self.parameters(), lr=learning_rate)
        train_stages['train_e2e'] = {'loss_fn': train_e2e_fn, 'optimizer': optimizer_e2e}

        return train_stages

    def predict(self, batch, num_particles, return_particles=False):
        """
        Runs the particle filter on a batch.
        """
        # (If init_with_true_state is True, you might want to pass the first state.)
        particles, particle_probs, encodings, particle_list, particle_probs_list = \
            self.connect_modules(batch, self.means, self.stds, self.state_mins, self.state_maxs, self.state_step_sizes)
        pred_states = self.particles_to_state(particle_list, particle_probs_list)
        if return_particles:
            return pred_states, particle_list, particle_probs_list
        else:
            return pred_states

    def load_model(self, model_path, model_file='best_validation.pth',
                   statistics_file='statistics.npz', connect_and_initialize=True, modules=None):
        """
        Loads saved parameters and, if requested, statistics.
        """
        if modules is None:
            modules = ('encoder', 'mo_noise_generator', 'mo_transition_model',
                       'obs_like_estimator', 'particle_proposer')
        if connect_and_initialize:
            stats = np.load(os.path.join(model_path, statistics_file), allow_pickle=True)
            stats = dict(stats)
            # (If needed, use stats to set self.means, self.stds, etc.)
        state_dict = torch.load(os.path.join(model_path, model_file))
        self.load_state_dict(state_dict)
        print("Loaded model from", os.path.join(model_path, model_file))

    def fit(self, data, model_path, train_individually, train_e2e, split_ratio,
            seq_len, batch_size, epoch_length, num_epochs, patience,
            learning_rate, dropout_keep_ratio, num_particles, particle_std,
            plot_task=None, plot=False):
        """
        Full training loop.
        (Here we assume that data utilities return dictionaries with keys 'train' and 'val'.)
        """
        self.particle_std = particle_std
        self.num_particles = num_particles

        # Preprocess data.
        data_split = split_data(data, ratio=split_ratio)
        epoch_lengths = {'train': epoch_length, 'val': epoch_length * 2}
        batch_iterators = {
            'train': make_batch_iterator(data_split['train'], seq_len=seq_len, batch_size=batch_size),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'],
                                                 batch_size=batch_size, seq_len=seq_len),
            'train_ex': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=seq_len),
            'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len),
            'train1': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=2),
            'val1': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'],
                                                  batch_size=batch_size, seq_len=2),
        }

        means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data_split['train'])
        self.means, self.stds = means, stds
        self.state_step_sizes, self.state_mins, self.state_maxs = state_step_sizes, state_mins, state_maxs

        # Build the full model graph (i.e. connect modules) on one batch.
        print("batch iterator type: ", type(batch_iterators['train']))
        _ = self.connect_modules(next(batch_iterators['train']), means, stds, state_mins, state_maxs, state_step_sizes)

        # Compile training stages.
        train_stages = self.compile_training_stages(means, stds, state_step_sizes, state_mins,
                                                      state_maxs, learning_rate, plot_task)

        # Save statistics for later.
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        np.savez(os.path.join(model_path, 'statistics'),
                 means=means, stds=stds, state_step_sizes=state_step_sizes,
                 state_mins=state_mins, state_maxs=state_maxs)

        # Define curriculum.
        curriculum = []
        if train_individually:
            if self.learn_odom:
                curriculum.append('train_odom')
            curriculum.append('train_motion_sampling')
            curriculum.append('train_measurement_model')
            if self.use_proposer:
                curriculum.append('train_particle_proposer')
        if train_e2e:
            curriculum.append('train_e2e')

        log = {c: {'train': {}, 'val': {}} for c in curriculum}

        for stage in curriculum:
            best_val_loss = float('inf')
            best_epoch = 0
            epoch = 0
            while epoch < num_epochs and (epoch - best_epoch) < patience:
                for phase in ['train', 'val']:
                    losses = []
                    # Retrieve an iterator for the current phase.
                    # (Here we assume the batch iterator dictionary contains keys matching stage names.)
                    iterator = batch_iterators.get(train_stages[stage].get('batch_iterator_names', {}).get(phase, phase))
                    # Iterate for the number of steps defined by epoch_lengths.
                    for _ in range(epoch_lengths[phase]):
                        batch = next(iterator)
                        self.train() if phase == 'train' else self.eval()
                        optimizer = train_stages[stage]['optimizer']
                        optimizer.zero_grad()
                        loss_val = train_stages[stage]['loss_fn'](batch)
                        if phase == 'train':
                            loss_val.backward()
                            optimizer.step()
                        losses.append(loss_val.item())
                    avg_loss = np.mean(losses)
                    log[stage][phase] = avg_loss
                current_val_loss = log[stage]['val']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    torch.save(self.state_dict(), os.path.join(model_path, 'best_validation.pth'))
                    txt = f"epoch {epoch:3d} >> "
                else:
                    txt = f"epoch {epoch:3d} == "
                txt += f"loss: train {log[stage]['train']:.2f}, val {log[stage]['val']:.2f}"
                print(txt)
                if plot and 'plot' in train_stages[stage]:
                    train_stages[stage]['plot'](epoch)
                epoch += 1
            # Restore the best model.
            self.load_model(model_path, model_file='best_validation.pth', connect_and_initialize=False)
        return log

    # === Plotting functions (adapted to PyTorch) ===

    def plot_motion_model(self, batch, motion_samples, task):
        s_motion_samples = motion_samples.detach().cpu().numpy()
        plt.figure('Motion Model')
        plt.clf()
        plot_maze(task)
        for i in range(min(len(s_motion_samples), 10)):
            plt.quiver(s_motion_samples[i, :, 0], s_motion_samples[i, :, 1],
                       np.cos(s_motion_samples[i, :, 2]), np.sin(s_motion_samples[i, :, 2]),
                       color='blue', width=0.001, scale=100)
            batch_s = batch['s'][i].cpu().numpy()
            plt.quiver(batch_s[0, 0], batch_s[0, 1], np.cos(batch_s[0, 2]), np.sin(batch_s[0, 2]),
                       color='black', scale=50, width=0.003)
            plt.quiver(batch_s[1, 0], batch_s[1, 1], np.cos(batch_s[1, 2]), np.sin(batch_s[1, 2]),
                       color='red', scale=50, width=0.003)
        plt.gca().set_aspect('equal')
        plt.pause(0.01)

    def plot_measurement_model(self, batch, measurement_model_out):
        s_measurement_model_out = measurement_model_out.detach().cpu().numpy()
        plt.figure('Measurement Model Output')
        plt.clf()
        plt.imshow(s_measurement_model_out, interpolation="nearest", cmap="coolwarm")
        plt.pause(0.01)

    def plot_particle_proposer(self, batch, proposed_particles, task):
        s_samples = proposed_particles.detach().cpu().numpy()
        plt.figure('Particle Proposer')
        plt.clf()
        plot_maze(task)
        for i in range(min(len(s_samples), 10)):
            color = np.random.uniform(0.0, 1.0, 3)
            plt.quiver(s_samples[i, :, 0], s_samples[i, :, 1],
                       np.cos(s_samples[i, :, 2]), np.sin(s_samples[i, :, 2]),
                       color=color, width=0.001, scale=100)
            batch_s = batch['s'][i].cpu().numpy()
            plt.quiver(batch_s[0, 0], batch_s[0, 1], np.cos(batch_s[0, 2]), np.sin(batch_s[0, 2]),
                       color=color, scale=50, width=0.003)
        plt.pause(0.01)

    def plot_particle_filter(self, batch, num_particles, state_step_sizes, task):
        # Here we fix num_particles to 1000 for visualization.
        num_particles = 1000
        head_scale = 1.5
        quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 40., 'width': 0.003,
                       'headlength': 5 * head_scale, 'headwidth': 3 * head_scale,
                       'headaxislength': 4.5 * head_scale}
        marker_kwargs = {'markersize': 4.5, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}
        color_list = plt.cm.tab10(np.linspace(0, 1, 10))
        colors = {'pf_ind_e2e': color_list[1]}
        pred, s_particle_list, s_particle_probs_list = self.predict(batch, num_particles, return_particles=True)
        num_steps = 20  # (Assumed number of steps.)
        for s in range(1):
            plt.figure(f"example {s}", figsize=[12, 5.15])
            plt.clf()
            for i in range(num_steps):
                ax = plt.subplot(4, 5, i + 1)
                plt.cla()
                plot_maze(task, margin=5, linewidth=0.5)
                if i < num_steps - 1:
                    s_particle = s_particle_list[s, i].detach().cpu().numpy()
                    s_probs = s_particle_probs_list[s, i].detach().cpu().numpy()
                    ax.quiver(s_particle[:, 0], s_particle[:, 1],
                              np.cos(s_particle[:, 2]), np.sin(s_particle[:, 2]),
                              s_probs, cmap='viridis_r', clim=[0, 2.0 / num_particles],
                              alpha=1.0, **quiv_kwargs)
                    current_state = batch['s'][s, i].cpu().numpy()
                    ax.quiver(current_state[0], current_state[1],
                              np.cos(current_state[2]), np.sin(current_state[2]),
                              color="red", **quiv_kwargs)
                    ax.plot(current_state[0], current_state[1], 'or', **marker_kwargs)
                else:
                    batch_s = batch['s'][s, :num_steps].cpu().numpy()
                    pred_s = pred[s, :num_steps].detach().cpu().numpy()
                    ax.plot(batch_s[:, 0], batch_s[:, 1], '-', linewidth=0.6, color='red')
                    ax.plot(pred_s[:, 0], pred_s[:, 1], '-', linewidth=0.6, color=colors['pf_ind_e2e'])
                    ax.plot(batch_s[0, 0], batch_s[0, 1], '.', color='red', markersize=3)
                    ax.plot(pred_s[0, 0], pred_s[0, 1], '.', color=colors['pf_ind_e2e'], markersize=3)
                plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0,
                                    wspace=0.001, hspace=0.1)
                plt.gca().set_aspect('equal')
                plt.xticks([])
                plt.yticks([])
        show_pause(pause=0.01)
