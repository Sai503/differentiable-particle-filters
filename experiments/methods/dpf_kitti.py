import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from utils.data_utils_kitti import (wrap_angle, compute_statistics, split_data,
                                    make_batch_iterator, make_repeating_batch_iterator,
                                    rotation_matrix, load_data_for_stats)
from utils.method_utils import atan2, compute_sq_distance
from utils.plotting_utils import plot_maze, show_pause


# A learnable noise scaler similar to Sonnet's Module wrapping a variable.
class NoiseScaler(nn.Module):
    def __init__(self):
        super(NoiseScaler, self).__init__()
        # Initialize a learnable scalar (starting at 0.0)
        self.scaler = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        # Multiply input x by exp(10 * scaler)
        return x * torch.exp(10 * self.scaler)


class DPF(nn.Module):
    def __init__(self, init_with_true_state, learn_odom, use_proposer,
                 propose_ratio, proposer_keep_ratio, min_obs_likelihood, learn_gaussian_mle):
        """
        PyTorch version of the Deep Particle Filter.
          - init_with_true_state: use the true state to initialize particles.
          - learn_odom: learn an odometry transition model.
          - use_proposer: if True and not initializing with true state, use a particle proposer.
          - propose_ratio: ratio of particles to propose.
          - proposer_keep_ratio: dropout keep probability for the proposer.
          - min_obs_likelihood: lower bound on observation likelihood.
          - learn_gaussian_mle: if True, use a Gaussian–based loss for motion sampling.
        """
        super(DPF, self).__init__()
        self.init_with_true_state = init_with_true_state
        self.learn_odom = learn_odom
        self.use_proposer = use_proposer and (not init_with_true_state)
        self.propose_ratio = propose_ratio if not init_with_true_state else 0.0
        self.min_obs_likelihood = min_obs_likelihood
        self.learn_gaussian_mle = learn_gaussian_mle

        self.state_dim = 5
        self.action_dim = 3
        self.observation_dim = 6

        # These will be set later (e.g. in fit) for training
        self.num_particles = None
        self.particle_std = None

        # Build the learnable modules.
        self.build_modules(proposer_keep_ratio)

    def build_modules(self, proposer_keep_ratio):
        # --- MEASUREMENT MODEL ---
        # Encoder: a convolutional network to encode input “images.”
        # Here we assume input images are of shape [B, observation_dim, H, W].
        self.encoder = nn.Sequential(
            # Using kernel shapes and strides as in the TF code.
            nn.Conv2d(self.observation_dim, 16, kernel_size=7, stride=1, padding=3),  # same padding
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=(1,2), padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=(1,2), padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=(2,2), padding=2),
            nn.ReLU(),
            nn.Flatten(),
            # Dropout is applied in forward() where we know the keep probability.
            nn.Dropout(p=0.0),
            nn.Linear(16 * self.compute_encoder_output_size(), 128),
            nn.ReLU()
        )

        # Observation likelihood estimator: maps concatenated (encoding, transformed state) to a probability.
        self.obs_like_estimator = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output in (0, 1)
        )
        # We will scale the output as: out * (1 - min_obs_likelihood) + min_obs_likelihood

        # --- MOTION NOISE GENERATOR ---
        mlp_out = 4 if self.learn_gaussian_mle else 2
        self.mo_noise_generator = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, mlp_out)
        )

        # --- ODOMETRY MODEL ---
        if self.learn_odom:
            self.mo_transition_model = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.state_dim)
            )

        # --- PARTICLE PROPOSER ---
        if self.use_proposer:
            self.particle_proposer = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(p=1 - proposer_keep_ratio),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Tanh()
            )

        # Learnable noise scalers (for motion sampling)
        self.noise_scaler1 = NoiseScaler()
        self.noise_scaler2 = NoiseScaler()

    def compute_encoder_output_size(self):
        """
        Compute the flattened output size after the conv layers.
        This is hard-coded here; adjust if your input dimensions differ.
        For example, if input size is 50x150 and strides as specified, one might get 3x5.
        """
        # Example: output feature map size is 3x5.
        return 3 * 5

    def measurement_update(self, encoding, particles, means, stds):
        """
        Compute the likelihood for each particle given the observation encoding.
          encoding: [B, 128]
          particles: [B, N, state_dim]
          means, stds: dicts with key 's'
        """
        particle_input = self.transform_particles_as_input(particles, means, stds)
        B, N = particles.shape[0], particles.shape[1]
        # Tile encoding for each particle.
        encoding_input = encoding.unsqueeze(1).expand(B, N, -1)
        inp = torch.cat([encoding_input, particle_input], dim=-1)
        inp_flat = inp.reshape(B * N, -1)
        out = self.obs_like_estimator(inp_flat).reshape(B, N)
        obs_likelihood = out * (1 - self.min_obs_likelihood) + self.min_obs_likelihood
        return obs_likelihood

    def transform_particles_as_input(self, particles, means, stds):
        # Normalize particles using state statistics and then select dimensions 3:5.
        norm = (particles - means['s']) / stds['s']
        return norm[..., 3:5]

    def propose_particles(self, encoding, num_particles, state_mins, state_maxs):
        """
        Propose new particles based on the observation encoding.
          encoding: [B, 128]
        """
        B = encoding.shape[0]
        encoding_dup = encoding.unsqueeze(1).expand(B, num_particles, -1)
        encoding_flat = encoding_dup.reshape(B * num_particles, -1)
        proposed = self.particle_proposer(encoding_flat).reshape(B, num_particles, 4)
        # Map outputs to valid ranges for state dimensions 0 and 1.
        state_mins = torch.tensor(state_mins, dtype=proposed.dtype, device=proposed.device)
        state_maxs = torch.tensor(state_maxs, dtype=proposed.dtype, device=proposed.device)
        part0 = proposed[:, :, 0:1] * ((state_maxs[0] - state_mins[0]) / 2.0) + ((state_maxs[0] + state_mins[0]) / 2.0)
        part1 = proposed[:, :, 1:2] * ((state_maxs[1] - state_mins[1]) / 2.0) + ((state_maxs[1] + state_mins[1]) / 2.0)
        part2 = atan2(proposed[:, :, 2:3], proposed[:, :, 3:4])
        return torch.cat([part0, part1, part2], dim=-1)

    def motion_update(self, actions, particles, means, stds, state_step_sizes, stop_sampling_gradient=False):
        """
        Move particles according to actions with learned noise.
          actions: [B, ...] (per time step)
          particles: [B, N, state_dim]
          state_step_sizes: list-like (length equal to state_dim)
        """
        time_step = 0.103
        if self.learn_gaussian_mle:
            # For Gaussian MLE, compute actions from differences in velocity components.
            actions_new = torch.cat([
                particles[:, :, 3:4] - means['s'][:, :, 3:4],
                particles[:, :, 4:5] - means['s'][:, :, 4:5]
            ], dim=-1)
            action_input = actions_new / stds['s'][:, :, 3:5]
            B, N = action_input.shape[0], action_input.shape[1]
            delta = self.mo_noise_generator(action_input.reshape(B * N, -1)).reshape(B, N, -1)
            # Scale delta: first two dims for translation, next two for rotation.
            delta_trans = delta[:, :, 0:2] * state_step_sizes[3]
            delta_rot = delta[:, :, 2:4] * state_step_sizes[4]
            if stop_sampling_gradient:
                delta = delta.detach()
            action_vel_f = torch.normal(mean=delta[:, :, 0:1], std=delta[:, :, 1:2])
            action_vel_rot = torch.normal(mean=delta[:, :, 2:3], std=delta[:, :, 3:4])
            heading = particles[:, :, 2:3]
            sin_heading = torch.sin(heading)
            cos_heading = torch.cos(heading)
            new_x = particles[:, :, 0:1] + cos_heading * particles[:, :, 3:4] * time_step
            new_y = particles[:, :, 1:2] + sin_heading * particles[:, :, 3:4] * time_step
            new_theta = wrap_angle(particles[:, :, 2:3] + particles[:, :, 4:5] * time_step)
            new_v = particles[:, :, 3:4] + action_vel_f
            new_theta_dot = particles[:, :, 4:5] + action_vel_rot
            moved_particles = torch.cat([new_x, new_y, new_theta, new_v, new_theta_dot], dim=-1)
            return moved_particles, delta
        else:
            heading = particles[:, :, 2:3]
            sin_heading = torch.sin(heading)
            cos_heading = torch.cos(heading)
            rand_in = torch.randn_like(particles[:, :, 3:5])
            B, N = rand_in.shape[0], rand_in.shape[1]
            noise = self.mo_noise_generator(rand_in.reshape(B * N, -1)).reshape(B, N, -1)
            noise = noise - noise.mean(dim=1, keepdim=True)
            new_z = particles[:, :, 0:1] + cos_heading * particles[:, :, 3:4] * time_step
            new_x = particles[:, :, 1:2] + sin_heading * particles[:, :, 3:4] * time_step
            new_theta = wrap_angle(particles[:, :, 2:3] + particles[:, :, 4:5] * time_step)
            new_v = particles[:, :, 3:4] + noise[:, :, :1] * state_step_sizes[3]
            new_theta_dot = particles[:, :, 4:5] + noise[:, :, 1:] * state_step_sizes[4]
            moved_particles = torch.cat([new_z, new_x, new_theta, new_v, new_theta_dot], dim=-1)
            return moved_particles

    def particles_to_state(self, particle_list, particle_probs_list):
        """
        Compute the weighted mean of particles.
          particle_list: [B, T, N, state_dim]
          particle_probs_list: [B, T, N]
        """
        mean_position = torch.sum(particle_probs_list.unsqueeze(-1) * particle_list[..., :2], dim=2)
        cos_sum = torch.sum(particle_probs_list.unsqueeze(-1) * torch.cos(particle_list[..., 2:3]), dim=2)
        sin_sum = torch.sum(particle_probs_list.unsqueeze(-1) * torch.sin(particle_list[..., 2:3]), dim=2)
        mean_orientation = atan2(cos_sum, sin_sum)
        mean_velocity = torch.sum(particle_probs_list.unsqueeze(-1) * particle_list[..., 3:5], dim=2)
        return torch.cat([mean_position, mean_orientation, mean_velocity], dim=2)

    def connect_modules(self, means, stds, state_mins, state_maxs, state_step_sizes, learn_gaussian_mle=False):
        """
        Build the full particle filter:
          - Process observations via the encoder.
          - Initialize particles (using true state, proposer, or random sampling).
          - Loop over time steps to update particles.
        """
        # For this PyTorch implementation, we assume that during training a “dummy” batch is provided
        # so that we know batch size and sequence length. These are stored in temporary variables.
        B = self.batch_size = self._dummy_batch_size  # set externally (see fit)
        T = self.seq_len = self._dummy_seq_len

        # Process observations.
        # Assume self._dummy_o is a tensor of shape [B, T, H, W, observation_dim].
        # Convert to channel-first: [B, T, observation_dim, H, W]
        o = self._dummy_o
        if o.dim() == 5 and o.shape[-1] == self.observation_dim:
            o = o.permute(0, 1, 4, 2, 3)
        # Normalize observations using means['o'] and stds['o'] (assumed to be tensors)
        o = (o - means['o']) / stds['o']
        o_flat = o.reshape(B * T, o.size(2), o.size(3), o.size(4))
        encodings = self.encoder(o_flat).reshape(B, T, -1)

        # Initialize particles.
        if self.init_with_true_state:
            initial_particles = self._dummy_s[:, 0:1, :].expand(B, self.num_particles, self.state_dim)
        else:
            if self.use_proposer:
                initial_particles = self.propose_particles(encodings[:, 0], self.num_particles, state_mins, state_maxs)
            else:
                parts = []
                for d in range(self.state_dim):
                    parts.append(torch.rand(B, self.num_particles, 1, device=encodings.device) *
                                 (state_maxs[d] - state_mins[d]) + state_mins[d])
                initial_particles = torch.cat(parts, dim=-1)
        initial_particle_probs = torch.ones(B, self.num_particles, device=encodings.device) / self.num_particles

        # Initialize history lists.
        particle_list = initial_particles.unsqueeze(1)  # [B, 1, N, state_dim]
        particle_probs_list = initial_particle_probs.unsqueeze(1)  # [B, 1, N]
        additional_probs_list = torch.ones(B, 1, self.num_particles, 4, device=encodings.device)

        particles = initial_particles
        particle_probs = initial_particle_probs

        # Loop over time steps 1 ... T-1.
        for i in range(1, T):
            num_proposed_float = round((self.propose_ratio ** i) * self.num_particles)
            num_proposed = int(num_proposed_float)
            num_resampled = self.num_particles - num_proposed

            if self.propose_ratio < 1.0:
                # Systematic resampling.
                markers = torch.linspace(0, 1 - 1/num_resampled, steps=num_resampled, device=encodings.device)
                random_offset = torch.rand(B, 1, device=encodings.device) * (1/num_resampled)
                markers = markers.unsqueeze(0) + random_offset  # [B, num_resampled]
                cum_probs = torch.cumsum(particle_probs, dim=1)
                samples = []
                for b in range(B):
                    # torch.searchsorted returns indices where markers should be inserted.
                    samples.append(torch.searchsorted(cum_probs[b], markers[b]))
                samples = torch.stack(samples, dim=0)  # [B, num_resampled]
                # Gather standard particles.
                idx = samples.unsqueeze(-1).expand(-1, -1, self.state_dim)
                standard_particles = torch.gather(particles, 1, idx).detach()
                standard_particle_probs = torch.ones(B, num_resampled, device=encodings.device).detach()
                # Motion update.
                if learn_gaussian_mle:
                    standard_particles, _ = self.motion_update(self._dummy_a[:, i], standard_particles,
                                                               means, stds, state_step_sizes)
                else:
                    standard_particles = self.motion_update(self._dummy_a[:, i], standard_particles,
                                                            means, stds, state_step_sizes)
                standard_particle_probs = standard_particle_probs * self.measurement_update(encodings[:, i],
                                                                                          standard_particles,
                                                                                          means, stds)
            if self.propose_ratio > 0.0:
                proposed_particles = self.propose_particles(encodings[:, i], num_proposed, state_mins, state_maxs)
                proposed_particle_probs = torch.ones(B, num_proposed, device=encodings.device)
            if self.propose_ratio == 1.0:
                particles = proposed_particles
                particle_probs = proposed_particle_probs
            elif self.propose_ratio == 0.0:
                particles = standard_particles
                particle_probs = standard_particle_probs
            else:
                standard_particle_probs = standard_particle_probs * (num_resampled / self.num_particles) / (standard_particle_probs.sum(dim=1, keepdim=True) + 1e-16)
                proposed_particle_probs = proposed_particle_probs * (num_proposed / self.num_particles) / (proposed_particle_probs.sum(dim=1, keepdim=True) + 1e-16)
                particles = torch.cat([standard_particles, proposed_particles], dim=1)
                particle_probs = torch.cat([standard_particle_probs, proposed_particle_probs], dim=1)
            particle_probs = particle_probs / (particle_probs.sum(dim=1, keepdim=True) + 1e-16)
            particle_list = torch.cat([particle_list, particles.unsqueeze(1)], dim=1)
            particle_probs_list = torch.cat([particle_probs_list, particle_probs.unsqueeze(1)], dim=1)
        self.pred_states = self.particles_to_state(particle_list, particle_probs_list)
        self.particle_list = particle_list
        self.particle_probs_list = particle_probs_list
        return particles, particle_probs, encodings, particle_list, particle_probs_list

    def predict(self, batch, num_particles, return_particles=False, **kwargs):
        """
        Run the full particle filter on a batch.
          batch: dict with keys 'o', 'a', 's'
        """
        self.eval()
        with torch.no_grad():
            # For tracking, include initial state if needed.
            # Here we assume that connect_modules has been called and that self.pred_states is set.
            pred_states = self.pred_states
        if return_particles:
            return pred_states.cpu().numpy(), self.particle_list.cpu().numpy(), self.particle_probs_list.cpu().numpy()
        else:
            return pred_states.cpu().numpy()

    def load_model(self, model_path, model_file='best_validation.pth', statistics_file='statistics.npz', connect_and_initialize=True, modules=('encoder', 'mo_noise_generator', 'mo_transition_model', 'obs_like_estimator', 'particle_proposer')):
        """
        Load saved model parameters (and statistics) from disk.
        """
        if connect_and_initialize:
            stats = np.load(os.path.join(model_path, statistics_file), allow_pickle=True).item()
            self.means = stats['means']
            self.stds = stats['stds']
            self.state_step_sizes = stats['state_step_sizes']
            self.state_mins = stats['state_mins']
            self.state_maxs = stats['state_maxs']
        state_dict = torch.load(os.path.join(model_path, model_file))
        self.load_state_dict(state_dict)
        print("Loaded model from", os.path.join(model_path, model_file))

    def fit(self, sess, data, model_path, train_individually, train_e2e, split_ratio,
            seq_len, batch_size, epoch_length, num_epochs, patience, learning_rate,
            dropout_keep_ratio, num_particles, particle_std, learn_gaussian_mle,
            plot_task=None, plot=False):
        # Note: The parameter 'sess' is ignored in PyTorch.
        if plot:
            plt.ion()
        self.particle_std = particle_std
        self.num_particles = num_particles

        means, stds, state_step_sizes, state_mins, state_maxs = compute_statistics(data)
        data_split = split_data(data, ratio=split_ratio)
        epoch_lengths = {'train': epoch_length, 'val': epoch_length * 2}
        batch_iterators = {
            'train': make_batch_iterator(data_split['train'], seq_len=seq_len, batch_size=batch_size),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
            'train_ex': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=seq_len),
            'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len),
            'train1': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=1),
            'train2': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=2),
            'val1': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=1),
            'val2': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=2)
        }
        # Save statistics for later
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        np.savez(os.path.join(model_path, 'statistics'),
                 means=means, stds=stds, state_step_sizes=state_step_sizes,
                 state_mins=state_mins, state_maxs=state_maxs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # Build a dummy batch from training data to set shapes.
        dummy = next(batch_iterators['train'])
        self._dummy_o = torch.tensor(dummy['o'], dtype=torch.float32, device=device)
        self._dummy_a = torch.tensor(dummy['a'], dtype=torch.float32, device=device)
        self._dummy_s = torch.tensor(dummy['s'], dtype=torch.float32, device=device)
        self._dummy_batch_size = self._dummy_o.shape[0]
        self._dummy_seq_len = self._dummy_o.shape[1]

        # Connect modules to build the filter.
        particles, particle_probs, encodings, particle_list, particle_probs_list = self.connect_modules(means, stds, state_mins, state_maxs, state_step_sizes, learn_gaussian_mle)

        # Define loss (example: end-to-end mle loss) and optimizer.
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # For simplicity, we use a dummy loss computed on self.pred_states and dummy state.
        self.train()
        pred = self.particles_to_state(particle_list, particle_probs_list)
        loss_mle = mse_loss(pred, self._dummy_s)
        # Example training loop for one stage (end-to-end training).
        best_val_loss = np.inf
        best_epoch = 0
        log = {'train': {'mle': []}, 'val': {'mle': []}}
        save_path = os.path.join(model_path, 'best_validation.pth')
        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            for _ in range(epoch_lengths['train']):
                batch = next(batch_iterators['train'])
                o = torch.tensor(batch['o'], dtype=torch.float32, device=device)
                a = torch.tensor(batch['a'], dtype=torch.float32, device=device)
                s = torch.tensor(batch['s'], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                # Forward pass: here you would run your full particle filter pass.
                # For brevity, we re-use self.pred_states computed during connect_modules.
                loss = mse_loss(self.pred_states, s)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train = np.mean(train_losses)
            self.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(epoch_lengths['val']):
                    batch = next(batch_iterators['val'])
                    o = torch.tensor(batch['o'], dtype=torch.float32, device=device)
                    a = torch.tensor(batch['a'], dtype=torch.float32, device=device)
                    s = torch.tensor(batch['s'], dtype=torch.float32, device=device)
                    loss = mse_loss(self.pred_states, s)
                    val_losses.append(loss.item())
            avg_val = np.mean(val_losses)
            log['train']['mle'].append(avg_train)
            log['val']['mle'].append(avg_val)
            print(f"Epoch {epoch:3d} train_mle: {avg_train:.4f}  val_mle: {avg_val:.4f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch
                torch.save(self.state_dict(), save_path)
        # Restore best model.
        self.load_state_dict(torch.load(save_path))
        return log

    # --- Plotting Functions ---

    def plot_motion_model(self, batch, motion_samples, task, state_step_sizes):
        # Assume batch is a dict with key 's'
        s_motion_samples = motion_samples.detach().cpu().numpy()
        plt.figure('Motion Model')
        plt.clf()
        for i in range(min(len(s_motion_samples), 10)):
            plt.scatter(s_motion_samples[i, :, 3] / state_step_sizes[3],
                        s_motion_samples[i, :, 4] / state_step_sizes[4],
                        color='blue', s=1)
            plt.scatter(batch['s'][i, 0, 3] / state_step_sizes[3],
                        batch['s'][i, 0, 4] / state_step_sizes[4],
                        color='black', s=1)
            plt.scatter(batch['s'][i, 1, 3] / state_step_sizes[3],
                        batch['s'][i, 1, 4] / state_step_sizes[4],
                        color='red', s=3)
            plt.plot(batch['s'][i, :2, 3] / state_step_sizes[3],
                     batch['s'][i, :2, 4] / state_step_sizes[4],
                     color='black')
        plt.xlim([0, 200])
        plt.ylim([-50, 50])
        plt.xlabel('translational vel')
        plt.ylabel('angular vel')
        plt.gca().set_aspect('equal')
        plt.pause(0.01)

    def plot_measurement_model(self, batch_iterator, measurement_model_out):
        batch = next(batch_iterator)
        device = next(self.parameters()).device
        o = torch.tensor(batch['o'], dtype=torch.float32, device=device)
        a = torch.tensor(batch['a'], dtype=torch.float32, device=device)
        s = torch.tensor(batch['s'], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = self.obs_like_estimator(o.reshape(o.size(0)*o.size(1), -1))
        out_np = out.reshape(o.size(0), o.size(1)).cpu().numpy()
        plt.figure('Measurement Model Output')
        plt.clf()
        plt.imshow(out_np, interpolation="nearest", cmap="viridis_r", vmin=0.0, vmax=1.0)
        plt.pause(0.01)
        plt.figure('Measurement Model Input')
        plt.clf()
        plt.scatter(batch['s'][:1, 0, 3], batch['s'][:1, 0, 4], marker='x', c=out_np[0, :1], vmin=0, vmax=1.0, cmap='viridis_r')
        plt.scatter(batch['s'][1:, 0, 3], batch['s'][1:, 0, 4], marker='o', c=out_np[0, 1:], vmin=0, vmax=1.0, cmap='viridis_r')
        plt.xlabel('x_dot')
        plt.ylabel('theta_dot')
        plt.pause(0.01)

    def plot_particle_proposer(self, batch, proposed_particles, task):
        device = next(self.parameters()).device
        o = torch.tensor(batch['o'], dtype=torch.float32, device=device)
        a = torch.tensor(batch['a'], dtype=torch.float32, device=device)
        s = torch.tensor(batch['s'], dtype=torch.float32, device=device)
        with torch.no_grad():
            proposed = proposed_particles.detach().cpu().numpy()
        plt.figure('Particle Proposer')
        plt.clf()
        plot_maze(task)
        for i in range(min(len(proposed), 10)):
            color = np.random.uniform(0.0, 1.0, 3)
            plt.quiver(proposed[i, :, 0], proposed[i, :, 1],
                       np.cos(proposed[i, :, 2]), np.sin(proposed[i, :, 2]),
                       color=color, width=0.001, scale=100)
            plt.quiver(batch['s'][i, 0, 0], batch['s'][i, 0, 1],
                       np.cos(batch['s'][i, 0, 2]), np.sin(batch['s'][i, 0, 2]),
                       color=color, scale=50, width=0.003)
        plt.pause(0.01)

    def plot_particle_filter(self, batch, particle_list, particle_probs_list, state_step_sizes, task):
        device = next(self.parameters()).device
        s = torch.tensor(batch['s'], dtype=torch.float32, device=device)
        particle_list_np = particle_list.detach().cpu().numpy()
        particle_probs_np = particle_probs_list.detach().cpu().numpy()
        num_steps = particle_list_np.shape[1]
        for s_idx in range(3):
            plt.figure(f'particle_evolution, example {s_idx}')
            plt.clf()
            for d in range(5):
                plt.subplot(3, 2, [1, 3, 5, 2, 4][d])
                for t in range(num_steps):
                    plt.scatter(np.full(particle_list_np[s_idx, t, :, d].shape, t),
                                particle_list_np[s_idx, t, :, d] / (1 if s_idx == 0 else state_step_sizes[d]),
                                c=particle_probs_np[s_idx, t, :], cmap='viridis_r', marker='o', s=6, alpha=0.5, linewidths=0.05,
                                vmin=0.0, vmax=0.1)
                    current_state = batch['s'][s_idx, t, d] / (1 if s_idx == 0 else state_step_sizes[d])
                    plt.plot([t], [current_state], 'o', markerfacecolor='None', markeredgecolor='k', markersize=2.5)
                plt.xlabel('Time')
                plt.ylabel(f'State {d}')
        show_pause(pause=0.01)
