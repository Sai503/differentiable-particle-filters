# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.data_utils import wrap_angle, compute_statistics, split_data, make_batch_iterator, make_repeating_batch_iterator
from utils.method_utils import atan2, compute_sq_distance
from utils.plotting_utils import plot_maze, show_pause

from .modules.encoder import Encoder
from .modules.motion_model import MotionModel
from .modules.measurement_model import MeasurementModel
from .modules.proposer import Proposer

# Helper function to move batch data to the target device
def move_batch_to_device(batch, device):
    """Moves all tensor values in a dictionary to the specified device."""
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            try:
                batch_device[key] = torch.from_numpy(value).float().to(device)
            except TypeError:
                batch_device[key] = value
        else:
            batch_device[key] = value
    return batch_device

class DPF(nn.Module):
    def __init__(self, init_with_true_state, learn_odom, use_proposer,
                 propose_ratio, proposer_keep_ratio, min_obs_likelihood,
                 dropout_keep_prob=1.0):
        """
        PyTorch version of the particle filter.
        """
        super(DPF, self).__init__()
        self.init_with_true_state = init_with_true_state
        self.learn_odom = learn_odom
        self.use_proposer = use_proposer and (not init_with_true_state)
        self.propose_ratio = propose_ratio if not self.init_with_true_state else 0.0
        self.min_obs_likelihood = min_obs_likelihood
        self.state_dim = 3
        self.dropout_keep_prob = dropout_keep_prob
        self.proposer_keep_ratio = proposer_keep_ratio

        # Initialize modules
        self.encoder = Encoder(dropout_keep_prob)
        self.motion_model = MotionModel(self.state_dim, learn_odom)
        self.measurement_model = MeasurementModel(self.state_dim, min_obs_likelihood)
        self.proposer = Proposer(self.state_dim, proposer_keep_ratio) if self.use_proposer else None

    def forward_encoder(self, o):
        return self.encoder(o)

    def measurement_update(self, encoding, particles):
        return self.measurement_model(encoding, particles, self.means_t, self.stds_t)

    def motion_update(self, actions, particles, stop_sampling_gradient=False):
        return self.motion_model(actions, particles, self.stds_t, self.means_t, self.state_step_sizes_t, stop_sampling_gradient)

    def propose_particles(self, encoding, num_particles):
        if self.proposer:
            return self.proposer(encoding, num_particles, self.state_mins_t, self.state_maxs_t)
        else:
            raise RuntimeError("Proposer is not initialized.")

    def particles_to_state(self, particle_list, particle_probs_list):
        mean_position = torch.sum(particle_probs_list.unsqueeze(-1) * particle_list[..., :2], dim=2)
        cos_sum = torch.sum(particle_probs_list.unsqueeze(-1) * torch.cos(particle_list[..., 2:3]), dim=2)
        sin_sum = torch.sum(particle_probs_list.unsqueeze(-1) * torch.sin(particle_list[..., 2:3]), dim=2)
        mean_orientation = torch.atan2(sin_sum, cos_sum)
        return torch.cat([mean_position, mean_orientation], dim=-1)

    def connect_modules(self, batch):
        o_batch = batch['o']
        a_batch = batch['a']
        s_batch = batch['s']
        device = s_batch.device
        dtype = s_batch.dtype

        B, T = o_batch.shape[0], o_batch.shape[1]

        o_flat = o_batch.reshape(B * T, *o_batch.shape[2:])
        if o_flat.dim() == 4 and o_flat.size(-1) == 3:
            o_flat = o_flat.permute(0, 3, 1, 2).contiguous()
        if o_flat.dtype != torch.float32:
            o_flat = o_flat.float()

        o_flat_norm = (o_flat - self.means_t['o'].unsqueeze(0)) / (self.stds_t['o'].unsqueeze(0) + 1e-8)
        encodings = self.forward_encoder(o_flat_norm).view(B, T, -1)

        if self.init_with_true_state:
            initial_particles = s_batch[:, 0].unsqueeze(1).expand(B, self.num_particles, -1)
        else:
            if self.use_proposer:
                initial_particles = self.propose_particles(encodings[:, 0], self.num_particles)
            else:
                state_mins = self.state_mins_t
                state_maxs = self.state_maxs_t
                parts = []
                for d in range(self.state_dim):
                    rand_tensor = torch.rand(B, self.num_particles, 1, device=device, dtype=dtype)
                    parts.append(rand_tensor * (state_maxs[d] - state_mins[d]) + state_mins[d])
                initial_particles = torch.cat(parts, dim=-1)

        initial_particle_probs = torch.ones(B, self.num_particles, device=device, dtype=dtype) / self.num_particles

        particle_list = [initial_particles]
        particle_probs_list = [initial_particle_probs]

        particles = initial_particles
        particle_probs = initial_particle_probs

        for i in range(1, T):
            num_proposed_float = (self.propose_ratio ** float(i)) * float(self.num_particles)
            num_proposed = int(round(num_proposed_float))
            num_resampled = self.num_particles - num_proposed

            combined_particles_list = []
            combined_probs_list = []

            if num_resampled > 0:
                markers = torch.linspace(0, 1 - 1/num_resampled, steps=num_resampled,
                                           device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
                random_offset = torch.rand(B, 1, device=device, dtype=dtype) * (1/num_resampled)
                markers = markers + random_offset
                cum_probs = torch.cumsum(particle_probs, dim=1)

                indices = torch.searchsorted(cum_probs, markers, right=True)
                indices = torch.clamp(indices, 0, self.num_particles - 1)

                indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.state_dim)
                standard_particles = torch.gather(particles, 1, indices_expanded)

                standard_particles = standard_particles.detach()

                standard_particles = self.motion_update(a_batch[:, i], standard_particles)

                meas = self.measurement_update(encodings[:, i], standard_particles)
                standard_particle_probs = torch.ones(B, num_resampled, device=device, dtype=dtype) * meas

                combined_particles_list.append(standard_particles)
                combined_probs_list.append(standard_particle_probs)

            if num_proposed > 0:
                proposed_particles = self.propose_particles(encodings[:, i], num_proposed)
                meas_proposed = self.measurement_update(encodings[:, i], proposed_particles)
                proposed_particle_probs = torch.ones(B, num_proposed, device=device, dtype=dtype) * meas_proposed

                combined_particles_list.append(proposed_particles)
                combined_probs_list.append(proposed_particle_probs)

            if not combined_particles_list:
                if self.num_particles > 0:
                    raise ValueError("No particles generated in loop, but num_particles > 0.")
                else:
                    particles = torch.empty((B, 0, self.state_dim), device=device, dtype=dtype)
                    particle_probs = torch.empty((B, 0), device=device, dtype=dtype)
            else:
                particles = torch.cat(combined_particles_list, dim=1)
                particle_probs = torch.cat(combined_probs_list, dim=1)

                prob_sum = particle_probs.sum(dim=1, keepdim=True)
                particle_probs = particle_probs / (prob_sum + 1e-16)

            particle_list.append(particles)
            particle_probs_list.append(particle_probs)

        particle_list_stacked = torch.stack(particle_list, dim=1)
        particle_probs_list_stacked = torch.stack(particle_probs_list, dim=1)

        pred_states = self.particles_to_state(particle_list_stacked, particle_probs_list_stacked)

        self.pred_states = pred_states
        self.particle_list = particle_list_stacked
        self.particle_probs_list = particle_probs_list_stacked

        return particle_list_stacked, particle_probs_list_stacked, encodings

    def predict(self, batch, num_particles, return_particles=False):
        self.eval()
        if self.device is None:
            raise ValueError("Device not set. Call fit() or manually set self.device and statistics.")
        if self.num_particles is None or self.num_particles != num_particles:
            self.num_particles = num_particles

        batch_device = move_batch_to_device(batch, self.device)

        with torch.no_grad():
            particle_list, particle_probs_list, _ = self.connect_modules(batch_device)
            pred_states = self.pred_states

        if return_particles:
            return pred_states, particle_list, particle_probs_list
        else:
            return pred_states

    def load_model(self, model_path, model_file='best_validation.pth',
                   statistics_file='statistics.npz', device='cpu'):
        stats_path = os.path.join(model_path, statistics_file)
        if os.path.exists(stats_path):
            stats = np.load(stats_path, allow_pickle=True)
            stats = dict(stats)
            self.means = stats['means'].item()
            self.stds = stats['stds'].item()
            self.state_step_sizes = stats['state_step_sizes']
            self.state_mins = stats['state_mins']
            self.state_maxs = stats['state_maxs']

            self._stats_to_tensors(device)
            print(f"Loaded statistics from {stats_path}")
        else:
            print(f"Warning: Statistics file not found at {stats_path}. Cannot initialize self.*_t tensors.")

        state_dict_path = os.path.join(model_path, model_file)
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device)
            self.load_state_dict(state_dict)
            self.to(device)
            self.device = device
            print(f"Loaded model state_dict from {state_dict_path} to device {device}")
        else:
            print(f"Warning: Model state_dict file not found at {state_dict_path}.")

    def _stats_to_tensors(self, device):
        dtype = torch.float32
        try:
            num_obs_channels = np.asarray(self.means['o']).squeeze().shape[-1]
            if num_obs_channels <= 1:
                num_obs_channels = np.asarray(self.means['o']).squeeze().size
            print(f"DEBUG: Inferred num_obs_channels = {num_obs_channels}")
            if num_obs_channels not in [3, 4]:
                print(f"Warning: Inferred num_obs_channels ({num_obs_channels}) seems unusual. Double-check data.")
        except Exception as e:
            print(f"Warning: Could not infer num_obs_channels from self.means['o']. Defaulting to 4. Error: {e}")
            num_obs_channels = 4

        self.means_t = {}
        for k, v_np in self.means.items():
            v_squeezed_np = np.asarray(v_np).squeeze()
            v_tensor = torch.tensor(v_squeezed_np, dtype=dtype, device=device)

            if k == 's':
                if v_tensor.numel() != 3:
                    raise ValueError(f"Expected means['s'] to have 3 elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                self.means_t[k] = v_tensor.view(1, 1, 3)
            elif k == 'o':
                if v_tensor.numel() != num_obs_channels:
                    raise ValueError(f"Expected means['o'] to have {num_obs_channels} elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                self.means_t[k] = v_tensor.view(num_obs_channels)
            else:
                self.means_t[k] = v_tensor

        self.stds_t = {}
        for k, v_np in self.stds.items():
            v_squeezed_np = np.asarray(v_np).squeeze()
            v_tensor = torch.tensor(v_squeezed_np, dtype=dtype, device=device)

            if k == 's':
                if v_tensor.numel() != 3:
                    raise ValueError(f"Expected stds['s'] to have 3 elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                self.stds_t[k] = v_tensor.view(1, 1, 3)
            elif k == 'a':
                if v_tensor.numel() != 3:
                    raise ValueError(f"Expected stds['a'] to have 3 elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                self.stds_t[k] = v_tensor.view(1, 1, 3)
            elif k == 'o':
                if v_tensor.numel() != num_obs_channels:
                    raise ValueError(f"Expected stds['o'] to have {num_obs_channels} elements, but got shape {np.asarray(v_np).shape} -> {v_tensor.shape}")
                self.stds_t[k] = v_tensor.view(num_obs_channels)
            else:
                self.stds_t[k] = v_tensor

        self.state_step_sizes_t = torch.tensor(self.state_step_sizes, dtype=dtype, device=device)
        self.state_mins_t = torch.tensor(self.state_mins, dtype=dtype, device=device)
        self.state_maxs_t = torch.tensor(self.state_maxs, dtype=dtype, device=device)
        print(f"Converted statistics to tensors on device {device}")
        if 'o' in self.means_t: print(f"DEBUG: Final self.means_t['o'] shape: {self.means_t['o'].shape}")
        if 'o' in self.stds_t: print(f"DEBUG: Final self.stds_t['o'] shape: {self.stds_t['o'].shape}")
        if 'a' in self.stds_t: print(f"DEBUG: Final self.stds_t['a'] shape: {self.stds_t['a'].shape}")

    def compile_training_stages(self, learning_rate, plot_task):
        """
        Creates a dictionary of training stages.
        """
        train_stages = {}
        device = self.device
        dtype = next(self.parameters()).dtype

        def train_motion_sampling_fn(batch):
            s_batch = batch['s']
            a_batch = batch['a']
            initial_state_expanded = s_batch[:, 0:1].expand(-1, self.num_particles, -1)
            motion_samples = self.motion_update(a_batch[:, 1], initial_state_expanded)
            sq_distance = compute_sq_distance(motion_samples, s_batch[:, 1:2], self.state_step_sizes_t)
            return sq_distance.mean()

        optimizer_motion = torch.optim.Adam(self.motion_model.parameters(), lr=learning_rate)
        train_stages['train_motion_sampling'] = {
            'loss_fn': train_motion_sampling_fn,
            'optimizer': optimizer_motion,
            'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
            'monitor_losses': ['loss'],
            'validation_loss': 'loss',
        }

        def train_measurement_model_fn(batch):
            s_batch = batch['s']
            o_batch = batch['o']
            B = s_batch.size(0)
            if B <= 1:
                return torch.tensor(0.0, device=device)

            test_particles = s_batch[:, 0].unsqueeze(1).expand(B, B, -1)
            o_first = o_batch[:, 0]
            o_first_norm = (o_first - self.means_t['o']) / (self.stds_t['o'] + 1e-8)
            encoding = self.forward_encoder(o_first_norm)

            encoding_expanded = encoding.unsqueeze(1).expand(B, B, -1)
            encoding_flat = encoding_expanded.reshape(B * B, -1)
            test_particles_flat = test_particles.reshape(B * B, 1, -1)
            measurement_out_flat = self.measurement_update(encoding_flat, test_particles_flat)
            measurement_out = measurement_out_flat.view(B, B)

            correct_samples = torch.diag(measurement_out)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            incorrect_samples = measurement_out[mask]

            loss_correct = -torch.log(correct_samples + 1e-16).sum() / B
            num_incorrect = B * (B - 1)
            loss_incorrect = -torch.log(1.0 - incorrect_samples + 1e-16).sum() / num_incorrect if num_incorrect > 0 else torch.tensor(0.0, device=device)
            return loss_correct + loss_incorrect

        params_measurement = list(self.encoder.parameters()) + list(self.measurement_model.parameters())
        optimizer_measurement = torch.optim.Adam(params_measurement, lr=learning_rate)
        train_stages['train_measurement_model'] = {
            'loss_fn': train_measurement_model_fn,
            'optimizer': optimizer_measurement,
            'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
            'monitor_losses': ['loss'],
            'validation_loss': 'loss',
        }

        if self.use_proposer:
            def train_particle_proposer_fn(batch):
                s_batch = batch['s']
                o_batch = batch['o']
                with torch.no_grad():
                    o_first_norm = (o_batch[:, 0] - self.means_t['o']) / (self.stds_t['o'] + 1e-8)
                    encoding = self.forward_encoder(o_first_norm)

                proposed_particles = self.propose_particles(encoding, self.num_particles)
                sq_distance = compute_sq_distance(proposed_particles, s_batch[:, 0:1].expand(-1, self.num_particles, -1), self.state_step_sizes_t)
                std_val = 0.2
                norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
                activations = (1 / float(self.num_particles)) * norm_const * torch.exp(-sq_distance / (2.0 * std_val ** 2))
                return (-torch.log(1e-16 + torch.sum(activations, dim=1))).mean()

            optimizer_proposer = torch.optim.Adam(self.proposer.parameters(), lr=learning_rate)
            train_stages['train_particle_proposer'] = {
                'loss_fn': train_particle_proposer_fn,
                'optimizer': optimizer_proposer,
                'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
                'monitor_losses': ['loss'],
                'validation_loss': 'loss',
            }

        def train_e2e_fn(batch):
            s_batch = batch['s']
            particle_list, particle_probs_list, _ = self.connect_modules(batch)
            std_val = self.particle_std
            true_state_expanded = s_batch.unsqueeze(2)
            sq_dist = compute_sq_distance(particle_list, true_state_expanded, self.state_step_sizes_t)
            norm_const = 1.0 / torch.sqrt(torch.tensor(2 * np.pi * std_val ** 2, device=device, dtype=dtype))
            activations = particle_probs_list * norm_const * torch.exp(-sq_dist / (2.0 * std_val ** 2))
            log_likelihood_per_step = torch.log(1e-16 + torch.sum(activations, dim=2))
            mle_loss = -log_likelihood_per_step.mean()
            pred_states = self.pred_states
            mse_last_step = compute_sq_distance(pred_states[:, -1:], s_batch[:, -1:], self.state_step_sizes_t).mean()
            return mle_loss, mse_last_step

        optimizer_e2e = torch.optim.Adam(self.parameters(), lr=learning_rate)
        train_stages['train_e2e'] = {
            'loss_fn': train_e2e_fn,
            'optimizer': optimizer_e2e,
            'batch_iterator_names': {'train': 'train', 'val': 'val'},
            'monitor_losses': ['mle', 'mse_last'],
            'validation_loss': 'mse_last',
        }

        return train_stages

    def fit(self, data, model_path, train_individually, train_e2e, split_ratio,
            seq_len, batch_size, epoch_length, num_epochs, patience,
            learning_rate, dropout_keep_ratio, num_particles, particle_std,
            plot_task=None, plot=False):
        """
        Full training loop. Handles device placement and training stages.
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
        if hasattr(self.encoder, 'dropout'):
            self.encoder.dropout.p = 1.0 - dropout_keep_ratio
        if hasattr(self.proposer, 'network'):
            for layer in self.proposer.network:
                if isinstance(layer, nn.Dropout):
                    layer.p = 1.0 - self.proposer_keep_ratio
                    break

        # --- Preprocess data and compute statistics ---
        data_split = split_data(data, ratio=split_ratio)
        means, stds, state_step_sizes, state_mins, state_maxs = compute_statistics(data_split['train'])

        self.means, self.stds = means, stds
        self.state_step_sizes, self.state_mins, self.state_maxs = state_step_sizes, state_mins, state_maxs
        self._stats_to_tensors(device)

        # --- Create Batch Iterators ---
        epoch_lengths = {'train': epoch_length, 'val': epoch_length * 2}
        batch_iterators = {
            'train': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=seq_len),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
            'train1': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=2),
            'val1': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=2),
            'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len),
        }

        # --- Compile Training Stages ---
        train_stages = self.compile_training_stages(learning_rate, plot_task)

        # --- Save Statistics ---
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        np.savez(os.path.join(model_path, 'statistics.npz'),
                 means=self.means, stds=self.stds, state_step_sizes=self.state_step_sizes,
                 state_mins=self.state_mins, state_maxs=self.state_maxs)
        print(f"Saved statistics to {os.path.join(model_path, 'statistics.npz')}")

        # --- Define Curriculum ---
        curriculum = []
        if train_individually:
            if self.learn_odom and 'train_motion_sampling' in train_stages:
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

        # --- Initialize Logs ---
        log = {}
        for stage_name in curriculum:
            stage_info = train_stages[stage_name]
            log[stage_name] = {'train': {}, 'val': {}}
            for loss_key in stage_info['monitor_losses']:
                log[stage_name]['train'][loss_key] = {'mean': [], 'se': []}
                log[stage_name]['val'][loss_key] = {'mean': [], 'se': []}

        best_overall_val_loss = float('inf')

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
            monitor_keys = stage_info['monitor_losses']
            validation_loss_key = stage_info['validation_loss']

            while epoch < num_epochs and (epoch - best_stage_epoch) < patience:
                epoch_loss_lists = {'train': {k: [] for k in monitor_keys}, 'val': {k: [] for k in monitor_keys}}

                for phase in ['train', 'val']:
                    is_train = phase == 'train'
                    self.train(is_train)
                    iterator = batch_iterators[train_iter_name if is_train else val_iter_name]
                    num_steps = epoch_lengths[phase]

                    for step in range(num_steps):
                        batch = next(iterator)
                        batch_device = move_batch_to_device(batch, device)

                        optimizer.zero_grad(set_to_none=True)
                        with torch.set_grad_enabled(is_train):
                            output = loss_fn(batch_device)
                            if isinstance(output, tuple):
                                primary_loss = output[0]
                                monitor_values = output
                            else:
                                primary_loss = output
                                monitor_values = (output,)

                            if is_train and torch.is_tensor(primary_loss) and primary_loss.requires_grad:
                                primary_loss.backward()
                                optimizer.step()

                        for i, key in enumerate(monitor_keys):
                            if i < len(monitor_values):
                                epoch_loss_lists[phase][key].append(monitor_values[i].item())
                            else:
                                epoch_loss_lists[phase][key].append(0.0)

                txt = ""
                current_val_loss = float('inf')
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
                            if phase == 'val' and key == validation_loss_key:
                                current_val_loss = mean_loss
                        else:
                            log[stage_name][phase][key]['mean'].append(np.nan)
                            log[stage_name][phase][key]['se'].append(np.nan)
                            txt += f'N/A/'
                    txt = txt[:-1] + ' -- '

                if current_val_loss < best_stage_val_loss:
                    best_stage_val_loss = current_val_loss
                    best_stage_epoch = epoch
                    if current_val_loss < best_overall_val_loss:
                        best_overall_val_loss = current_val_loss
                        torch.save(self.state_dict(), os.path.join(model_path, 'best_validation.pth'))
                        print(f"  * Overall: New best validation loss: {current_val_loss:.4f}. Saved model.")
                    txt_prefix = f"epoch {epoch:3d} >> "
                else:
                    txt_prefix = f"epoch {epoch:3d} == "

                print(f"{txt_prefix}Stage: {stage_name} -- {txt}Patience: {epoch - best_stage_epoch}/{patience}")

                if plot and plot_task:
                    try:
                        plot_batch = next(batch_iterators['val_ex'])
                        plot_batch_device = move_batch_to_device(plot_batch, device)
                        self.plot_particle_filter(plot_batch_device, plot_task)
                    except StopIteration:
                        batch_iterators['val_ex'] = make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len)
                    except Exception as e:
                        print(f"Error during plotting: {e}")

                epoch += 1

        print("\n--- Training Finished ---")
        if os.path.exists(os.path.join(model_path, 'best_validation.pth')):
            print(f"Loading final best model from {os.path.join(model_path, 'best_validation.pth')}")
            self.load_state_dict(torch.load(os.path.join(model_path, 'best_validation.pth'), map_location=device))

        return log

    # === Plotting functions ===

    def plot_motion_model(self, batch_cpu, motion_samples_cpu, task):
        """Plots motion model samples vs ground truth."""
        plt.figure('Motion Model')
        plt.clf()
        ax = plt.gca()
        plot_maze(task, ax=ax)

        num_to_plot = min(motion_samples_cpu.shape[0], 10)
        for i in range(num_to_plot):
            s_motion = motion_samples_cpu[i]  # Samples for example i [N, 3]
            s_true = batch_cpu['s'][i]       # True states for example i [T, 3] (T=2 here)

            ax.quiver(s_motion[:, 0], s_motion[:, 1],
                      np.cos(s_motion[:, 2]), np.sin(s_motion[:, 2]),
                      color='blue', width=0.001, scale=100, alpha=0.5)
            ax.quiver(s_true[0, 0], s_true[0, 1], np.cos(s_true[0, 2]), np.sin(s_true[0, 2]),
                      color='black', scale=50, width=0.003, label='Start (t=0)' if i == 0 else "")
            ax.quiver(s_true[1, 0], s_true[1, 1], np.cos(s_true[1, 2]), np.sin(s_true[1, 2]),
                      color='red', scale=50, width=0.003, label='Target (t=1)' if i == 0 else "")

        ax.set_aspect('equal')
        ax.set_title('Motion Model (Blue: Samples)')
        ax.legend(fontsize='small')
        show_pause(pause=0.01)

    def plot_measurement_model(self, measurement_model_out_cpu):
        """Plots the measurement likelihood matrix."""
        plt.figure('Measurement Model Output')
        plt.clf()
        ax = plt.gca()
        im = ax.imshow(measurement_model_out_cpu, interpolation="nearest", cmap="coolwarm")
        ax.set_title('Measurement Likelihood (Obs vs State)')
        ax.set_xlabel('Particle State Index (from Batch)')
        ax.set_ylabel('Observation Index (from Batch)')
        plt.colorbar(im, ax=ax)
        show_pause(pause=0.01)

    def plot_particle_proposer(self, batch_cpu, proposed_particles_cpu, task):
        """Plots proposed particles vs true initial state."""
        plt.figure('Particle Proposer')
        plt.clf()
        ax = plt.gca()
        plot_maze(task, ax=ax)

        num_to_plot = min(proposed_particles_cpu.shape[0], 10)
        for i in range(num_to_plot):
            s_proposed = proposed_particles_cpu[i]  # Proposed particles [N, 3]
            s_true_initial = batch_cpu['s'][i, 0]  # True state at t=0 [3]
            color = np.random.uniform(0.0, 1.0, 3)

            ax.quiver(s_proposed[:, 0], s_proposed[:, 1],
                      np.cos(s_proposed[:, 2]), np.sin(s_proposed[:, 2]),
                      color=color, width=0.001, scale=100, alpha=0.5)
            ax.quiver(s_true_initial[0], s_true_initial[1],
                      np.cos(s_true_initial[2]), np.sin(s_true_initial[2]),
                      color=color, scale=50, width=0.003, label=f'True State {i}' if i < 3 else "")

        ax.set_aspect('equal')
        ax.set_title('Particle Proposer (Colors: Proposed Clouds + True State)')
        ax.legend(fontsize='small')
        show_pause(pause=0.01)

    def plot_particle_filter(self, batch_cpu, task, num_particles_plot=500):
        """Plots the filter evolution for one example from the batch."""
        print(f"Plotting particle filter for task {task}...")
        device = self.device or torch.device('cpu')
        self.to(device)
        self._stats_to_tensors(device)

        batch_first_tensor = {k: torch.from_numpy(v[0:1]).float() if isinstance(v, np.ndarray) else v for k, v in batch_cpu.items()}
        pred, particle_list, particle_probs_list = self.predict(batch_first_tensor, num_particles=num_particles_plot, return_particles=True)

        pred_cpu = pred[0].detach().cpu().numpy()
        particles_cpu = particle_list[0].detach().cpu().numpy()
        probs_cpu = particle_probs_list[0].detach().cpu().numpy()
        true_states_cpu = batch_cpu['s'][0]

        T = particles_cpu.shape[0]
        N = particles_cpu.shape[1]

        head_scale = 1.5
        quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 40., 'width': 0.003,
                       'headlength': 5 * head_scale, 'headwidth': 3 * head_scale,
                       'headaxislength': 4.5 * head_scale}
        marker_kwargs = {'markersize': 4.5, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}
        color_list = plt.cm.tab10(np.linspace(0, 1, 10))
        pf_color = color_list[1]

        num_rows = 4
        num_cols = 5
        num_plots = num_rows * num_cols
        num_steps_to_plot = min(T, num_plots - 1)

        fig = plt.figure(f"Particle Filter Evolution (Example 0, N={N})", figsize=[12, 9.6])
        fig.clf()

        for i in range(num_steps_to_plot):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.cla()
            plot_maze(task, margin=5, linewidth=0.5, ax=ax)

            s_particle = particles_cpu[i]
            s_probs = probs_cpu[i]

            clim = [0.0, 2.0 / N]

            ax.quiver(s_particle[:, 0], s_particle[:, 1],
                      np.cos(s_particle[:, 2]), np.sin(s_particle[:, 2]),
                      s_probs, cmap='viridis_r', clim=clim,
                      alpha=1.0, **quiv_kwargs)

            current_state = true_states_cpu[i]
            ax.quiver(current_state[0], current_state[1],
                      np.cos(current_state[2]), np.sin(current_state[2]),
                      color="red", **quiv_kwargs)
            ax.plot(current_state[0], current_state[1], 'or', **marker_kwargs)

            ax.set_title(f'Step {i}')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        ax = fig.add_subplot(num_rows, num_cols, num_plots)
        ax.cla()
        plot_maze(task, margin=5, linewidth=0.5, ax=ax)
        ax.plot(true_states_cpu[:T, 0], true_states_cpu[:T, 1], '-', linewidth=0.6, color='red', label='True Traj.')
        ax.plot(pred_cpu[:T, 0], pred_cpu[:T, 1], '-', linewidth=0.6, color=pf_color, label='PF Traj.')
        ax.plot(true_states_cpu[0, 0], true_states_cpu[0, 1], '.', color='red', markersize=3)
        ax.plot(pred_cpu[0, 0], pred_cpu[0, 1], '.', color=pf_color, markersize=3)
        ax.set_title(f'Full Trajectory (T={T})')
        ax.legend(fontsize='small', loc='best')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        plt.tight_layout(pad=0.5)
        show_pause(pause=0.01)

