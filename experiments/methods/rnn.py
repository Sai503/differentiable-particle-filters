import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime # For timing epochs

# Assuming these utils handle torch tensors correctly or return numpy arrays
# Use the standard data_utils, not KITTI specific ones for this generic RNN
from utils.data_utils import (compute_statistics, split_data,
                              make_batch_iterator, make_repeating_batch_iterator)
# from utils.method_utils import compute_sq_distance # Not directly used here, but compute_statistics is

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

class RNN(nn.Module):
    def __init__(self, init_with_true_state=False, model='2lstm',
                 dropout_keep_prob=1.0, # Added dropout config here
                 obs_channels=3, obs_height=24, obs_width=24, # Make obs dims configurable
                 action_dim=3, state_dim=3, # Make state/action dims configurable
                 **unused_kwargs):
        """
        PyTorch implementation of an RNN model with GPU support.
        - init_with_true_state: if True, encoder output is augmented with tracking info.
        - model: one of '2lstm', '2gru', or 'ff'.
        - dropout_keep_prob: Probability of *keeping* units (1.0 = no dropout).
        """
        super(RNN, self).__init__()
        self.init_with_true_state = init_with_true_state
        self.model_type = model # Renamed from 'model' to avoid conflict with nn.Module.model
        self.obs_channels = obs_channels
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.action_dim = action_dim
        self.state_dim = state_dim

        # --- Encoder for observations ---
        # Calculate flattened size dynamically
        self._encoder_output_flat_size = self._compute_encoder_output_size_dynamic()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.obs_channels, 16, kernel_size=3, stride=2, padding=1), # H,W -> H/2, W/2
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # H/2,W/2 -> H/4, W/4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4,W/4 -> H/8, W/8
            nn.ReLU(),
        )
        self.encoder_flatten = nn.Flatten()
        # Dropout layer - p is the probability of *zeroing* an element
        self.encoder_dropout = nn.Dropout(p=1.0 - dropout_keep_prob)
        self.encoder_linear = nn.Linear(self._encoder_output_flat_size, 128)
        self.encoder_relu = nn.ReLU() # Separate ReLU for clarity

        # --- Recurrent network ---
        # Compute input size for the RNN
        extra = self.state_dim + 1 if self.init_with_true_state else 0 # state_dim + flag
        rnn_input_size = 128 + extra + self.action_dim # 128 from encoder, extra, action_dim

        if self.model_type == '2lstm':
            self.rnn1 = nn.LSTM(input_size=rnn_input_size, hidden_size=512, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
            self.rnn_layers = nn.ModuleList([self.rnn1, self.rnn2]) # Group for easier handling
        elif self.model_type == '2gru':
            self.rnn1 = nn.GRU(input_size=rnn_input_size, hidden_size=512, batch_first=True)
            self.rnn2 = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
            self.rnn_layers = nn.ModuleList([self.rnn1, self.rnn2])
        elif self.model_type == 'ff':
            # Use ModuleList to be consistent, though Sequential is also fine
            self.rnn_layers = nn.ModuleList([
                nn.Linear(rnn_input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
            ])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # --- Belief decoder ---
        self.belief_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim) # Output state_dim
        )

        # --- Statistics & Device Placeholders ---
        self.means_t = None
        self.stds_t = None
        # state_step_sizes, state_mins, state_maxs not used by RNN directly, but store numpy versions
        self.means = None
        self.stds = None
        self.state_step_sizes = None
        self.state_mins = None
        self.state_maxs = None
        self.device = None # Will be set in fit/load_model

    def _compute_encoder_output_size_dynamic(self):
        """Helper to compute the flattened size after conv layers dynamically."""
        dummy_input = torch.zeros(1, self.obs_channels, self.obs_height, self.obs_width)
        with torch.no_grad():
            dummy_output = self.encoder_conv(dummy_input)
        return int(np.prod(dummy_output.shape[1:]))

    def _stats_to_tensors(self, device):
        """Converts loaded numpy statistics to tensors on the specified device."""
        if self.means is None or self.stds is None:
            print("Warning: Statistics (means, stds) not set. Cannot create tensors.")
            return

        dtype = torch.float32 # Default dtype
        self.means_t = {}
        self.stds_t = {}

        for k, v in self.means.items():
            tensor = torch.tensor(v, dtype=dtype, device=device)
            # Add necessary dimensions for broadcasting
            # Assume means['o'] needs [C, H, W] or similar, means['s'], means['a'] are [D]
            if k == 'o':
                # Needs shape like [1, 1, C, 1, 1] or [1, 1, C, H, W]?
                # Assuming normalization is per-channel: [1, 1, C, 1, 1]
                if tensor.ndim == 1: # Assume C
                    tensor = tensor.reshape(1, 1, -1, 1, 1)
                elif tensor.ndim == 3: # Assume C, H, W - maybe just use channel means?
                     print(f"Warning: means['o'] has shape {v.shape}. Using mean across H, W.")
                     tensor = tensor.mean(dim=(1, 2)).reshape(1, 1, -1, 1, 1)
                else: # Fallback
                    print(f"Warning: Unexpected shape for means['o']: {v.shape}. Reshaping to [1, 1, C, 1, 1].")
                    tensor = tensor.reshape(1, 1, -1, 1, 1)
            elif k in ['s', 'a']:
                # Needs shape like [1, 1, D] for broadcasting with [B, T, D]
                tensor = tensor.reshape(1, 1, -1)
            self.means_t[k] = tensor

        for k, v in self.stds.items():
            # Add epsilon during conversion for stability
            tensor = torch.tensor(v + 1e-8, dtype=dtype, device=device)
            if k == 'o':
                if tensor.ndim == 1: tensor = tensor.reshape(1, 1, -1, 1, 1)
                elif tensor.ndim == 3:
                     print(f"Warning: stds['o'] has shape {v.shape}. Using mean across H, W.")
                     tensor = tensor.mean(dim=(1, 2)).reshape(1, 1, -1, 1, 1)
                else:
                    print(f"Warning: Unexpected shape for stds['o']: {v.shape}. Reshaping to [1, 1, C, 1, 1].")
                    tensor = tensor.reshape(1, 1, -1, 1, 1)
            elif k in ['s', 'a']:
                tensor = tensor.reshape(1, 1, -1)
            self.stds_t[k] = tensor
        print(f"Converted statistics to tensors on device {device}")


    def set_statistics(self, means, stds, state_step_sizes, state_mins, state_maxs, device=None):
        """Store statistics computed from training data and create tensor versions."""
        self.means = means
        self.stds = stds
        self.state_step_sizes = state_step_sizes
        self.state_mins = state_mins
        self.state_maxs = state_maxs
        # Also create tensor versions if device is known
        target_device = device if device is not None else self.device
        if target_device is not None:
            self._stats_to_tensors(target_device)
        else:
            print("Warning: Device not known in set_statistics. Tensor versions not created.")


    def forward(self, o, a, s=None): # s is optional, only needed for init_with_true_state
        """
        Forward pass. Assumes inputs o, a, s are already tensors on the correct device.
          o: observations, shape [B, T, C, H, W] (normalized)
          a: actions, shape [B, T, action_dim] (normalized)
          s: states, shape [B, T, state_dim] (used only if init_with_true_state, normalized)
        """
        if self.means_t is None or self.stds_t is None:
            raise ValueError("Statistics tensors (means_t, stds_t) not initialized.")
        if self.init_with_true_state and s is None:
            raise ValueError("State tensor 's' must be provided when init_with_true_state is True.")

        B, T = o.shape[0], o.shape[1]
        device = o.device # Get device from input tensor

        # --- 1. Encode Observations ---
        # Observations 'o' should already be normalized outside forward if possible,
        # but let's keep normalization here for compatibility with original structure.
        # Ensure o has shape [B, T, C, H, W]
        if o.dim() == 5 and o.shape[-1] == self.obs_channels:
             o = o.permute(0, 1, 4, 2, 3).contiguous()
        if o.dtype != torch.float32: o = o.float()

        # Normalize observations
        norm_o = (o - self.means_t['o']) / self.stds_t['o']

        # Merge batch and time for encoder processing.
        norm_o_flat = norm_o.reshape(B * T, self.obs_channels, self.obs_height, self.obs_width)

        # Pass through convolutional part
        x = self.encoder_conv(norm_o_flat)
        x = self.encoder_flatten(x)
        # Apply dropout and linear layer
        x = self.encoder_dropout(x)
        x = self.encoder_linear(x)
        encodings_flat = self.encoder_relu(x) # [B*T, 128]
        encodings = encodings_flat.view(B, T, -1)  # [B, T, 128]

        # --- 2. Prepare RNN Input ---
        # Normalize actions 'a'
        if a.dtype != torch.float32: a = a.float()
        norm_a = (a - self.means_t['a']) / self.stds_t['a'] # [B, T, action_dim]

        # Augment encoder output if using true state initialization
        if self.init_with_true_state:
            if s.dtype != torch.float32: s = s.float()
            norm_s = (s - self.means_t['s']) / self.stds_t['s']  # [B, T, state_dim]
            # Tracking info: normalized state at t=0, zeros otherwise
            tracking_info = torch.cat([norm_s[:, 0:1, :],
                                       torch.zeros(B, T - 1, self.state_dim, device=device, dtype=s.dtype)], dim=1)
            # Flag: 1 for t=0, 0 otherwise
            flag = torch.cat([torch.ones(B, 1, 1, device=device, dtype=s.dtype),
                              torch.zeros(B, T - 1, 1, device=device, dtype=s.dtype)], dim=1)
            # Concatenate along the feature dimension
            rnn_input = torch.cat([encodings, tracking_info, flag, norm_a], dim=-1)
        else:
            # Concatenate just encodings and normalized actions
            rnn_input = torch.cat([encodings, norm_a], dim=-1) # [B, T, 128 + action_dim]

        # --- 3. Process through Recurrent/FF Layers ---
        if self.model_type in ['2lstm', '2gru']:
            # Pass through RNN layers
            hidden_state = None # Initialize hidden state for sequence
            out1, hidden_state = self.rnn1(rnn_input, hidden_state)
            belief_list, _ = self.rnn2(out1, hidden_state if isinstance(hidden_state, tuple) else (hidden_state, hidden_state) if self.model_type=='2lstm' else hidden_state) # Need to handle GRU vs LSTM state passing if mixing
            # For simplicity, assume hidden state resets per batch sequence
            # out1, _ = self.rnn1(rnn_input) # Let PyTorch handle hidden state initialization
            # belief_list, _ = self.rnn2(out1)
        elif self.model_type == 'ff':
            # Apply FF layers sequentially (equivalent to Sequential)
            belief_list = rnn_input
            for layer in self.rnn_layers:
                 belief_list = layer(belief_list) # [B, T, 512]
        else: # Should not happen due to check in __init__
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # --- 4. Decode Beliefs ---
        belief_flat = belief_list.reshape(B * T, -1)
        pred_flat = self.belief_decoder(belief_flat)  # [B*T, state_dim]
        pred_states_norm = pred_flat.view(B, T, self.state_dim) # Normalized predictions

        # --- 5. Denormalize Predictions ---
        pred_states = pred_states_norm * self.stds_t['s'] + self.means_t['s']

        return pred_states


    def fit(self, data, model_path, split_ratio, seq_len, batch_size, epoch_length,
            num_epochs, patience, learning_rate, dropout_keep_prob, # Changed from ratio
            **unused_kwargs):
        """
        Train the model with GPU support.
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
        print(f"Model moved to device: {device}")

        # --- Update Dropout ---
        # Find the dropout layer and set its probability
        for module in self.modules():
             if isinstance(module, nn.Dropout):
                 module.p = 1.0 - dropout_keep_prob
                 print(f"Set dropout probability to: {module.p:.2f}")
                 # Assuming only one dropout layer intended to be controlled this way

        # --- Preprocess data and compute statistics ---
        print("Splitting data...")
        data_split = split_data(data, ratio=split_ratio)
        print("Computing statistics...")
        # Use the standard compute_statistics function
        means, stds, state_step_sizes, state_mins, state_maxs = compute_statistics(data_split['train'])
        # Store stats and create tensor versions on the correct device
        self.set_statistics(means, stds, state_step_sizes, state_mins, state_maxs, device=device)
        print("Statistics computed and converted to tensors.")

        # --- Create Batch Iterators ---
        epoch_lengths = {'train': epoch_length, 'val': int(epoch_length * 0.2) + 1} # Adjust val length?
        print("Creating batch iterators...")
        # Use repeating iterators for both train and val during epoch-based training
        batch_iterators = {
            'train': make_repeating_batch_iterator(data_split['train'], epoch_lengths['train'], batch_size=batch_size, seq_len=seq_len),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
            # Add non-repeating if needed for final eval/plotting
            # 'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len)
        }
        print("Iterators created.")

        # --- Optimizer and Loss ---
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Use reduction='mean' by default
        mse_loss = nn.MSELoss()

        # --- Logging ---
        log = {'train_loss': [], 'val_loss': [], 'train_loss_last': [], 'val_loss_last': []}
        best_val_loss = float('inf')
        best_epoch = 0
        save_path = os.path.join(model_path, 'best_validation.pth')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # Save statistics used for this training run
        stats_save_path = os.path.join(model_path, 'statistics.npz')
        np.savez(stats_save_path,
                 means=self.means, stds=self.stds, state_step_sizes=self.state_step_sizes,
                 state_mins=self.state_mins, state_maxs=self.state_maxs)
        print(f"Saved statistics to {stats_save_path}")


        # --- Training Loop ---
        for epoch in range(num_epochs):
            epoch_log = {'train_loss': [], 'val_loss': [], 'train_loss_last': [], 'val_loss_last': []}
            start_time = datetime.now()

            # --- Training Phase ---
            self.train() # Set model to training mode (enables dropout)
            iterator = batch_iterators['train']
            num_steps = epoch_lengths['train']
            for step in range(num_steps):
                batch = next(iterator)
                # Move batch data to the device
                batch_device = move_batch_to_device(batch, device)
                o, a, s = batch_device['o'], batch_device['a'], batch_device['s']

                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                pred = self.forward(o, a, s if self.init_with_true_state else None)

                # Calculate loss (on the whole sequence and last step)
                loss_all = mse_loss(pred, s)
                loss_last = mse_loss(pred[:, -1, :], s[:, -1, :])
                # Use loss_all for optimization (as in original)
                loss = loss_all

                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_log['train_loss'].append(loss_all.item())
                epoch_log['train_loss_last'].append(loss_last.item())

            # --- Validation Phase ---
            if 'val' in batch_iterators:
                self.eval() # Set model to evaluation mode (disables dropout)
                iterator = batch_iterators['val']
                num_steps = epoch_lengths['val']
                with torch.no_grad():
                    for step in range(num_steps):
                        batch = next(iterator)
                        batch_device = move_batch_to_device(batch, device)
                        o, a, s = batch_device['o'], batch_device['a'], batch_device['s']

                        pred = self.forward(o, a, s if self.init_with_true_state else None)
                        loss_all = mse_loss(pred, s)
                        loss_last = mse_loss(pred[:, -1, :], s[:, -1, :])

                        epoch_log['val_loss'].append(loss_all.item())
                        epoch_log['val_loss_last'].append(loss_last.item())

            # --- Epoch Summary & Saving ---
            epoch_time = (datetime.now() - start_time).total_seconds()
            avg_train_loss = np.mean(epoch_log['train_loss'])
            avg_train_loss_last = np.mean(epoch_log['train_loss_last'])
            log['train_loss'].append(avg_train_loss)
            log['train_loss_last'].append(avg_train_loss_last)

            if 'val' in batch_iterators:
                avg_val_loss = np.mean(epoch_log['val_loss'])
                avg_val_loss_last = np.mean(epoch_log['val_loss_last'])
                log['val_loss'].append(avg_val_loss)
                log['val_loss_last'].append(avg_val_loss_last)

                current_val_loss = avg_val_loss_last # Use last step MSE for validation metric

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    torch.save(self.state_dict(), save_path)
                    txt_prefix = f"epoch {epoch:3d} >> "
                else:
                    txt_prefix = f"epoch {epoch:3d} == "

                print(f"{txt_prefix}Train Loss: {avg_train_loss:.4f} (Last: {avg_train_loss_last:.4f}) | "
                      f"Val Loss: {avg_val_loss:.4f} (Last: {avg_val_loss_last:.4f}) | "
                      f"Patience: {epoch - best_epoch}/{patience} | Time: {epoch_time:.2f}s")

                if epoch - best_epoch >= patience:
                    print(f"Early stopping triggered after epoch {epoch}.")
                    break
            else: # No validation set
                best_epoch = epoch
                torch.save(self.state_dict(), save_path) # Save every epoch if no validation
                txt_prefix = f"epoch {epoch:3d} >> "
                print(f"{txt_prefix}Train Loss: {avg_train_loss:.4f} (Last: {avg_train_loss_last:.4f}) | Time: {epoch_time:.2f}s")


        print("\n--- Training Finished ---")
        # Restore the best model found during validation
        if os.path.exists(save_path) and 'val' in batch_iterators:
             print(f"Loading best model from epoch {best_epoch} ({save_path})")
             self.load_state_dict(torch.load(save_path, map_location=device))
        elif os.path.exists(save_path):
             print(f"Loading last saved model from {save_path}")
             self.load_state_dict(torch.load(save_path, map_location=device))
        else:
             print("Warning: No model checkpoint found.")

        return log


    def predict(self, batch, **unused_kwargs):
        """
        Predict on a batch using the current device.
        Returns predictions as a NumPy array on CPU.
        """
        self.eval() # Ensure model is in evaluation mode
        if self.device is None:
             raise ValueError("Device not set. Call fit() or load_model() first.")

        # Move batch to device
        batch_device = move_batch_to_device(batch, self.device)
        o, a = batch_device['o'], batch_device['a']
        # 's' might not be present in prediction batch, only needed if init_with_true_state
        s = batch_device.get('s', None)

        with torch.no_grad():
            pred = self.forward(o, a, s if self.init_with_true_state else None)

        return pred.cpu().numpy() # Return predictions as numpy array on CPU


    def load_model(self, model_path, model_file='best_validation.pth',
                   statistics_file='statistics.npz', device='cpu'):
        """
        Load model parameters and statistics, moving model to the specified device.
        """
        # Determine target device
        if isinstance(device, str):
             device = torch.device(device)
        self.device = device # Store the device

        # Load statistics first
        stats_path = os.path.join(model_path, statistics_file)
        if os.path.exists(stats_path):
             stats = np.load(stats_path, allow_pickle=True)
             # Use .item() to extract dictionaries/values from 0-d arrays
             means = stats['means'].item()
             stds = stats['stds'].item()
             state_step_sizes = stats['state_step_sizes']
             state_mins = stats['state_mins']
             state_maxs = stats['state_maxs']
             stats.close()
             # Store numpy versions and create tensor versions on the target device
             self.set_statistics(means, stds, state_step_sizes, state_mins, state_maxs, device=device)
             print(f"Loaded and converted statistics from {stats_path} to device {device}")
        else:
             print(f"Warning: Statistics file not found at {stats_path}. Cannot initialize stats tensors.")

        # Load model state dict
        state_dict_path = os.path.join(model_path, model_file)
        if os.path.exists(state_dict_path):
             state_dict = torch.load(state_dict_path, map_location=device) # Load directly to target device
             self.load_state_dict(state_dict)
             self.to(device) # Ensure model parameters are on the device
             print(f"Loaded model state_dict from {state_dict_path} to device {device}")
        else:
             print(f"Warning: Model state_dict file not found at {state_dict_path}.")

        self.eval() # Set to evaluation mode after loading

