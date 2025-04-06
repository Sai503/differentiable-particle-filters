import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.data_utils import split_data, make_batch_iterator, make_repeating_batch_iterator#, plot_maze, show_pause
from utils.method_utils import compute_sq_distance

class RNN(nn.Module):
    def __init__(self, init_with_true_state=False, model='2lstm', **unused_kwargs):
        """
        PyTorch implementation of an RNN model.
        - init_with_true_state: if True, encoder output is augmented with tracking info.
        - model: one of '2lstm', '2gru', or 'ff'.
        """
        super(RNN, self).__init__()
        self.init_with_true_state = init_with_true_state
        self.model = model

        # --- Encoder for observations ---
        # Expected input: images of size 24x24 with 3 channels (assumed shape [B, 3, 24, 24])
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 6 -> 3
            nn.ReLU(),
            nn.Flatten(),
            # Dropout probability will be adjusted in forward() using keep_prob.
            nn.Dropout(p=0.0),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU()
        )

        # --- Recurrent network ---
        # Compute input size for the RNN from the encoder output plus action vector.
        # If using true state initialization, we also concatenate tracking info (3 dims)
        # and a flag (1 dim) so extra = 4.
        extra = 4 if self.init_with_true_state else 0
        rnn_input_size = 128 + extra + 3  # 128 from encoder, 3 from action, extra if applicable

        if self.model == '2lstm':
            self.rnn1 = nn.LSTM(input_size=rnn_input_size, hidden_size=512, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        elif self.model == '2gru':
            self.rnn1 = nn.GRU(input_size=rnn_input_size, hidden_size=512, batch_first=True)
            self.rnn2 = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        elif self.model == 'ff':
            self.ff_lstm_replacement = nn.Sequential(
                nn.Linear(rnn_input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
            )
        else:
            raise ValueError("Unsupported model type")

        # --- Belief decoder ---
        self.belief_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        # --- Statistics placeholders ---
        # These will be set via set_statistics() from training data.
        self.means = None
        self.stds = None
        self.state_step_sizes = None
        self.state_mins = None
        self.state_maxs = None

    def set_statistics(self, means, stds, state_step_sizes, state_mins, state_maxs):
        """Store statistics computed from training data."""
        self.means = means
        self.stds = stds
        self.state_step_sizes = state_step_sizes
        self.state_mins = state_mins
        self.state_maxs = state_maxs

    def forward(self, o, a, s, keep_prob=1.0):
        """
        Forward pass.
          o: observations, shape [B, T, 24, 24, 3] or [B, T, 3, 24, 24]
          a: actions, shape [B, T, 3]
          s: states, shape [B, T, 3]
          keep_prob: dropout keep probability (applied in encoder)
        """
        B, T = o.shape[0], o.shape[1]
        # Ensure observations are in [B, T, 3, 24, 24]
        if o.dim() == 5 and o.shape[-1] == 3:
            o = o.permute(0, 1, 4, 2, 3)

        # Normalize observations: (o - means['o']) / stds['o']
        o = (o - self.means['o']) / self.stds['o']
        # Merge batch and time for encoder processing.
        o_flat = o.contiguous().view(B * T, o.size(2), o.size(3), o.size(4))
        # Adjust dropout probability in the encoder.
        for m in self.encoder.modules():
            if isinstance(m, nn.Dropout):
                m.p = 1 - keep_prob
        x = self.encoder(o_flat)  # [B*T, 128]
        encodings = x.view(B, T, -1)  # [B, T, 128]

        # If true state initialization is used, augment encoder output with tracking info.
        if self.init_with_true_state:
            # tracking_info: normalized state at t=0 (first time step) and zeros for later timesteps.
            norm_s = (s - self.means['s']) / self.stds['s']  # [B, T, 3]
            tracking_info = torch.cat([norm_s[:, :1, :], torch.zeros(B, T - 1, 3, device=s.device, dtype=s.dtype)], dim=1)
            # flag: 1 for first time step, 0 for the rest.
            flag = torch.cat([torch.ones(B, 1, 1, device=s.device, dtype=s.dtype),
                              torch.zeros(B, T - 1, 1, device=s.device, dtype=s.dtype)], dim=1)
            # Concatenate along the feature dimension.
            encodings = torch.cat([encodings, tracking_info, flag], dim=-1)  # now dims = 128 + 3 + 1

        # Normalize actions: a / stds['a']
        a_norm = a / self.stds['a']  # [B, T, 3]

        # Concatenate encoder output and actions.
        preproc_ao = torch.cat([encodings, a_norm], dim=-1)  # [B, T, (128+extra) + 3]

        # Process through recurrent layers.
        if self.model in ['2lstm', '2gru']:
            out1, _ = self.rnn1(preproc_ao)  # [B, T, 512]
            out2, _ = self.rnn2(out1)         # [B, T, 512]
            belief_list = out2
        elif self.model == 'ff':
            belief_list = self.ff_lstm_replacement(preproc_ao)
        else:
            raise ValueError("Unsupported model type")

        # Decode beliefs at each time step.
        B, T, _ = belief_list.shape
        belief_flat = belief_list.contiguous().view(B * T, -1)
        pred_flat = self.belief_decoder(belief_flat)  # [B*T, 3]
        pred_states = pred_flat.view(B, T, 3)
        # Rescale predictions: pred_states * stds['s'] + means['s']
        pred_states = pred_states * self.stds['s'] + self.means['s']
        return pred_states

    def fit(self, data, model_path, split_ratio, seq_len, batch_size, epoch_length,
            num_epochs, patience, learning_rate, dropout_keep_ratio, **unused_kwargs):
        """
        Train the model.
          data: training data dictionary.
          model_path: directory to save the model.
          Other parameters define the training procedure.
        """
        # Preprocess data and compute statistics.
        data_split = split_data(data, ratio=split_ratio)
        epoch_lengths = {'train': epoch_length, 'val': epoch_length * 2}
        batch_iterators = {
            'train': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=seq_len),
            'val': make_repeating_batch_iterator(data_split['val'], epoch_lengths['val'],
                                                 batch_size=batch_size, seq_len=seq_len),
            'train_ex': make_batch_iterator(data_split['train'], batch_size=batch_size, seq_len=seq_len),
            'val_ex': make_batch_iterator(data_split['val'], batch_size=batch_size, seq_len=seq_len)
        }
        means, stds, state_step_sizes, state_mins, state_maxs = compute_sq_distance.__globals__['compute_staticstics'](data_split['train'])
        self.set_statistics(means, stds, state_step_sizes, state_mins, state_maxs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        mse_loss = nn.MSELoss()

        # Set up logging.
        loss_keys = ['mse_last', 'mse']
        data_keys = ['train', 'val'] if split_ratio < 1.0 else ['train']
        log = {dk: {lk: {'mean': [], 'se': []} for lk in loss_keys} for dk in data_keys}

        best_val_loss = np.inf
        best_epoch = 0
        i = 0
        save_path = os.path.join(model_path, 'best_validation.pth')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        while i < num_epochs and i - best_epoch < patience:
            loss_lists = {dk: {lk: [] for lk in loss_keys} for dk in data_keys}
            for dk in data_keys:
                for _ in range(epoch_lengths[dk]):
                    batch = next(batch_iterators[dk])
                    # Assume batch is a dict with keys: 'o', 'a', 's' (numpy arrays).
                    o = torch.tensor(batch['o'], dtype=torch.float32, device=device)
                    a = torch.tensor(batch['a'], dtype=torch.float32, device=device)
                    s = torch.tensor(batch['s'], dtype=torch.float32, device=device)
                    if dk == 'train':
                        self.train()
                        optimizer.zero_grad()
                        pred = self.forward(o, a, s, keep_prob=dropout_keep_ratio)
                        loss_all = mse_loss(pred, s)
                        loss_last = mse_loss(pred[:, -1, :], s[:, -1, :])
                        loss = loss_all
                        loss.backward()
                        optimizer.step()
                        l_mse = loss_all.item()
                        l_mse_last = loss_last.item()
                    else:
                        self.eval()
                        with torch.no_grad():
                            pred = self.forward(o, a, s, keep_prob=1.0)
                            loss_all = mse_loss(pred, s)
                            loss_last = mse_loss(pred[:, -1, :], s[:, -1, :])
                        l_mse = loss_all.item()
                        l_mse_last = loss_last.item()
                    loss_lists[dk]['mse'].append(l_mse)
                    loss_lists[dk]['mse_last'].append(l_mse_last)
            # Log average losses.
            for dk in data_keys:
                for lk in loss_keys:
                    log[dk][lk]['mean'].append(np.mean(loss_lists[dk][lk]))
                    log[dk][lk]['se'].append(np.std(loss_lists[dk][lk], ddof=1) / np.sqrt(epoch_lengths[dk]))
            # Check validation performance.
            if 'val' in data_keys:
                current_val_loss = log['val']['mse_last']['mean'][-1]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = i
                    torch.save(self.state_dict(), save_path)
                    txt = f'epoch {i:3d} >> '
                else:
                    txt = f'epoch {i:3d} == '
            else:
                best_epoch = i
                torch.save(self.state_dict(), save_path)
                txt = f'epoch {i:3d} >> '
            for lk in loss_keys:
                txt += f'{lk}: '
                for dk in data_keys:
                    txt += f'{log[dk][lk]["mean"][-1]:.2f}+-{log[dk][lk]["se"][-1]:.2f}/'
                txt = txt[:-1] + ' -- '
            print(txt)
            i += 1

        # Restore the best model.
        self.load_state_dict(torch.load(save_path))
        return log

    def predict(self, batch, **unused_kwargs):
        """
        Predict on a batch.
          batch: dict with keys 'o', 'a', 's'
        Returns predictions as a NumPy array.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            o = torch.tensor(batch['o'], dtype=torch.float32, device=device)
            a = torch.tensor(batch['a'], dtype=torch.float32, device=device)
            s = torch.tensor(batch['s'], dtype=torch.float32, device=device)
            pred = self.forward(o, a, s, keep_prob=1.0)
        return pred.cpu().numpy()

    def load_model(self, model_path, model_file='best_validation.pth', statistics_file='statistics.npz', connect_and_initialize=True):
        """
        Load model parameters and, if desired, statistics.
        """
        if connect_and_initialize:
            stats = np.load(os.path.join(model_path, statistics_file), allow_pickle=True)
            stats = dict(stats)
            for key in stats.keys():
                if stats[key].shape == ():
                    stats[key] = stats[key].item()
            self.set_statistics(stats['means'], stats['stds'], stats['state_step_sizes'],
                                stats['state_mins'], stats['state_maxs'])
        self.load_state_dict(torch.load(os.path.join(model_path, model_file)))
        print("Loaded model from", os.path.join(model_path, model_file))

