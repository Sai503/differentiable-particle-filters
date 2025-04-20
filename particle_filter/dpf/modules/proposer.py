import torch
import torch.nn as nn

class Proposer(nn.Module):
    def __init__(self, state_dim, proposer_keep_ratio):
        super(Proposer, self).__init__()
        self.encoding_dim = 256  # Assuming the encoding dimension is 256
        self.network = nn.Sequential(
            nn.Linear(self.encoding_dim, 128),
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

    def forward(self, encoding, num_particles, state_mins_t, state_maxs_t):
        """
        Proposes new particles from an image encoding using a learned proposer network.

        Args:
            encoding (torch.Tensor): Image encodings, shape [B, encoding_dim] (on device).
            num_particles (int): Number of particles to propose per batch item.
            state_mins_t (torch.Tensor): Minimum state bounds, shape [state_dim] (on device).
            state_maxs_t (torch.Tensor): Maximum state bounds, shape [state_dim] (on device).

        Returns:
            torch.Tensor: Proposed particle states, shape [B, num_particles, state_dim] (on device).
        """
        B = encoding.size(0)
        device = encoding.device
        dtype = encoding.dtype

        # Handle empty batch or zero particles
        if B == 0 or num_particles == 0:
            return torch.empty((B, num_particles, self.state_dim), dtype=dtype, device=device)

        # Duplicate encoding for each particle
        encoding_dup = encoding.unsqueeze(1).expand(-1, num_particles, -1)  # [B, num_particles, 256]

        # Apply proposer network
        inp_flat = encoding_dup.reshape(B * num_particles, -1)  # [B * num_particles, 256]
        proposed_raw = self.network(inp_flat).view(B, num_particles, 4)  # [B, num_particles, 4]

        # Transform the outputs to valid state values
        range_x = state_maxs_t[0] - state_mins_t[0]
        mid_x = (state_maxs_t[0] + state_mins_t[0]) / 2.0
        part0 = proposed_raw[:, :, 0:1] * (range_x / 2.0) + mid_x

        range_y = state_maxs_t[1] - state_mins_t[1]
        mid_y = (state_maxs_t[1] + state_mins_t[1]) / 2.0
        part1 = proposed_raw[:, :, 1:2] * (range_y / 2.0) + mid_y

        part2 = torch.atan2(proposed_raw[:, :, 2:3], proposed_raw[:, :, 3:4])  # Compute angle (theta)

        # Concatenate results
        proposed_particles = torch.cat([part0, part1, part2], dim=-1)  # [B, num_particles, state_dim]

        return proposed_particles


"""
old proposer code
   
        # Proposes new particles from an image encoding using a learned proposer network.

        # Args:
        #     encoding (torch.Tensor): Image encodings, shape [B, encoding_dim] (on device).
        #     num_particles (int): Number of particles to propose per batch item.
        # Uses self.state_mins_t and self.state_maxs_t which should be tensors on the device.

        # Returns:
        #     torch.Tensor: Proposed particle states, shape [B, num_particles, 3] (on device).
        
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

"""