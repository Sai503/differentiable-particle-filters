import torch
import torch.nn as nn
from particle_filter.utils.data_utils import wrap_angle

class MotionModel(nn.Module):
    def __init__(self, state_dim, learn_odom):
        super(MotionModel, self).__init__()
        self.state_dim = state_dim
        self.learn_odom = learn_odom
        self.noise_generator = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        if learn_odom:
            self.transition_model = nn.Sequential(
                nn.Linear(7, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim)
            )

    def forward(self, actions, particles, stds_t, means_t, state_step_sizes_t, stop_sampling_gradient=False):
        """
        Moves particles according to actions with added learned noise.
        actions: [B, 3] (on device)
        particles: [B, N, 3] (on device)
        stds_t, means_t, state_step_sizes_t: tensors for normalization and scaling.
        """
        if stds_t is None or means_t is None or state_step_sizes_t is None:
            raise ValueError("Statistics (stds_t, means_t, state_step_sizes_t) not initialized or not on device.")

        # Expand actions to match particles
        actions_exp = actions.unsqueeze(1)  # [B, 1, 3]
        std_a = stds_t['a']  # [1, 1, 3]
        action_input = actions_exp / (std_a + 1e-8)  # Normalize actions
        B, N, _ = particles.shape
        action_input = action_input.expand(-1, N, -1)

        # Add random noise
        random_input = torch.randn_like(action_input)  # [B, N, 3]
        inp = torch.cat([action_input, random_input], dim=-1)  # [B, N, 6]
        inp_flat = inp.view(B * N, -1)
        delta = self.noise_generator(inp_flat).view(B, N, -1)

        if stop_sampling_gradient:
            delta = delta.detach()

        # Center the noise delta
        delta = delta - delta.mean(dim=1, keepdim=True)
        noisy_actions = actions_exp + delta  # [B, N, 3]

        if self.learn_odom:
            # Use learned odometry model
            state_input = self._transform_particles_as_input(particles, means_t, stds_t)  # [B, N, 4]
            action_input_odom = noisy_actions / (std_a + 1e-8)
            inp2 = torch.cat([state_input, action_input_odom], dim=-1)  # [B, N, 7]
            inp2_flat = inp2.view(B * N, -1)
            state_delta = self.transition_model(inp2_flat).view(B, N, -1)

            # Scale state_delta using state_step_sizes_t
            state_delta = state_delta * state_step_sizes_t.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
            new_states = particles + state_delta
            new_states[:, :, 2:3] = wrap_angle(new_states[:, :, 2:3])  # Wrap orientation
            return new_states
        else:
            # Standard odometry model
            theta = particles[:, :, 2:3]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            delta_x = noisy_actions[:, :, 0:1] * cos_theta - noisy_actions[:, :, 1:2] * sin_theta
            delta_y = noisy_actions[:, :, 0:1] * sin_theta + noisy_actions[:, :, 1:2] * cos_theta
            new_x = particles[:, :, 0:1] + delta_x
            new_y = particles[:, :, 1:2] + delta_y
            new_theta = wrap_angle(particles[:, :, 2:3] + noisy_actions[:, :, 2:3])
            return torch.cat([new_x, new_y, new_theta], dim=-1)

    def _transform_particles_as_input(self, particles, means_t, stds_t):
        """
        Normalizes particle poses and augments with cosine and sine of orientation.
        particles: [B, N, 3] (on device)
        """
        means_xy = means_t['s'][:, :, :2]
        stds_xy = stds_t['s'][:, :, :2]
        particles_xy = particles[:, :, :2]
        norm_pos = (particles_xy - means_xy) / (stds_xy + 1e-8)
        cos_theta = torch.cos(particles[:, :, 2:3])
        sin_theta = torch.sin(particles[:, :, 2:3])
        return torch.cat([norm_pos, cos_theta, sin_theta], dim=-1)




"""
old motion model code
# Moves particles according to actions with added learned noise.
# actions: [B, 3] (on device)Å
# particles: [B, N, 3] (on device)
# Uses self.stds_t, self.means_t, self.state_step_sizes_t (tensors on device).
# 
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
"""