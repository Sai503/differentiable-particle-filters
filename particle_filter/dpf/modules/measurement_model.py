import torch
import torch.nn as nn

class MeasurementModel(nn.Module):
    def __init__(self, state_dim, min_obs_likelihood):
        super(MeasurementModel, self).__init__()
        self.min_obs_likelihood = min_obs_likelihood
        self.encoding_dim = 256  # Assuming the encoding dimension is 128
        self.estimator = nn.Sequential(
            nn.Linear(self.encoding_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, encoding, particles, means_t, stds_t):
        """
        Computes the observation likelihood for each particle.
        encoding: [B, 128] (on device)
        particles: [B, N, 3] (on device)
        means_t, stds_t: dictionaries containing tensors for normalization.
        """
        if means_t is None or stds_t is None:
            raise ValueError("Statistics (means_t, stds_t) not initialized or not on device.")

        # Normalize particle poses and augment with cosine and sine of orientation
        means_xy = means_t['s'][:, :, :2]
        stds_xy = stds_t['s'][:, :, :2]
        particles_xy = particles[:, :, :2]
        norm_pos = (particles_xy - means_xy) / (stds_xy + 1e-8)
        cos_theta = torch.cos(particles[:, :, 2:3])
        sin_theta = torch.sin(particles[:, :, 2:3])
        particle_input = torch.cat([norm_pos, cos_theta, sin_theta], dim=-1)  # [B, N, 4]

        # Tile encoding to match the number of particles
        encoding_input = encoding.unsqueeze(1).expand(-1, particles.size(1), -1)  # [B, N, 256]
        inp = torch.cat([encoding_input, particle_input], dim=-1)  # [B, N, 260]

        # Flatten input for the estimator
        B, N, _ = inp.shape
        inp_flat = inp.view(B * N, -1)
        out = self.estimator(inp_flat).view(B, N)

        # Scale the output between min_obs_likelihood and 1
        return out * (1 - self.min_obs_likelihood) + self.min_obs_likelihood


"""
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

"""