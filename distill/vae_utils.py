"""
Standalone VAE utilities for the Orion student planning decoder.

Extracted from Orion's mmcv/models/utils/distributions.py with all mmcv
dependencies removed — only standard PyTorch is required.
"""

import torch
import torch.nn as nn


class DistributionEncoder1DV2(nn.Module):
    """1-D convolutional encoder for the VAE distribution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels * 2, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s_t):
        s_t = self.relu(self.conv1(s_t))
        s_t = self.relu(self.conv2(s_t))
        s_t = self.conv3(s_t)
        return s_t


class DistributionModule(nn.Module):
    """Parametrises a diagonal Gaussian distribution from a 1-D feature sequence.

    Args:
        in_channels (int): Number of input channels (feature dimension).
        latent_dim (int): Dimension of the latent variable z.
        min_log_sigma (float): Lower clamp for log-sigma (numerical stability).
        max_log_sigma (float): Upper clamp for log-sigma.
    """

    def __init__(self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.encoder = DistributionEncoder1DV2(in_channels, self.compress_dim)
        self.last_conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1),
        )

    def forward(self, s_t):
        """
        Args:
            s_t (Tensor): shape (B, T, C)
        Returns:
            mu (Tensor): shape (B, T, latent_dim)
            log_sigma (Tensor): shape (B, T, latent_dim), clamped to [min, max]
        """
        encoding = self.encoder(s_t.permute(0, 2, 1).float())   # (B, C, T) → encoder → (B, compress_dim, T)
        mu_log_sigma = self.last_conv(encoding).permute(0, 2, 1)  # (B, T, 2*latent_dim)
        mu = mu_log_sigma[:, :, :self.latent_dim]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:]
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma


class PredictModel(nn.Module):
    """GRU-based future state predictor.

    Args:
        in_channels (int): Input size for GRU (latent_dim).
        out_channels (int): Output size (present_distribution_in_channels).
        hidden_channels (int): GRU hidden size.
        num_layers (int): Number of GRU layers.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.linear2 = nn.Linear(hidden_channels * 2, hidden_channels * 4)
        self.linear3 = nn.Linear(hidden_channels * 4, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, h):
        x, h = self.gru(x, h)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
