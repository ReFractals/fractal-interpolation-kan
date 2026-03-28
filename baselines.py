# FI-KAN: Baseline Architectures
# --------------------------------------------------------
# KANLinear / KAN: Efficient-KAN implementation.
#   Verbatim from https://github.com/Blealtan/efficient-kan
#   Author: Blealtan | License: MIT
#
# MLP: Standard multi-layer perceptron with SiLU activation
#   (same as KAN's base activation) for fair comparison.
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Efficient-KAN Baseline
# Verbatim from https://github.com/Blealtan/efficient-kan
# Author: Blealtan | License: MIT
# ================================================================

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h
                 + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1,
                      self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline
                 if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)])
                / (grid[:, k:-1] - grid[:, :-(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x, y):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1,
                           dtype=torch.int64, device=x.device)]
        uniform_step = (
            x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32,
                         device=x.device).unsqueeze(1)
            * uniform_step + x_sorted[0] - margin)
        grid = (self.grid_eps * grid_uniform
                + (1 - self.grid_eps) * grid_adaptive)
        grid = torch.concatenate([
            grid[:1] - uniform_step * torch.arange(
                self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + uniform_step * torch.arange(
                1, self.spline_order + 1, device=x.device).unsqueeze(1),
        ], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(
            self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0,
                            regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(
                in_f, out_f, grid_size=grid_size,
                spline_order=spline_order, scale_noise=scale_noise,
                scale_base=scale_base, scale_spline=scale_spline,
                base_activation=base_activation, grid_eps=grid_eps,
                grid_range=grid_range))

    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, ra=1.0, re=1.0):
        return sum(l.regularization_loss(ra, re) for l in self.layers)


# ================================================================
# MLP Baseline
# ================================================================

class MLP(nn.Module):
    """
    Standard MLP baseline.

    For fair comparison with KAN/FI-KAN, we match the architecture
    depth and use SiLU activation (same as KAN's base path).
    Width is adjusted to match parameter count where specified.
    """
    def __init__(self, layers_hidden, activation=nn.SiLU):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i, (inf, outf) in enumerate(
                zip(layers_hidden, layers_hidden[1:])):
            self.layers.append(nn.Linear(inf, outf))
            if i < len(layers_hidden) - 2:
                self.acts.append(activation())

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.acts):
                x = self.acts[i](x)
        return x

    def regularization_loss(self, *args, **kwargs):
        """Compatibility stub: L2 weight decay on all parameters."""
        return 0.01 * sum(p.pow(2).sum() for p in self.parameters())


def mlp_matching_params(target_params, layers_hidden, activation=nn.SiLU):
    """
    Create an MLP with approximately target_params parameters
    by adjusting the hidden width.

    Args:
        target_params: desired parameter count
        layers_hidden: [in_dim, ..., out_dim] template
                       (hidden dims will be replaced)
        activation: activation function class

    Returns:
        MLP instance, actual parameter count
    """
    in_dim = layers_hidden[0]
    out_dim = layers_hidden[-1]
    n_hidden = len(layers_hidden) - 2

    if n_hidden == 0:
        return MLP([in_dim, out_dim], activation), in_dim * out_dim + out_dim

    best_h = 1
    best_diff = float('inf')
    for h in range(1, 2048):
        arch = [in_dim] + [h] * n_hidden + [out_dim]
        mlp = MLP(arch, activation)
        p = sum(pp.numel() for pp in mlp.parameters())
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
        if p > target_params * 2:
            break

    arch = [in_dim] + [best_h] * n_hidden + [out_dim]
    mlp = MLP(arch, activation)
    actual = sum(pp.numel() for pp in mlp.parameters())
    return mlp, actual
