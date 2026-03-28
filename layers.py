# FI-KAN: Layer Implementations
# --------------------------------------------------------
# PureFIKANLinear:   Barnsley framework (Section 3.2)
#   Replaces B-splines entirely with FIF bases.
#   When d = 0: hat functions (order-1 KAN).
#
# HybridFIKANLinear: Navascues framework (Section 3.3)
#   Retains B-spline path (= classical approximant b) and adds
#   fractal correction path (= fractal perturbation h).
#   f_b^alpha = b + h (alpha-fractal decomposition, Eq. 11).
#   When d = 0, w_frac = 0: reduces exactly to standard KAN.
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .bases import fractal_bases, fractal_dim_from_d
except ImportError:
    from bases import fractal_bases, fractal_dim_from_d


# ================================================================
# Pure FI-KAN Linear Layer (Barnsley Framework)
# ================================================================

class PureFIKANLinear(nn.Module):
    """
    Pure Fractal Interpolation KAN Linear Layer.

    Replaces B-spline basis functions entirely with fractal interpolation
    function (FIF) bases, testing the regularity-matching hypothesis in
    its strongest form (Section 3.2).

    Edge function (Definition 3.2):
        phi_{j,i}(x) = w_base * sigma(x)
                      + w_scale * sum_m w_frac_m * phi_m(x; d_i)

    When d = 0, the FIF bases reduce to piecewise linear hat functions
    and the layer becomes an order-1 spline KAN edge.
    """
    def __init__(self, in_features, out_features, grid_size=5,
                 fractal_depth=8, scale_noise=0.1, scale_base=1.0,
                 scale_fractal=1.0,
                 enable_standalone_scale_fractal=True,
                 base_activation=nn.SiLU, grid_range=[-1, 1],
                 d_init_std=0.01, d_max=0.99):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.fractal_depth = fractal_depth
        self.grid_range = grid_range
        self.d_max = d_max

        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.fractal_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + 1))
        if enable_standalone_scale_fractal:
            self.fractal_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features))
        self.enable_standalone_scale_fractal = enable_standalone_scale_fractal
        self.d_raw = nn.Parameter(
            torch.Tensor(in_features, grid_size))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_fractal = scale_fractal
        self.d_init_std = d_init_std
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.out_features, self.in_features,
                                 self.grid_size + 1) - 0.5
                     ) * self.scale_noise / self.grid_size
            self.fractal_weight.data.copy_(
                (self.scale_fractal
                 if not self.enable_standalone_scale_fractal
                 else 1.0) * noise)
            if self.enable_standalone_scale_fractal:
                nn.init.kaiming_uniform_(
                    self.fractal_scaler,
                    a=math.sqrt(5) * self.scale_fractal)
            self.d_raw.data.normal_(0, self.d_init_std)

    @property
    def d(self):
        """Contraction parameters via tanh reparameterization (Section 3.1)."""
        return self.d_max * torch.tanh(self.d_raw)

    @property
    def scaled_fractal_weight(self):
        return self.fractal_weight * (
            self.fractal_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_fractal else 1.0)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(
            self.base_activation(x), self.base_weight)
        bases = fractal_bases(
            x, self.d, self.grid_size, self.fractal_depth,
            self.grid_range)
        fractal_output = F.linear(
            bases.view(x.size(0), -1),
            self.scaled_fractal_weight.view(self.out_features, -1))
        return base_output + fractal_output

    def fractal_dimension(self):
        """Learned box-counting dimension (Theorem 2.3)."""
        return fractal_dim_from_d(self.d, self.grid_size)

    def regularization_loss(self, reg_act=1.0, reg_ent=1.0,
                            reg_frac=0.0):
        """
        Regularization combining activation L1, entropy,
        and fractal dimension penalty (Definition 3.4).
        """
        l1 = self.fractal_weight.abs().mean(-1)
        ra = l1.sum()
        p = l1 / (ra + 1e-10)
        re = -torch.sum(p * (p + 1e-10).log())
        loss = reg_act * ra + reg_ent * re
        if reg_frac > 0:
            dims = self.fractal_dimension()
            loss = loss + reg_frac * ((dims - 1.0) ** 2).sum()
        return loss


# ================================================================
# Hybrid FI-KAN Linear Layer (Navascues Framework)
# ================================================================

class HybridFIKANLinear(nn.Module):
    """
    Hybrid Spline-Fractal KAN Linear Layer.

    Implements the alpha-fractal decomposition (Navascues, 2005)
    within the KAN architecture (Section 3.3):

        phi_{j,i}(x) = w_base * sigma(x)         [base path]
                      + w_s.sc * sum w_spl B_m(x)  [spline path = b]
                      + sum w_frac phi_m(x; d)      [fractal correction = h]

    The spline path is the classical approximant b, and the fractal
    path provides the correction h, so f_b^alpha = b + h (Eq. 15).

    When d = 0 and w_frac = 0: identical to efficient-KAN.
    """
    def __init__(self, in_features, out_features, grid_size=5,
                 spline_order=3, fractal_depth=8,
                 scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, scale_fractal=0.1,
                 enable_standalone_scale_spline=True,
                 base_activation=nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1], d_init_std=0.01, d_max=0.99):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.fractal_depth = fractal_depth
        self.grid_range = grid_range
        self.d_max = d_max

        # B-Spline grid (same as efficient-KAN)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(
            -spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        # Parameters
        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features,
                         grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features))
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.fractal_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + 1))
        self.d_raw = nn.Parameter(
            torch.Tensor(in_features, grid_size))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_fractal = scale_fractal
        self.d_init_std = d_init_std
        self.grid_eps = grid_eps
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(
                self.grid_size + 1, self.in_features,
                self.out_features) - 0.5)
                * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline
                 if not self.enable_standalone_scale_spline
                 else 1.0)
                * self._curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order],
                    noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline)
            # Fractal path starts at zero (KAN recovery, Section 3.3)
            self.fractal_weight.data.zero_()
            self.d_raw.data.normal_(0, self.d_init_std)

    def b_splines(self, x):
        """Evaluate B-spline basis functions (same as efficient-KAN)."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k+1)])
                / (grid[:, k:-1] - grid[:, :-(k+1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k+1:] - x)
                / (grid[:, k+1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)

    @property
    def d(self):
        """Contraction parameters via tanh reparameterization (Section 3.1)."""
        return self.d_max * torch.tanh(self.d_raw)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        # Base path
        base_output = F.linear(
            self.base_activation(x), self.base_weight)
        # Spline path (= classical approximant b)
        spline_b = self.b_splines(x)
        spline_output = F.linear(
            spline_b.view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1))
        # Fractal correction path (= perturbation h)
        frac_b = fractal_bases(
            x, self.d, self.grid_size, self.fractal_depth,
            self.grid_range)
        fractal_output = F.linear(
            frac_b.view(x.size(0), -1),
            self.fractal_weight.view(self.out_features, -1))
        return base_output + spline_output + fractal_output

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        """Adaptive grid update (inherited from efficient-KAN)."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(
            0, batch-1, self.grid_size+1,
            dtype=torch.int64, device=x.device)]
        uniform_step = (
            x_sorted[-1] - x_sorted[0] + 2*margin) / self.grid_size
        grid_uniform = (torch.arange(
            self.grid_size+1, dtype=torch.float32,
            device=x.device).unsqueeze(1) * uniform_step
            + x_sorted[0] - margin)
        grid = (self.grid_eps * grid_uniform
                + (1 - self.grid_eps) * grid_adaptive)
        grid = torch.concatenate([
            grid[:1] - uniform_step * torch.arange(
                self.spline_order, 0, -1,
                device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + uniform_step * torch.arange(
                1, self.spline_order+1,
                device=x.device).unsqueeze(1),
        ], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(
            self._curve2coeff(x, unreduced))

    def fractal_dimension(self):
        """Learned box-counting dimension (Theorem 2.3)."""
        return fractal_dim_from_d(self.d, self.grid_size)

    def fractal_energy_ratio(self):
        """
        Diagnostic: rho = ||w_frac||_1 / ||w_spl||_1 (Eq. 16).
        rho ~ 0: spline-dominated. rho >> 0: fractal active.
        """
        with torch.no_grad():
            fn = self.fractal_weight.abs().mean()
            sn = self.spline_weight.abs().mean()
            return (fn / (sn + 1e-10)).item()

    def regularization_loss(self, reg_act=1.0, reg_ent=1.0,
                            reg_frac=0.0):
        """
        Combined regularization:
          - Activation L1 (on spline + fractal weights)
          - Entropy regularization (on spline weights)
          - Fractal dimension penalty R(d) = (dimB(d) - 1)^2
            (Definition 3.4, geometry-aware Occam's razor)
        """
        l1_s = self.spline_weight.abs().mean(-1)
        ra_s = l1_s.sum()
        p = l1_s / (ra_s + 1e-10)
        re_s = -torch.sum(p * (p + 1e-10).log())
        l1_f = self.fractal_weight.abs().mean(-1).sum()
        loss = reg_act * (ra_s + l1_f) + reg_ent * re_s
        if reg_frac > 0:
            dims = self.fractal_dimension()
            loss = loss + reg_frac * ((dims - 1.0) ** 2).sum()
        return loss
