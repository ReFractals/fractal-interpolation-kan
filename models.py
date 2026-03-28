# FI-KAN: Network-Level Models
# --------------------------------------------------------
# PureFIKAN:   Multi-layer Pure FI-KAN (Barnsley framework)
# HybridFIKAN: Multi-layer Hybrid FI-KAN (Navascues framework)
#
# Both follow the KAN convention: layers_hidden specifies the
# width of each layer, e.g. [1, 16, 1] for a two-layer network
# with 16 hidden units.
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

import torch.nn as nn

try:
    from .layers import PureFIKANLinear, HybridFIKANLinear
except ImportError:
    from layers import PureFIKANLinear, HybridFIKANLinear


class PureFIKAN(nn.Module):
    """
    Pure Fractal Interpolation KAN Network (Barnsley framework).

    All edges carry fractal interpolation function (FIF) bases
    with learnable contraction parameters. No B-spline path.

    Recommended for targets known to be fractal or when
    computational overhead must be minimized.
    """
    def __init__(self, layers_hidden, grid_size=5, fractal_depth=8,
                 scale_noise=0.1, scale_base=1.0, scale_fractal=1.0,
                 base_activation=nn.SiLU, grid_range=[-1, 1],
                 d_init_std=0.01, d_max=0.99):
        super().__init__()
        self.grid_size = grid_size
        self.fractal_depth = fractal_depth
        self.layers = nn.ModuleList()
        for inf, outf in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(PureFIKANLinear(
                inf, outf, grid_size=grid_size,
                fractal_depth=fractal_depth,
                scale_noise=scale_noise, scale_base=scale_base,
                scale_fractal=scale_fractal,
                base_activation=base_activation,
                grid_range=grid_range, d_init_std=d_init_std,
                d_max=d_max))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, ra=1.0, re=1.0, rf=0.0):
        return sum(
            l.regularization_loss(ra, re, rf) for l in self.layers)

    def fractal_dimensions(self):
        """Return learned fractal dimensions for all layers."""
        return [l.fractal_dimension() for l in self.layers]


class HybridFIKAN(nn.Module):
    """
    Hybrid Spline-Fractal KAN Network (Navascues framework).

    Each edge carries dual paths: B-spline (classical approximant b)
    and fractal correction (perturbation h), implementing the
    alpha-fractal decomposition f_b^alpha = b + h.

    When d = 0 and w_frac = 0: reduces exactly to standard KAN.

    Recommended as the default FI-KAN variant. Use K = 2 for
    optimal performance (Section 5.9).
    """
    def __init__(self, layers_hidden, grid_size=5, spline_order=3,
                 fractal_depth=8, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, scale_fractal=0.1,
                 base_activation=nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1], d_init_std=0.01, d_max=0.99):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.fractal_depth = fractal_depth
        self.layers = nn.ModuleList()
        for inf, outf in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(HybridFIKANLinear(
                inf, outf, grid_size=grid_size,
                spline_order=spline_order,
                fractal_depth=fractal_depth,
                scale_noise=scale_noise, scale_base=scale_base,
                scale_spline=scale_spline,
                scale_fractal=scale_fractal,
                base_activation=base_activation,
                grid_eps=grid_eps, grid_range=grid_range,
                d_init_std=d_init_std, d_max=d_max))

    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, ra=1.0, re=1.0, rf=0.0):
        return sum(l.regularization_loss(ra, re, rf)
                   for l in self.layers)

    def fractal_dimensions(self):
        """Return learned fractal dimensions for all layers."""
        return [l.fractal_dimension() for l in self.layers]

    def fractal_energy_ratios(self):
        """Return fractal energy ratios rho for all layers (Eq. 16)."""
        return [l.fractal_energy_ratio() for l in self.layers]
