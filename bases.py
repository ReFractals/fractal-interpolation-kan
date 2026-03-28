# FI-KAN: Fractal Interpolation Basis Functions
# --------------------------------------------------------
# Implements Algorithm 1 from the paper: truncated Read-Bajraktarevic
# iteration for computing fractal interpolation function (FIF) bases.
#
# The FIF basis functions {phi_i(x; d)}_{i=0}^N satisfy:
#   (i)   Kronecker property: phi_j(x_i; d) = delta_{ij}
#   (ii)  Partition of unity: sum_j phi_j(x; d) = 1
#   (iii) Degeneration: when d = 0, phi_j reduces to the standard
#         piecewise linear hat function centered at x_j
#
# Reference: Barnsley (1986), "Fractal functions and interpolation",
#            Constructive Approximation, 2(1):303-329.
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

import math
import torch
import torch.nn.functional as F


def fractal_bases(x, d, grid_size, depth, grid_range):
    """
    Compute fractal interpolation basis function values via truncated
    Read-Bajraktarevic (RB) iteration (Algorithm 1 in the paper).

    Mathematical guarantee (Theorem 4.1):
        f*(x; y, d) = sum_i y_i * phi_i(x; d)
    where phi_i are fractal basis functions satisfying
    phi_i(x_j) = delta_{ij} (Kronecker property).

    Truncation error (Proposition 3.1):
        |f^{(K)}(x) - f*(x)| <= d_max^K / (1 - d_max)

    Args:
        x: (batch, in_features), values in grid_range
        d: (in_features, grid_size), contraction parameters with |d_j| < 1
        grid_size: N, number of intervals
        depth: K, recursion depth
        grid_range: [a, b], domain endpoints

    Returns:
        bases: (batch, in_features, grid_size + 1), basis function values
    """
    batch, in_f = x.shape
    N = grid_size
    dev, dt = x.device, x.dtype

    a, b = grid_range
    u = ((x - a) / (b - a)).clamp(1e-7, 1.0 - 1e-7)
    bases = torch.zeros(batch, in_f, N + 1, device=dev, dtype=dt)
    rd = torch.ones(batch, in_f, device=dev, dtype=dt)

    for k in range(depth):
        j = (u * N).long().clamp(0, N - 1)
        t = u * N - j.float()
        dj = torch.gather(
            d.unsqueeze(0).expand(batch, -1, -1),
            2, j.unsqueeze(-1)).squeeze(-1)

        bases.scatter_add_(
            2, (j + 1).unsqueeze(-1), (rd * t).unsqueeze(-1))
        bases.scatter_add_(
            2, j.unsqueeze(-1), (rd * (1.0 - t)).unsqueeze(-1))
        bases[:, :, 0] = bases[:, :, 0] - rd * dj * (1.0 - t)
        bases[:, :, N] = bases[:, :, N] - rd * dj * t

        rd = rd * dj
        u = t

    # Base case: piecewise linear
    j = (u * N).long().clamp(0, N - 1)
    t = u * N - j.float()
    bases.scatter_add_(
        2, j.unsqueeze(-1), (rd * (1.0 - t)).unsqueeze(-1))
    bases.scatter_add_(
        2, (j + 1).unsqueeze(-1), (rd * t).unsqueeze(-1))

    return bases.contiguous()


def fractal_dim_from_d(d, N):
    """
    Compute box-counting dimension from contraction parameters.

    dimB(d) = 1 + max(0, log(sum|d_j|) / log(N))

    This is the differentiable version of Theorem 2.3 (Barnsley, 1986).
    When sum|d_j| <= 1, dimB = 1 (smooth regime).
    When sum|d_j| > 1,  dimB > 1 (fractal regime).

    Args:
        d: (..., N) contraction parameters
        N: number of intervals (grid_size)

    Returns:
        dimB: (...) box-counting dimensions
    """
    s = d.abs().sum(dim=-1).clamp(min=1e-10)
    return 1.0 + F.relu(torch.log(s) / math.log(max(N, 2)))
