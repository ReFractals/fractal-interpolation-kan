# FI-KAN: Target Functions
# --------------------------------------------------------
# Target functions spanning the Holder regularity spectrum,
# from smooth (alpha = 2) to fractal (dimB > 1).
#
# 1D targets: polynomial, exp_sin, chirp, Weierstrass (two
#   roughness levels), Takagi sawtooth, multiscale (mixed).
# Holder family: f_alpha(x) = |x|^alpha for alpha in [0.2, 2.0].
# 2D targets: Ackley, 2D Weierstrass product.
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

import math
import torch


# ================================================================
# 1D Target Functions
# ================================================================

def weierstrass(x, a=0.5, b=7.0, n_terms=30):
    """
    Weierstrass function: continuous, nowhere differentiable.

    W(x) = sum_{n=0}^{n_terms-1} a^n cos(b^n pi x)

    Graph dimension: dimB = 2 + log(a)/log(b).
      a=0.5, b=7 -> dimB ~ 1.644
      a=0.7, b=3 -> dimB ~ 1.675
    """
    r = torch.zeros_like(x)
    for n in range(n_terms):
        r = r + (a ** n) * torch.cos((b ** n) * math.pi * x)
    return r


def fractal_sawtooth(x, depth=12):
    """
    Takagi-Landsberg function with w = 2^{-1/2}.

    T_w(x) = sum_{n=0}^{depth-1} w^n phi(2^n x)

    where phi(x) = dist(x, Z) is the tent map and w = 2^{-1/2}.
    Graph dimension: dimB = 2 + log_2(w) = 1.5 exactly.
    Holder exponent: 1/2.

    Reference: Lagarias (2012), "The Takagi function: a survey".
    """
    w = 2.0 ** (-0.5)  # 1/sqrt(2)
    r = torch.zeros_like(x)
    for n in range(depth):
        s = (2.0 ** n) * x
        r = r + (w ** n) * (s - s.round()).abs()
    return r


def smooth_poly(x):
    """Polynomial: x^3 - 2x^2 + x - 0.5. dimB = 1.0."""
    return x ** 3 - 2 * x ** 2 + x - 0.5


def smooth_exp_sin(x):
    """Smooth: exp(sin(pi x)). dimB = 1.0."""
    return torch.exp(torch.sin(math.pi * x))


def chirp_fn(x):
    """Chirp: sin(20 pi x^2). dimB = 1.0."""
    return torch.sin(20.0 * math.pi * x ** 2)


def multiscale_fn(x):
    """Mixed regularity: smooth in [-1,0], rough in [0,1]."""
    return (torch.sin(2 * math.pi * x)
            + 0.3 * weierstrass(x, 0.5, 7.0, 25)
            * torch.sigmoid(10.0 * x))


def holder_fn(x, alpha):
    """Holder family: f_alpha(x) = |x|^alpha."""
    return x.abs().pow(alpha)


# ================================================================
# 2D Target Functions
# ================================================================

def ackley_2d(xy):
    """Ackley function on R^2. Smooth outside the origin."""
    x, y = xy[:, 0], xy[:, 1]
    t1 = -20.0 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2)))
    t2 = -torch.exp(0.5 * (torch.cos(2*math.pi*x) + torch.cos(2*math.pi*y)))
    return (t1 + t2 + math.e + 20.0).unsqueeze(-1)


def weierstrass_2d(xy):
    """2D Weierstrass product: W(x)*W(y)."""
    return (weierstrass(xy[:, 0], 0.5, 7.0, 20)
            * weierstrass(xy[:, 1], 0.5, 7.0, 20)).unsqueeze(-1)


# ================================================================
# Registries
# ================================================================

TARGETS_1D = {
    'polynomial':        (smooth_poly, 1.0),
    'exp_sin':           (smooth_exp_sin, 1.0),
    'chirp':             (chirp_fn, 1.0),
    'weierstrass_std':   (lambda x: weierstrass(x, 0.5, 7.0, 25), 1.356),
    'weierstrass_rough': (lambda x: weierstrass(x, 0.7, 3.0, 25), 1.675),
    'sawtooth':          (lambda x: fractal_sawtooth(x, 8), 1.5),
    'multiscale':        (multiscale_fn, None),
}

TARGETS_2D = {
    'ackley_2d': ackley_2d,
    'weierstrass_2d': weierstrass_2d,
}

HOLDER_EXPONENTS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
