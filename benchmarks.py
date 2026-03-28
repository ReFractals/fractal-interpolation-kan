# FI-KAN: Benchmark Data Generators
# --------------------------------------------------------
# Data generators for non-smooth PDE solutions and structured
# roughness targets (Section 5.11).
#
# Dependencies (install via pip):
#   scikit-fem >= 9.0   (finite element assembly)
#   fbm >= 0.3          (fractional Brownian motion)
#   scipy >= 1.10       (sparse linear algebra)
#
# Each generator returns (xt, yt, xe, ye, metadata) where
# xt, yt are training tensors and xe, ye are test tensors.
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

import math
import numpy as np
import torch

try:
    from fbm import FBM
except ImportError:
    FBM = None

try:
    import scipy.sparse.linalg as sla
    import scipy.sparse as sp
except ImportError:
    sla = None
    sp = None


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# Fractional Brownian Motion (fbm package)
# ================================================================

def generate_fbm_data(n_train, n_test, H, seed=42, device=None):
    """
    Generate fBm path regression data using the fbm package (Cholesky method).

    The path has Holder exponent exactly H.

    Args:
        n_train, n_test: number of train/test samples
        H: Hurst parameter in (0, 1)
        seed: random seed
        device: torch device

    Returns:
        xt, yt, xe, ye: tensors on device
    """
    assert FBM is not None, "Install fbm: pip install fbm"
    if device is None:
        device = _get_device()
    f = FBM(n=n_train + n_test - 1, hurst=H, length=1.0, method='cholesky')
    np.random.seed(seed)
    path = f.fbm()
    t = f.times()
    t_norm = (t * 2 - 1).astype(np.float32)
    path = path.astype(np.float32)
    pm, ps = path.mean(), path.std() + 1e-8
    path = (path - pm) / ps
    xt = torch.tensor(t_norm[:n_train], device=device).unsqueeze(-1)
    yt = torch.tensor(path[:n_train], device=device).unsqueeze(-1)
    xe = torch.tensor(t_norm[n_train:], device=device).unsqueeze(-1)
    ye = torch.tensor(path[n_train:], device=device).unsqueeze(-1)
    return xt, yt, xe, ye


# ================================================================
# L-Shaped Domain Laplacian (scikit-fem)
# ================================================================

def generate_lshaped_fem_data(n_refine=5, n_train=3000, n_test=1000,
                              seed=42, device=None):
    """
    L-shaped domain: -Delta u = 1, u = 0 on boundary.

    Corner singularity at the origin: u ~ r^{2/3} sin(2 theta / 3).
    Holder exponent = 2/3. Reference: Grisvard (1985).

    Uses scikit-fem P1 FEM on a refined triangular mesh.

    Args:
        n_refine: number of mesh refinements
        n_train, n_test: number of train/test samples
        seed: random seed
        device: torch device

    Returns:
        xt, yt, xe, ye, metadata
    """
    from skfem import MeshTri, ElementTriP1, Basis
    from skfem.models.poisson import laplace, mass

    if device is None:
        device = _get_device()

    m = MeshTri.init_lshaped().refined(n_refine)
    e = ElementTriP1()
    ib = Basis(m, e)

    A = laplace.assemble(ib)
    M = mass.assemble(ib)
    load = M @ np.ones(M.shape[0])

    D = ib.get_dofs(m.boundary_facets())
    dofs = D.all()
    interior = np.setdiff1d(np.arange(A.shape[0]), dofs)

    u = np.zeros(A.shape[0])
    u[interior] = sla.spsolve(A[interior][:, interior], load[interior])

    x_all = m.p.T.astype(np.float32)
    y_all = u.astype(np.float32).reshape(-1, 1)

    ym, ys = y_all.mean(), y_all.std() + 1e-8
    y_all = (y_all - ym) / ys

    np.random.seed(seed)
    idx = np.random.permutation(len(x_all))
    n_tr = min(n_train, len(idx) - n_test)
    n_te = min(n_test, len(idx) - n_tr)

    xt = torch.tensor(x_all[idx[:n_tr]], device=device)
    yt = torch.tensor(y_all[idx[:n_tr]], device=device)
    xe = torch.tensor(x_all[idx[n_tr:n_tr + n_te]], device=device)
    ye = torch.tensor(y_all[idx[n_tr:n_tr + n_te]], device=device)
    return xt, yt, xe, ye, {'mesh': m, 'u_fem': u, 'y_mean': ym, 'y_std': ys}


# ================================================================
# Rough-Coefficient Diffusion (scikit-fem + fbm)
# ================================================================

def generate_rough_diffusion_fem_data(n_fem=500, n_train=1500, n_test=500,
                                      H_coeff=0.3, seed=42, device=None):
    """
    Rough-coefficient diffusion: -d/dx(a(x) du/dx) = 1 on [0,1],
    u(0) = u(1) = 0, where a(x) = exp(0.5 * B_H(x)).

    The solution inherits structured roughness from the coefficient
    field through the PDE operator. This is the setting producing
    the strongest results in the paper (65-79x improvement, Table 10).

    Uses scikit-fem P1 FEM with fbm-generated coefficient field.

    Args:
        n_fem: number of FEM elements
        n_train, n_test: number of train/test samples
        H_coeff: Hurst parameter for the coefficient field
        seed: random seed
        device: torch device

    Returns:
        xt, yt, xe, ye, metadata
    """
    assert FBM is not None, "Install fbm: pip install fbm"
    from skfem import MeshLine, ElementLineP1, Basis
    from skfem.models.poisson import laplace, mass

    if device is None:
        device = _get_device()

    f = FBM(n=n_fem - 1, hurst=H_coeff, length=1.0, method='cholesky')
    np.random.seed(seed)
    fbm_path = f.fbm()
    a_vals = np.exp(0.5 * fbm_path).astype(np.float64)

    m = MeshLine(np.linspace(0, 1, n_fem))
    e = ElementLineP1()
    ib = Basis(m, e)

    K = laplace.assemble(ib)
    M_mat = mass.assemble(ib)
    load = M_mat @ np.ones(M_mat.shape[0])

    a_diag = np.zeros(K.shape[0])
    for i in range(len(a_vals)):
        a_diag[i] = a_vals[i]

    A_scaled = sp.diags(a_diag) @ K

    D = ib.get_dofs(m.boundary_facets())
    dofs = D.all()
    interior = np.setdiff1d(np.arange(K.shape[0]), dofs)

    u = np.zeros(K.shape[0])
    u[interior] = sla.spsolve(A_scaled[interior][:, interior], load[interior])

    x_verts = m.p[0].astype(np.float32)
    x_norm = x_verts * 2 - 1
    u_vals = u.astype(np.float32)

    um, us = u_vals.mean(), u_vals.std() + 1e-8
    u_norm = (u_vals - um) / us

    np.random.seed(seed + 1)
    idx = np.random.permutation(len(x_norm))
    n_tr = min(n_train, len(idx) - n_test)
    n_te = min(n_test, len(idx) - n_tr)

    xt = torch.tensor(x_norm[idx[:n_tr]], device=device).unsqueeze(-1)
    yt = torch.tensor(u_norm[idx[:n_tr]], device=device).unsqueeze(-1)
    xe = torch.tensor(x_norm[idx[n_tr:n_tr + n_te]], device=device).unsqueeze(-1)
    ye = torch.tensor(u_norm[idx[n_tr:n_tr + n_te]], device=device).unsqueeze(-1)
    return xt, yt, xe, ye, {'H_coeff': H_coeff, 'a_vals': a_vals, 'u_fem': u}


# ================================================================
# Stochastic Heat Equation (exact spectral)
# ================================================================

def generate_stochastic_heat_data(n_x=2000, n_train=1500, n_test=500,
                                  nu=0.01, sigma_noise=0.5, t_final=0.1,
                                  n_modes=50, seed=42, device=None):
    """
    Spatial snapshot of the stochastic heat equation at t = t_final
    via exact spectral representation.

    du = nu * u_xx dt + sigma dW, periodic BC on [0,1].
    Exact mode-by-mode solution with n_modes Fourier modes.

    Args:
        n_x: spatial grid resolution
        n_train, n_test: number of train/test samples
        nu: diffusion coefficient
        sigma_noise: noise intensity
        t_final: snapshot time
        n_modes: number of Fourier modes
        seed: random seed
        device: torch device

    Returns:
        xt, yt, xe, ye, metadata
    """
    if device is None:
        device = _get_device()
    np.random.seed(seed)
    x = np.linspace(0, 1, n_x).astype(np.float32)
    u = np.zeros(n_x, dtype=np.float64)
    for k in range(1, n_modes + 1):
        lam_k = nu * (k * math.pi) ** 2
        a0 = np.random.randn() / k**1.5
        a_det = a0 * np.exp(-lam_k * t_final)
        var_k = sigma_noise**2 / (2 * lam_k + 1e-15) * (1 - np.exp(-2 * lam_k * t_final))
        a_stoch = np.sqrt(max(var_k, 0)) * np.random.randn()
        u += (a_det + a_stoch) * np.sin(k * math.pi * x)
    u = u.astype(np.float32)
    um, us = u.mean(), u.std() + 1e-8
    u = (u - um) / us
    x_norm = (x * 2 - 1).astype(np.float32)
    np.random.seed(seed + 2)
    idx = np.random.permutation(n_x)
    xt = torch.tensor(x_norm[idx[:n_train]], device=device).unsqueeze(-1)
    yt = torch.tensor(u[idx[:n_train]], device=device).unsqueeze(-1)
    xe = torch.tensor(x_norm[idx[n_train:n_train + n_test]], device=device).unsqueeze(-1)
    ye = torch.tensor(u[idx[n_train:n_train + n_test]], device=device).unsqueeze(-1)
    return xt, yt, xe, ye, {'nu': nu, 'sigma': sigma_noise}


# ================================================================
# Rough Volatility Paths (fbm package)
# ================================================================

def generate_rough_vol_data(n_train, n_test, H=0.1, eta=0.5,
                            seed=42, device=None):
    """
    Rough volatility: sigma(t) = exp(eta * B_H(t)).

    H = 0.1 is the empirical value for equity markets
    (Gatheral, Jaisson, Rosenbaum, 2018).

    Args:
        n_train, n_test: number of train/test samples
        H: Hurst parameter (default 0.1, rough regime)
        eta: volatility-of-volatility
        seed: random seed
        device: torch device

    Returns:
        xt, yt, xe, ye, metadata
    """
    assert FBM is not None, "Install fbm: pip install fbm"
    if device is None:
        device = _get_device()
    f = FBM(n=n_train + n_test - 1, hurst=H, length=1.0, method='cholesky')
    np.random.seed(seed)
    path = f.fbm()
    sigma = np.exp(eta * path).astype(np.float32)
    t = f.times().astype(np.float32)
    t_norm = t * 2 - 1
    sm, ss = sigma.mean(), sigma.std() + 1e-8
    sigma_n = (sigma - sm) / ss
    xt = torch.tensor(t_norm[:n_train], device=device).unsqueeze(-1)
    yt = torch.tensor(sigma_n[:n_train], device=device).unsqueeze(-1)
    xe = torch.tensor(t_norm[n_train:], device=device).unsqueeze(-1)
    ye = torch.tensor(sigma_n[n_train:], device=device).unsqueeze(-1)
    return xt, yt, xe, ye, {'H': H, 'eta': eta}


# ================================================================
# Fractal Terrain (Diamond-Square Algorithm)
# ================================================================

def diamond_square(size_exp, roughness=0.5, seed=42):
    """
    Diamond-square fractal terrain generator.

    Surface fractal dimension: dimB ~ 3 - roughness.

    Args:
        size_exp: terrain size = 2^size_exp + 1
        roughness: roughness parameter in (0, 1)
        seed: random seed

    Returns:
        terrain: (n, n) float32 array
    """
    np.random.seed(seed)
    n = 2**size_exp + 1
    terrain = np.zeros((n, n), dtype=np.float64)
    terrain[0, 0] = np.random.randn()
    terrain[0, n-1] = np.random.randn()
    terrain[n-1, 0] = np.random.randn()
    terrain[n-1, n-1] = np.random.randn()
    step = n - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        for y in range(0, n-1, step):
            for x in range(0, n-1, step):
                avg = (terrain[y, x] + terrain[y, x+step]
                       + terrain[y+step, x] + terrain[y+step, x+step]) / 4.0
                terrain[y+half, x+half] = avg + scale * np.random.randn()
        for y in range(0, n, half):
            for x in range((y+half) % step, n, step):
                vals = []
                if y >= half:
                    vals.append(terrain[y-half, x])
                if y+half < n:
                    vals.append(terrain[y+half, x])
                if x >= half:
                    vals.append(terrain[y, x-half])
                if x+half < n:
                    vals.append(terrain[y, x+half])
                terrain[y, x] = np.mean(vals) + scale * np.random.randn()
        scale *= 2.0**(-roughness)
        step = half
    return terrain.astype(np.float32)


def generate_terrain_data(n_train, n_test, roughness=0.5, size_exp=7,
                          seed=42, device=None):
    """
    Generate fractal terrain regression data.

    Maps (x, y) -> elevation z via bilinear interpolation
    on a diamond-square terrain.

    Args:
        n_train, n_test: number of train/test samples
        roughness: terrain roughness (dimB ~ 3 - roughness)
        size_exp: terrain grid size = 2^size_exp + 1
        seed: random seed
        device: torch device

    Returns:
        xt, yt, xe, ye, metadata
    """
    if device is None:
        device = _get_device()
    terrain = diamond_square(size_exp, roughness, seed)
    n = terrain.shape[0]
    terrain = (terrain - terrain.mean()) / (terrain.std() + 1e-8)
    np.random.seed(seed + 100)
    total = n_train + n_test
    xi = np.random.rand(total) * (n-1)
    yi = np.random.rand(total) * (n-1)
    x0 = np.floor(xi).astype(int).clip(0, n-2)
    y0 = np.floor(yi).astype(int).clip(0, n-2)
    fx = xi - x0
    fy = yi - y0
    z = (terrain[y0, x0] * (1-fx) * (1-fy)
         + terrain[y0, x0+1] * fx * (1-fy)
         + terrain[y0+1, x0] * (1-fx) * fy
         + terrain[y0+1, x0+1] * fx * fy)
    xy = np.stack([(xi/(n-1))*2-1, (yi/(n-1))*2-1], axis=1).astype(np.float32)
    z = z.astype(np.float32).reshape(-1, 1)
    xt = torch.tensor(xy[:n_train], device=device)
    yt = torch.tensor(z[:n_train], device=device)
    xe = torch.tensor(xy[n_train:], device=device)
    ye = torch.tensor(z[n_train:], device=device)
    return xt, yt, xe, ye, {
        'roughness': roughness,
        'approx_dim': 3 - roughness,
        'terrain': terrain,
    }
