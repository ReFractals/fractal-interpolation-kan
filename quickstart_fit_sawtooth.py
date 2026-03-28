"""
Quickstart: Fit the Takagi sawtooth function (dimB = 1.5).

Demonstrates Hybrid FI-KAN vs. KAN on a fractal target.
Expected result: FI-KAN achieves ~50x lower MSE than KAN.

Usage:
    python quickstart_fit_sawtooth.py
"""
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from models import HybridFIKAN
from baselines import KAN
from targets import fractal_sawtooth
from training import train_model, count_params

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# --- Data ---
n_train, n_test = 1000, 400
x_train = torch.linspace(-0.95, 0.95, n_train, device=device).unsqueeze(-1)
y_train = fractal_sawtooth(x_train.squeeze(-1), depth=8).unsqueeze(-1)
x_test = torch.linspace(-0.95, 0.95, n_test, device=device).unsqueeze(-1)
y_test = fractal_sawtooth(x_test.squeeze(-1), depth=8).unsqueeze(-1)

# --- KAN baseline ---
torch.manual_seed(42)
kan = KAN([1, 16, 1], grid_size=8, spline_order=3).to(device)
print(f"KAN params: {count_params(kan)}")
res_kan = train_model(
    kan, x_train, y_train, x_test, y_test,
    epochs=500, lr=1e-3, model_type='kan')
print(f"KAN best test MSE: {res_kan['best_test']:.6e}")

# --- Hybrid FI-KAN ---
torch.manual_seed(42)
fikan = HybridFIKAN(
    [1, 16, 1], grid_size=8, spline_order=3,
    fractal_depth=6, d_init_std=0.01).to(device)
print(f"Hybrid FI-KAN params: {count_params(fikan)}")
res_fikan = train_model(
    fikan, x_train, y_train, x_test, y_test,
    epochs=500, lr=1e-3, reg_frac=0.001, model_type='fikan_hybrid')
print(f"Hybrid FI-KAN best test MSE: {res_fikan['best_test']:.6e}")

# --- Summary ---
ratio = res_kan['best_test'] / (res_fikan['best_test'] + 1e-15)
print(f"\nImprovement ratio: {ratio:.1f}x")

# Report learned fractal dimensions
fikan.eval()
dims = fikan.fractal_dimensions()
for i, d in enumerate(dims):
    print(f"Layer {i} learned dimB: {d.mean().item():.4f}")
