# FI-KAN: Training Infrastructure
# --------------------------------------------------------
# Minimal training loop with multi-seed evaluation,
# gradient clipping, ReduceLROnPlateau scheduling,
# and fractal dimension tracking.
#
# Default configuration (Section 5.1):
#   500 epochs, Adam lr=1e-3, patience 50, factor 0.5,
#   gradient clipping at norm 1.0, lambda_frac = 0.001,
#   5 seeds: {42, 123, 456, 789, 2024}.
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SEEDS = [42, 123, 456, 789, 2024]


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, xt, yt, xe, ye, epochs=500, lr=1e-3,
                batch_size=None, reg_act=0.0, reg_ent=0.0,
                reg_frac=0.0, patience=50, model_type='kan'):
    """
    Train a model and return comprehensive metrics.

    Args:
        model: nn.Module (KAN, PureFIKAN, HybridFIKAN, or MLP)
        xt, yt: training data tensors
        xe, ye: test data tensors
        epochs: number of training epochs
        lr: initial learning rate for Adam
        batch_size: mini-batch size (None for full-batch)
        reg_act: activation regularization weight
        reg_ent: entropy regularization weight
        reg_frac: fractal dimension regularization weight
        patience: ReduceLROnPlateau patience
        model_type: 'kan', 'fikan', 'fikan_hybrid', or 'mlp'

    Returns:
        dict with best_test, final_test, history, timing, etc.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=0.5, min_lr=1e-6)

    if batch_size and batch_size < len(xt):
        dataset = TensorDataset(xt, yt)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = [(xt, yt)]

    hist = {
        'train_loss': [], 'test_loss': [],
        'epoch_times': [], 'fractal_dims': [],
        'fractal_energy': [], 'lr': [],
    }
    best_test = float('inf')
    best_epoch = 0
    t_total_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            if reg_frac > 0 and model_type in ('fikan', 'fikan_hybrid'):
                loss = loss + model.regularization_loss(
                    reg_act, reg_ent, reg_frac)
            elif (reg_act > 0 or reg_ent > 0):
                if model_type == 'kan':
                    loss = loss + model.regularization_loss(
                        reg_act, reg_ent)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        epoch_loss /= max(n_samples, 1)
        epoch_time = time.time() - t0

        model.eval()
        with torch.no_grad():
            pred_test = model(xe)
            test_loss = F.mse_loss(pred_test, ye).item()

        scheduler.step(test_loss)
        if test_loss < best_test:
            best_test = test_loss
            best_epoch = epoch

        hist['train_loss'].append(epoch_loss)
        hist['test_loss'].append(test_loss)
        hist['epoch_times'].append(epoch_time)
        hist['lr'].append(optimizer.param_groups[0]['lr'])

        if model_type in ('fikan', 'fikan_hybrid') and hasattr(model, 'fractal_dimensions'):
            with torch.no_grad():
                dims = [d.cpu().numpy().tolist()
                        for d in model.fractal_dimensions()]
                hist['fractal_dims'].append(dims)
                if hasattr(model, 'fractal_energy_ratios'):
                    hist['fractal_energy'].append(
                        model.fractal_energy_ratios())

    total_time = time.time() - t_total_start

    return {
        'best_test': best_test,
        'best_epoch': best_epoch,
        'final_test': hist['test_loss'][-1],
        'total_time': total_time,
        'avg_epoch_time': np.mean(hist['epoch_times']),
        'params': count_params(model),
        'history': hist,
    }


def run_multiseed(model_fn, xt, yt, xe, ye, seeds=None, epochs=500,
                  lr=1e-3, reg_frac=0.0, model_type='kan', **kwargs):
    """
    Run training across multiple seeds. Returns aggregated results.

    Args:
        model_fn: callable returning a fresh model instance
        xt, yt, xe, ye: train/test data tensors
        seeds: list of random seeds (default: SEEDS)
        epochs, lr, reg_frac, model_type: forwarded to train_model
        **kwargs: additional arguments for train_model

    Returns:
        dict with mean_test, std_test, individual results, etc.
    """
    if seeds is None:
        seeds = SEEDS

    device = xt.device
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = model_fn().to(device)
        res = train_model(
            model, xt, yt, xe, ye, epochs=epochs, lr=lr,
            reg_frac=reg_frac, model_type=model_type, **kwargs)
        res['seed'] = seed
        results.append(res)

    tests = [r['best_test'] for r in results]
    times = [r['total_time'] for r in results]
    agg = {
        'mean_test': np.mean(tests),
        'std_test': np.std(tests),
        'median_test': np.median(tests),
        'min_test': np.min(tests),
        'max_test': np.max(tests),
        'mean_time': np.mean(times),
        'params': results[0]['params'],
        'n_seeds': len(seeds),
        'individual': results,
    }
    return agg
