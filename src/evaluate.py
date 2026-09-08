import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

try:
    import properscoring as ps
    HAS_PROPERSCORING = True
except ImportError:
    HAS_PROPERSCORING = False


# ============================================================
# Metrics
# ============================================================

def mae(pred, true):
    return np.mean(np.abs(pred - true))

def mse(pred, true):
    return np.mean((pred - true) ** 2)

def rmse(pred, true):
    return np.sqrt(mse(pred, true))

def mape(pred, true, eps=1e-8):
    return np.mean(np.abs((pred - true) / (np.abs(true) + eps))) * 100

def crps_ensemble(truth, samples):
    """
    CRPS via properscoring.
    truth:   [B, H]
    samples: [B, H, n_samples]
    """
    if not HAS_PROPERSCORING:
        return float('nan')
    return ps.crps_ensemble(truth, samples).mean()

def coverage(samples, truth, lo_pct=5, hi_pct=95):
    """
    Empirical coverage: fraction of truth inside [lo, hi] prediction interval.
    samples: [n_samples, B, H, 1]
    truth:   [B, H, 1]
    """
    lo = np.percentile(samples, lo_pct, axis=0)
    hi = np.percentile(samples, hi_pct, axis=0)
    return ((truth >= lo) & (truth <= hi)).mean()

def interval_width(samples, lo_pct=5, hi_pct=95):
    """Mean width of prediction interval."""
    lo = np.percentile(samples, lo_pct, axis=0)
    hi = np.percentile(samples, hi_pct, axis=0)
    return (hi - lo).mean()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_models(dlinear, patchtst, diffusion, test_loader, test_dataset,
                    device='cuda', n_samples=50, ddim_steps=20):
    """
    Evaluate all three models on the test set.

    Returns
    -------
    results   : dict of aggregated metrics
    plot_data : dict with arrays for the first batch (used for plotting)
    """
    dlinear.eval()
    patchtst.eval()
    diffusion.eval()

    # accumulators
    dl_preds, pt_preds, diff_samples_all, truths = [], [], [], []

    for x, y, c in test_loader:
        x, y, c = x.to(device), y.to(device), c.to(device)
        y_np = y.cpu().numpy()

        # point predictions
        dl_pred = dlinear(x, c).cpu().numpy()
        pt_pred = patchtst(x, c).cpu().numpy()

        # diffusion samples  [n_samples, B, H, 1]
        samples = diffusion.sample(x, c, n_samples=n_samples, ddim_steps=ddim_steps)

        dl_preds.append(dl_pred)
        pt_preds.append(pt_pred)
        diff_samples_all.append(samples)
        truths.append(y_np)

    # concatenate across batches
    dl_preds  = np.concatenate(dl_preds, axis=1) if len(dl_preds[0].shape) == 4 \
                else np.concatenate(dl_preds, axis=0)
    pt_preds  = np.concatenate(pt_preds, axis=0)
    truths    = np.concatenate(truths, axis=0)                    # [N, H, 1]

    # diff_samples_all: list of [n_samples, B_i, H, 1] -> concat along B
    diff_all  = np.concatenate(diff_samples_all, axis=1)          # [n_samples, N, H, 1]
    diff_median = np.median(diff_all, axis=0)                     # [N, H, 1]

    # --- point metrics ---
    results = {
        'dlinear_mae':  float(mae(dl_preds, truths)),
        'dlinear_rmse': float(rmse(dl_preds, truths)),
        'patchtst_mae':  float(mae(pt_preds, truths)),
        'patchtst_rmse': float(rmse(pt_preds, truths)),
        'diffusion_mae':  float(mae(diff_median, truths)),
        'diffusion_rmse': float(rmse(diff_median, truths)),
    }

    # --- probabilistic metrics ---
    # CRPS: properscoring expects truth [N, H], samples [N, H, n_samples]
    truth_sq = truths.squeeze(-1)                                  # [N, H]
    samp_sq  = diff_all.squeeze(-1).transpose(1, 2, 0)            # [N, H, n_samples]
    results['diffusion_crps'] = float(crps_ensemble(truth_sq, samp_sq))

    results['diffusion_cov90'] = float(coverage(diff_all, truths, 5, 95))
    results['diffusion_cov80'] = float(coverage(diff_all, truths, 10, 90))
    results['diffusion_width90'] = float(interval_width(diff_all, 5, 95))

    # save first-batch data for downstream plots
    first_batch_samples = diff_samples_all[0]                     # [n_samples, B, H, 1]
    first_x = next(iter(test_loader))[0].cpu().numpy()
    first_y = truths[:first_batch_samples.shape[1]]

    plot_data = {
        'x':  first_x,
        'y':  first_y,
        'dl': dl_preds[:first_batch_samples.shape[1]],
        'pt': pt_preds[:first_batch_samples.shape[1]],
        'td': first_batch_samples,
    }
    return results, plot_data


# ============================================================
# Plots
# ============================================================

def plot_forecast_comparison(plot_data, save_dir='results', n_windows=4):
    """Multi-panel standard academic forecast plot."""
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    os.makedirs(save_dir, exist_ok=True)

    x      = plot_data['x']
    y      = plot_data['y']
    dl     = plot_data['dl']
    pt     = plot_data['pt']
    td     = plot_data['td']          # [n_samples, B, H, 1]

    seq_len  = x.shape[1]
    pred_len = y.shape[1]
    t_past   = np.arange(-seq_len, 0)
    t_future = np.arange(0, pred_len)

    n_windows = min(n_windows, x.shape[0])
    
    # Save individual plots (3-4 separate files)
    for i in range(n_windows):
        fig_ind, ax_ind = plt.subplots(figsize=(6, 4))
        ax_ind.plot(t_past, x[i, :, 0], color='black', linewidth=1.5, label='Lookback')
        ax_ind.plot(t_future, y[i, :, 0], color='black', linewidth=2.0, label='Ground Truth')
        ax_ind.plot(t_future, dl[i, :, 0], color='#1f77b4', linestyle='--', linewidth=1.5, label='DLinear')
        ax_ind.plot(t_future, pt[i, :, 0], color='#2ca02c', linestyle='-.', linewidth=1.5, label='PatchTST')
        
        p5  = np.percentile(td[:, i, :, 0], 5,  axis=0)
        p95 = np.percentile(td[:, i, :, 0], 95, axis=0)
        med = np.median(td[:, i, :, 0], axis=0)
        
        ax_ind.fill_between(t_future, p5, p95, color='#ff7f0e', alpha=0.3, linewidth=0)
        ax_ind.plot(t_future, med, color='#d62728', linewidth=1.5, label='CVAE (Median)')
        ax_ind.plot([], [], color='#ff7f0e', alpha=0.3, linewidth=10, label='CVAE (90% CI)')
        
        ax_ind.set_ylabel('Normalized Temp')
        ax_ind.set_xlabel('Time Steps')
        ax_ind.grid(True, alpha=0.5, linestyle='--')
        if i == 0:
            fig_ind.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        fig_ind.tight_layout()
        fig_ind.savefig(os.path.join(save_dir, f'forecast_window_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_ind)

    # Save the 2x2 grid for convenience
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for i in range(4):
        if i >= n_windows:
            axes[i].axis('off')
            continue
        ax = axes[i]

        # Past context
        ax.plot(t_past, x[i, :, 0], color='black', linewidth=1.5, label='Lookback' if i==0 else "")
        
        # Ground truth
        ax.plot(t_future, y[i, :, 0], color='black', linewidth=2.0, label='Ground Truth' if i==0 else "")

        # DLinear
        ax.plot(t_future, dl[i, :, 0], color='#1f77b4', linestyle='--', linewidth=1.5, label='DLinear' if i==0 else "")
        
        # PatchTST
        ax.plot(t_future, pt[i, :, 0], color='#2ca02c', linestyle='-.', linewidth=1.5, label='PatchTST' if i==0 else "")

        # DiffusionTS
        p5  = np.percentile(td[:, i, :, 0], 5,  axis=0)
        p95 = np.percentile(td[:, i, :, 0], 95, axis=0)
        med = np.median(td[:, i, :, 0], axis=0)

        ax.fill_between(t_future, p5, p95, color='#ff7f0e', alpha=0.3, linewidth=0)
        ax.plot(t_future, med, color='#d62728', linewidth=1.5, label='DiffusionTS (Median)' if i==0 else "")
        if i == 0:
            ax.plot([], [], color='#ff7f0e', alpha=0.3, linewidth=10, label='DiffusionTS (90% CI)')

        ax.set_ylabel('Normalized Temp')
        ax.grid(True, alpha=0.5, linestyle='--')
        if i >= 2:
            ax.set_xlabel('Time Steps')
        
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'forecast_comparison.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_extreme_event(plot_data, save_dir='results', threshold_sigma=2.0):
    """
    Show how the diffusion model captures extreme-event probability
    while point models cannot.
    """
    os.makedirs(save_dir, exist_ok=True)
    td = plot_data['td']   # [n_samples, B, H, 1]
    y  = plot_data['y']    # [B, H, 1]

    idx = 0
    samples_i = td[:, idx, :, 0]       # [n_samples, H]
    truth_i   = y[idx, :, 0]           # [H]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of day-1 forecast samples
    day = 0
    ax1.hist(samples_i[:, day], bins=20, color='#F44336', alpha=0.7,
             edgecolor='white')
    ax1.axvline(truth_i[day], color='black', linewidth=2, label='Truth')
    ax1.axvline(threshold_sigma, color='orange', linewidth=2, linestyle=':',
                label=f'Threshold ({threshold_sigma}σ)')
    prob = (samples_i[:, day] > threshold_sigma).mean()
    ax1.set_title(f'Day-1 Forecast Distribution\n'
                  f'P(extreme) = {prob*100:.1f}%')
    ax1.set_xlabel('Normalised Temperature')
    ax1.legend()

    # Right: P(extreme) over forecast horizon
    probs = (samples_i > threshold_sigma).mean(axis=0)   # [H]
    ax2.bar(range(len(probs)), probs * 100, color='#F44336', alpha=0.7)
    ax2.set_xlabel('Forecast Day')
    ax2.set_ylabel('P(extreme) %')
    ax2.set_title('Extreme-Event Probability Over Horizon')
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(save_dir, 'extreme_event.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_calibration(plot_data, save_dir='results'):
    """Reliability diagram: nominal coverage vs actual coverage."""
    os.makedirs(save_dir, exist_ok=True)
    td = plot_data['td']   # [n_samples, B, H, 1]
    y  = plot_data['y']    # [B, H, 1]

    nominals = np.arange(10, 100, 10)     # 10%, 20%, ... 90%
    actuals  = []
    for p in nominals:
        lo = (100 - p) / 2
        hi = 100 - lo
        actuals.append(coverage(td, y, lo, hi))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 100], [0, 100], 'k--', label='Perfect')
    ax.plot(nominals, np.array(actuals) * 100, 'o-', color='#F44336',
            label='DiffusionTS')
    ax.set_xlabel('Nominal Coverage (%)')
    ax.set_ylabel('Actual Coverage (%)')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(save_dir, 'calibration.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_results_json(results, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path}")
