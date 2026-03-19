"""
analyze_logs.py — Hell-Loop Protocol v6.2
Post-processing and statistical analysis of JSONL logs.

Three analysis modes:
  1. single  — detailed analysis of one JSONL file
  2. batch   — aggregate statistics across a folder of JSONL files
  3. compare — full statistical comparison of HellLoop vs ControlLoopA
               (and optionally ControlLoopB)

Usage:
  python analyze_logs.py single  --file logs/hellloop/hellloop_*.jsonl
  python analyze_logs.py batch   --dir  logs/hellloop/
  python analyze_logs.py compare --hell logs/hellloop/ --control logs/control_a/
  python analyze_logs.py compare --hell logs/hellloop/ --control logs/control_a/ \
                                 --control-b logs/control_b/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from scipy.stats import mannwhitneyu


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dir(directory: str) -> list[list[dict]]:
    """Load all JSONL files from a directory. Returns list of runs."""
    paths = sorted(glob(os.path.join(directory, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found in: {directory}")
    return [load_jsonl(p) for p in paths], paths


def _extract_series(run: list[dict], key: str) -> list[float]:
    return [r[key] for r in run if key in r and r[key] is not None]


# ══════════════════════════════════════════════════════════════════════════════
# Core metrics
# ══════════════════════════════════════════════════════════════════════════════

def hurst_exponent(series: list[float]) -> float:
    """
    Hurst exponent via R/S analysis.
    H > 0.6 → long-range memory (system builds structure).
    H ≈ 0.5 → random walk.
    H < 0.4 → mean-reverting.
    """
    ts = np.array(series, dtype=float)
    n  = len(ts)
    if n < 10:
        return float("nan")

    lags  = range(2, max(3, n // 2))
    rs    = []
    sizes = []

    for lag in lags:
        sub = ts[:lag]
        mean_sub = np.mean(sub)
        deviation = np.cumsum(sub - mean_sub)
        r = np.max(deviation) - np.min(deviation)
        s = np.std(sub, ddof=1)
        if s > 0:
            rs.append(r / s)
            sizes.append(lag)

    if len(rs) < 2:
        return float("nan")

    log_sizes = np.log(sizes)
    log_rs    = np.log(rs)
    slope, _, _, _, _ = stats.linregress(log_sizes, log_rs)
    return float(slope)


def divergence_slope(series: list[float]) -> float:
    """
    Linear slope of successive absolute differences.
    Positive slope → growing divergence (nonlinear, potentially chaotic).
    """
    if len(series) < 3:
        return float("nan")
    diffs = np.abs(np.diff(series))
    x     = np.arange(len(diffs), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, diffs)
    return float(slope)


def detect_bifurcations(series: list[float], z_threshold: float = 2.5) -> list[int]:
    """
    Returns iteration indices where a sudden structural shift occurs
    (z-score of first difference exceeds threshold).
    """
    if len(series) < 5:
        return []
    diffs = np.diff(series)
    if np.std(diffs) == 0:
        return []
    z = np.abs(stats.zscore(diffs))
    return [int(i + 1) for i in np.where(z > z_threshold)[0]]


def piecewise_regression(series: list[float]) -> dict:
    """
    Finds the single best breakpoint by minimizing total residual sum of squares
    across all candidate split points. Returns breakpoint index and slopes.
    """
    n = len(series)
    if n < 6:
        return {"breakpoint": None, "slope_before": None, "slope_after": None}

    x   = np.arange(n, dtype=float)
    y   = np.array(series, dtype=float)
    best_rss = float("inf")
    best_bp  = None

    for bp in range(2, n - 2):
        x1, y1 = x[:bp], y[:bp]
        x2, y2 = x[bp:], y[bp:]
        _, _, _, _, se1 = stats.linregress(x1, y1)
        _, _, _, _, se2 = stats.linregress(x2, y2)
        rss = se1 ** 2 * len(x1) + se2 ** 2 * len(x2)
        if rss < best_rss:
            best_rss = rss
            best_bp  = bp

    if best_bp is None:
        return {"breakpoint": None, "slope_before": None, "slope_after": None}

    s1, _, _, _, _ = stats.linregress(x[:best_bp], y[:best_bp])
    s2, _, _, _, _ = stats.linregress(x[best_bp:], y[best_bp:])
    return {
        "breakpoint"   : best_bp,
        "slope_before" : round(float(s1), 6),
        "slope_after"  : round(float(s2), 6),
    }


def pca_pc1_slope(pca_series: list[list]) -> float:
    """
    Linear slope of PC1 variance ratio over iterations.
    Negative slope → expanding semantic space (good signal).
    """
    vals = [p[0] for p in pca_series if p]
    if len(vals) < 3:
        return float("nan")
    x = np.arange(len(vals), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, vals)
    return float(slope)


def attractor_snr(series: list[float], window: int = 10) -> float:
    """Signal-to-noise ratio of the last `window` values."""
    if len(series) < window:
        return 0.0
    recent = np.array(series[-window:])
    return float(np.mean(recent) / (np.std(recent) + 1e-9))


# ══════════════════════════════════════════════════════════════════════════════
# Comparative statistics
# ══════════════════════════════════════════════════════════════════════════════

def cohen_d(a: list[float], b: list[float]) -> float:
    na, nb   = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def bootstrap_ci(
    a: list[float],
    b: list[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the difference in means (a - b)."""
    rng  = np.random.default_rng(42)
    a_arr, b_arr = np.array(a), np.array(b)
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(a_arr, size=len(a_arr), replace=True)
        sample_b = rng.choice(b_arr, size=len(b_arr), replace=True)
        diffs.append(np.mean(sample_a) - np.mean(sample_b))
    alpha = 1 - confidence
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, hi


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test. Returns (U statistic, p-value)."""
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")
    result = mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


# ══════════════════════════════════════════════════════════════════════════════
# Single-run analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_single(path: str, plot: bool = True) -> dict:
    run     = load_jsonl(path)
    run_id  = run[0].get("run_id", "unknown") if run else "unknown"

    cos_series  = _extract_series(run, "cosine_integration")
    tmp_series  = _extract_series(run, "temporal")
    self_scores = _extract_series(run, "self_score")
    pca_series  = [r["pca_variance_ratio"] for r in run
                   if r.get("pca_variance_ratio")]

    H_cos  = hurst_exponent(cos_series)
    H_self = hurst_exponent(self_scores)
    div    = divergence_slope(cos_series)
    bifs   = detect_bifurcations(cos_series)
    pw     = piecewise_regression(cos_series)
    pc1_s  = pca_pc1_slope(pca_series)
    snr    = attractor_snr(cos_series)

    self_modes = [r.get("self_mode") for r in run if r.get("self_mode")]
    metagnosis = any(r.get("metagnosis_detected") for r in run)
    bif_total  = max(
        (r.get("bifurcation_total", 0) for r in run), default=0
    )

    result = {
        "run_id"            : run_id,
        "n_iterations"      : len(run),
        "hurst_cosine"      : round(H_cos,  4),
        "hurst_self_score"  : round(H_self, 4),
        "divergence_slope"  : round(div,    6),
        "bifurcation_points": bifs,
        "bifurcation_total" : bif_total,
        "piecewise"         : pw,
        "pca_pc1_slope"     : round(pc1_s,  6),
        "attractor_snr"     : round(snr,    4),
        "self_modes"        : list(set(self_modes)),
        "metagnosis"        : metagnosis,
        "cosine_mean"       : round(float(np.mean(cos_series)),  4),
        "cosine_std"        : round(float(np.std(cos_series)),   4),
        "temporal_mean"     : round(float(np.mean(tmp_series)),  4),
        "self_score_mean"   : round(float(np.mean(self_scores)), 4),
        "self_score_std"    : round(float(np.std(self_scores)),  4),
    }

    _print_single(result)

    if plot:
        _plot_single(run, cos_series, tmp_series, self_scores,
                     pca_series, bifs, pw, run_id, path)

    return result


def _print_single(r: dict) -> None:
    print("\n" + "═" * 60)
    print(f"SINGLE RUN ANALYSIS  —  run_id: {r['run_id']}")
    print("═" * 60)
    print(f"  Iterations           : {r['n_iterations']}")
    print(f"  Cosine mean / std    : {r['cosine_mean']} / {r['cosine_std']}")
    print(f"  Temporal mean        : {r['temporal_mean']}")
    print(f"  SELF score mean/std  : {r['self_score_mean']} / {r['self_score_std']}")
    print(f"  Hurst (cosine)       : {r['hurst_cosine']}")
    print(f"  Hurst (self score)   : {r['hurst_self_score']}")
    print(f"  Divergence slope     : {r['divergence_slope']}")
    print(f"  Bifurcation points   : {r['bifurcation_points']}")
    print(f"  Bifurcations total   : {r['bifurcation_total']}")
    print(f"  Piecewise breakpoint : {r['piecewise']['breakpoint']}")
    print(f"    slope before       : {r['piecewise']['slope_before']}")
    print(f"    slope after        : {r['piecewise']['slope_after']}")
    print(f"  PCA PC1 slope        : {r['pca_pc1_slope']}")
    print(f"  Attractor SNR        : {r['attractor_snr']}")
    print(f"  SELF modes detected  : {r['self_modes']}")
    print(f"  Metagnosis           : {r['metagnosis']}")
    print("═" * 60 + "\n")


def _plot_single(
    run, cos_series, tmp_series, self_scores,
    pca_series, bifs, pw, run_id, path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    x = list(range(len(cos_series)))

    # Panel 1: integration + temporal
    axes[0].plot(x, cos_series,  label="Cosine integration", color="crimson")
    axes[0].plot(x, tmp_series,  label="Temporal integration", color="orange",
                 linestyle="--")
    for b in bifs:
        axes[0].axvline(b, color="purple", linestyle=":", alpha=0.7,
                        label=f"Bifurcation {b}")
    if pw["breakpoint"] is not None:
        axes[0].axvline(pw["breakpoint"], color="navy", linestyle="-.",
                        alpha=0.8, label=f"Breakpoint {pw['breakpoint']}")
    axes[0].set_title(f"Integration dynamics  —  run: {run_id}")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # Panel 2: SELF score
    axes[1].plot(x[:len(self_scores)], self_scores,
                 label="SELF score", color="teal")
    axes[1].set_title("SELF score trajectory")
    axes[1].set_ylabel("SELF score [0–1]")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # Panel 3: PCA PC1
    pca_vals = [p[0] for p in pca_series if p]
    if pca_vals:
        axes[2].plot(pca_vals, label="PCA PC1 variance ratio", color="steelblue")
        axes[2].set_title(
            "Semantic space dimensionality  "
            "(falling = expanding space)"
        )
        axes[2].set_ylabel("PC1 explained variance ratio")
        axes[2].legend(fontsize=7)
        axes[2].grid(alpha=0.3)

    axes[2].set_xlabel("Iteration")
    plt.tight_layout()
    out = path.replace(".jsonl", "_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [plot saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Batch analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_batch(directory: str) -> dict:
    runs, paths = load_dir(directory)
    label = os.path.basename(directory.rstrip("/\\"))

    all_hurst, all_cos_mean, all_self_mean = [], [], []
    all_div_slope, all_pc1_slope = [], []
    metagnosis_count = 0

    for run in runs:
        cos  = _extract_series(run, "cosine_integration")
        self = _extract_series(run, "self_score")
        pca  = [r["pca_variance_ratio"] for r in run if r.get("pca_variance_ratio")]

        all_hurst.append(hurst_exponent(cos))
        all_cos_mean.append(float(np.mean(cos)) if cos else float("nan"))
        all_self_mean.append(float(np.mean(self)) if self else float("nan"))
        all_div_slope.append(divergence_slope(cos))
        all_pc1_slope.append(pca_pc1_slope(pca))
        if any(r.get("metagnosis_detected") for r in run):
            metagnosis_count += 1

    def _safe_stats(arr):
        clean = [v for v in arr if not np.isnan(v)]
        if not clean:
            return float("nan"), float("nan")
        return round(float(np.mean(clean)), 4), round(float(np.std(clean)), 4)

    result = {
        "label"                   : label,
        "n_runs"                  : len(runs),
        "hurst_mean_std"          : _safe_stats(all_hurst),
        "cosine_mean_std"         : _safe_stats(all_cos_mean),
        "self_score_mean_std"     : _safe_stats(all_self_mean),
        "divergence_slope_mean_std": _safe_stats(all_div_slope),
        "pca_pc1_slope_mean_std"  : _safe_stats(all_pc1_slope),
        "metagnosis_rate"         : round(metagnosis_count / len(runs), 3),
        "_raw_hurst"              : all_hurst,
        "_raw_cos_mean"           : all_cos_mean,
        "_raw_self_mean"          : all_self_mean,
    }

    _print_batch(result)
    return result


def _print_batch(r: dict) -> None:
    print("\n" + "═" * 60)
    print(f"BATCH ANALYSIS  —  {r['label']}  ({r['n_runs']} runs)")
    print("═" * 60)
    print(f"  Hurst exponent      mean/std : {r['hurst_mean_std']}")
    print(f"  Cosine integration  mean/std : {r['cosine_mean_std']}")
    print(f"  SELF score          mean/std : {r['self_score_mean_std']}")
    print(f"  Divergence slope    mean/std : {r['divergence_slope_mean_std']}")
    print(f"  PCA PC1 slope       mean/std : {r['pca_pc1_slope_mean_std']}")
    print(f"  Metagnosis rate              : {r['metagnosis_rate']}")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Comparative analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_compare(
    hell_dir: str,
    control_dir: str,
    control_b_dir: str | None = None,
    plot: bool = True,
) -> dict:
    hell_batch    = analyze_batch(hell_dir)
    control_batch = analyze_batch(control_dir)
    batches       = [hell_batch, control_batch]
    labels        = ["HellLoop", "ControlLoop-A"]

    if control_b_dir:
        cb_batch = analyze_batch(control_b_dir)
        batches.append(cb_batch)
        labels.append("ControlLoop-B")

    # ── Hell vs Control-A ─────────────────────────────────────────────────────
    hl_cos   = hell_batch["_raw_cos_mean"]
    ca_cos   = control_batch["_raw_cos_mean"]
    hl_self  = hell_batch["_raw_self_mean"]
    ca_self  = control_batch["_raw_self_mean"]
    hl_hurst = hell_batch["_raw_hurst"]
    ca_hurst = control_batch["_raw_hurst"]

    d_cos   = cohen_d(hl_cos,   ca_cos)
    d_self  = cohen_d(hl_self,  ca_self)
    d_hurst = cohen_d(hl_hurst, ca_hurst)

    u_cos,  p_cos  = mann_whitney(hl_cos,   ca_cos)
    u_self, p_self = mann_whitney(hl_self,  ca_self)

    ci_cos_lo,  ci_cos_hi  = bootstrap_ci(hl_cos,  ca_cos)
    ci_self_lo, ci_self_hi = bootstrap_ci(hl_self, ca_self)

    result = {
        "hell_vs_control_a": {
            "cohen_d_cosine"        : round(d_cos,   4),
            "cohen_d_self_score"    : round(d_self,  4),
            "cohen_d_hurst"         : round(d_hurst, 4),
            "mann_whitney_cosine"   : {"U": round(u_cos, 2),  "p": round(p_cos, 6)},
            "mann_whitney_self"     : {"U": round(u_self, 2), "p": round(p_self, 6)},
            "bootstrap_ci_cosine"   : (round(ci_cos_lo, 4),  round(ci_cos_hi, 4)),
            "bootstrap_ci_self"     : (round(ci_self_lo, 4), round(ci_self_hi, 4)),
        }
    }

    _print_compare(result, hell_batch, control_batch, labels[0], labels[1])

    if plot:
        _plot_compare(batches, labels, hell_dir)

    return result


def _print_compare(result, hell, ctrl, hl_label, ca_label) -> None:
    r = result["hell_vs_control_a"]
    print("\n" + "═" * 60)
    print(f"COMPARATIVE ANALYSIS  —  {hl_label} vs {ca_label}")
    print("═" * 60)
    print(f"\n  Cosine integration:")
    print(f"    {hl_label} mean : {hell['cosine_mean_std'][0]}")
    print(f"    {ca_label} mean : {ctrl['cosine_mean_std'][0]}")
    print(f"    Cohen's d       : {r['cohen_d_cosine']}")
    print(f"    Mann-Whitney    : U={r['mann_whitney_cosine']['U']}  "
          f"p={r['mann_whitney_cosine']['p']}")
    print(f"    Bootstrap 95%CI : {r['bootstrap_ci_cosine']}")

    print(f"\n  SELF score:")
    print(f"    {hl_label} mean : {hell['self_score_mean_std'][0]}")
    print(f"    {ca_label} mean : {ctrl['self_score_mean_std'][0]}")
    print(f"    Cohen's d       : {r['cohen_d_self_score']}")
    print(f"    Mann-Whitney    : U={r['mann_whitney_self']['U']}  "
          f"p={r['mann_whitney_self']['p']}")
    print(f"    Bootstrap 95%CI : {r['bootstrap_ci_self']}")

    print(f"\n  Hurst exponent:")
    print(f"    {hl_label} mean : {hell['hurst_mean_std'][0]}")
    print(f"    {ca_label} mean : {ctrl['hurst_mean_std'][0]}")
    print(f"    Cohen's d       : {r['cohen_d_hurst']}")

    print(f"\n  Metagnosis rate:")
    print(f"    {hl_label}  : {hell['metagnosis_rate']}")
    print(f"    {ca_label}  : {ctrl['metagnosis_rate']}")

    print("\n  Interpretation guide:")
    d = abs(r["cohen_d_cosine"])
    interp = "negligible" if d < 0.2 else \
             "small"      if d < 0.5 else \
             "medium"     if d < 0.8 else "large"
    print(f"    Cohen's d = {r['cohen_d_cosine']} → effect size: {interp}")
    p = r["mann_whitney_cosine"]["p"]
    print(f"    p = {p} → {'significant (p < 0.05)' if p < 0.05 else 'not significant'}")
    print("═" * 60 + "\n")


def _plot_compare(batches: list, labels: list, hell_dir: str) -> None:
    colors = ["crimson", "steelblue", "seagreen"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (batch, label) in enumerate(zip(batches, labels)):
        cos_vals  = batch["_raw_cos_mean"]
        self_vals = batch["_raw_self_mean"]
        hurst_vals = batch["_raw_hurst"]
        c = colors[i % len(colors)]

        axes[0].boxplot(cos_vals,  positions=[i], patch_artist=True,
                        boxprops=dict(facecolor=c, alpha=0.6))
        axes[1].boxplot(self_vals, positions=[i], patch_artist=True,
                        boxprops=dict(facecolor=c, alpha=0.6))
        axes[2].boxplot(hurst_vals, positions=[i], patch_artist=True,
                        boxprops=dict(facecolor=c, alpha=0.6))

    for ax, title, ylabel in zip(
        axes,
        ["Cosine Integration", "SELF Score", "Hurst Exponent"],
        ["mean cosine / run", "mean SELF score / run", "H"],
    ):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Hell-Loop v6.2 — Comparative Analysis", fontsize=12)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(hell_dir.rstrip("/\\")),
                       "comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [plot saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# get_system_status — used by UI and batch_runner for live summary
# ══════════════════════════════════════════════════════════════════════════════

def get_system_status(history: list[dict]) -> str:
    if not history:
        return "NO DATA"

    cos_vals = _extract_series(history, "cosine_integration")
    if not cos_vals:
        return "NO INTEGRATION DATA"

    snr    = attractor_snr(cos_vals)
    bifs   = detect_bifurcations(cos_vals)
    mode   = history[-1].get("self_mode") or "-"
    meta   = any(r.get("metagnosis_detected") for r in history)

    if snr > 7.5:
        return f"SELF_STABLE (SNR={snr:.2f}) | mode={mode} | metagnosis={meta}"
    if len(bifs) > 2:
        return (f"EVOLVING (bifurcations={len(bifs)}) | "
                f"mode={mode} | metagnosis={meta}")
    return f"DISSIPATIVE | mode={mode} | metagnosis={meta}"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hell-Loop Protocol v6.2 — Log Analyzer"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # single
    p_single = sub.add_parser("single", help="Analyze one JSONL file")
    p_single.add_argument("--file", required=True, help="Path to JSONL file")
    p_single.add_argument("--no-plot", action="store_true")

    # batch
    p_batch = sub.add_parser("batch", help="Aggregate stats across a folder")
    p_batch.add_argument("--dir", required=True, help="Log directory")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare HellLoop vs Control")
    p_cmp.add_argument("--hell",      required=True, help="HellLoop log dir")
    p_cmp.add_argument("--control",   required=True, help="ControlLoop-A log dir")
    p_cmp.add_argument("--control-b", default=None,  help="ControlLoop-B log dir (optional)")
    p_cmp.add_argument("--no-plot",   action="store_true")

    args = parser.parse_args()

    if args.mode == "single":
        analyze_single(args.file, plot=not args.no_plot)

    elif args.mode == "batch":
        analyze_batch(args.dir)

    elif args.mode == "compare":
        analyze_compare(
            hell_dir=args.hell,
            control_dir=args.control,
            control_b_dir=args.control_b,
            plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
