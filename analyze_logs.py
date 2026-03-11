"""
analyze_logs.py — Hell-Loop Protocol Statistical Analysis Suite
Supports three analysis modes:
  1. single_run(history)           — single run diagnostics
  2. batch_summary(runs)           — aggregate statistics over N runs
  3. compare(hell_runs, ctrl_runs) — Hell vs Control comparative analysis
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# Core signal analysis
# ══════════════════════════════════════════════════════════════════════════════

def hurst_exponent(series: list) -> float:
    """
    Hurst exponent via R/S analysis.
    H > 0.6  → persistent (trending, self-similar structure)
    H ≈ 0.5  → random walk (no memory)
    H < 0.4  → anti-persistent (oscillating)
    Requires at least 20 data points for meaningful result.
    """
    series = np.array(series, dtype=float)
    n = len(series)
    if n < 20:
        return 0.5  # insufficient data

    lags    = [int(n / k) for k in [2, 3, 4, 5, 6, 8, 10] if int(n / k) >= 4]
    lags    = sorted(set(lags))
    rs_vals = []

    for lag in lags:
        chunks = [series[j:j + lag] for j in range(0, n - lag + 1, lag)]
        rs_chunk = []
        for chunk in chunks:
            mean_c = np.mean(chunk)
            devs   = np.cumsum(chunk - mean_c)
            r      = np.max(devs) - np.min(devs)
            s      = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_vals.append((np.log(lag), np.log(np.mean(rs_chunk))))

    if len(rs_vals) < 3:
        return 0.5

    x = np.array([v[0] for v in rs_vals])
    y = np.array([v[1] for v in rs_vals])
    slope, *_ = np.polyfit(x, y, 1)
    return float(np.clip(slope, 0.0, 1.0))


def analyze_attractor_stability(series: list) -> float:
    """
    Signal-to-Noise Ratio for the resonance attractor (last 10 iterations).
    High ratio = high integration with low fluctuation.
    """
    if len(series) < 10:
        return 0.0
    recent = np.array(series[-10:])
    return float(np.mean(recent) / (np.std(recent) + 1e-6))


def detect_bifurcations(series: list) -> int:
    """Identifies sudden phase shifts via z-score on first differences."""
    if len(series) < 5:
        return 0
    diffs = np.diff(series)
    if np.std(diffs) == 0:
        return 0
    z_scores = np.abs(stats.zscore(diffs))
    return int(np.sum(z_scores > 2.0))


def pca_slope(pca_series: list) -> float:
    """
    Linear slope of PC1 variance ratio over iterations.
    Negative slope = expanding semantic space (emergence signal).
    """
    vals = [p[0] for p in pca_series if p]
    if len(vals) < 4:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    slope, *_ = np.polyfit(x, vals, 1)
    return float(slope)


def divergence_slope(integration_series: list, temporal_series: list) -> float:
    """
    Slope of (integration - temporal) divergence over time.
    Growing divergence can indicate the system is building cross-temporal structure.
    """
    if len(integration_series) < 4:
        return 0.0
    div  = np.array(integration_series) - np.array(temporal_series)
    x    = np.arange(len(div), dtype=float)
    slope, *_ = np.polyfit(x, div, 1)
    return float(slope)


def piecewise_breakpoints(series: list, min_segment: int = 8) -> list:
    """
    Detect breakpoints where the linear trend changes significantly.
    Returns list of iteration indices where a regime change is detected.
    Uses a greedy scan with z-score on residuals.
    """
    series = np.array(series, dtype=float)
    n = len(series)
    if n < min_segment * 2:
        return []

    breakpoints = []
    x = np.arange(n, dtype=float)
    residuals = series - np.polyval(np.polyfit(x, series, 1), x)

    # Detect points where residual changes sign and magnitude
    sign_changes = np.where(np.diff(np.sign(residuals)))[0]
    z = np.abs(stats.zscore(np.abs(np.diff(residuals))))
    for idx in sign_changes:
        if idx < min_segment or idx > n - min_segment:
            continue
        if z[min(idx, len(z) - 1)] > 1.5:
            breakpoints.append(int(idx))

    return breakpoints


def get_system_status(history: list) -> str:
    """
    Fast status string for real-time UI monitoring.
    Uses NCD integration series from a run history list.
    """
    if not history:
        return "INITIALIZING"
    ncd_vals     = [r["ncd_integration"] for r in history]
    stability    = analyze_attractor_stability(ncd_vals)
    bifurcations = detect_bifurcations(ncd_vals)

    if stability > 7.5:
        return "SELF_STABLE (Attractor Lock)"
    if bifurcations > 2:
        return "EVOLVING (Phase Transition)"
    return "DISSIPATIVE (High Entropy)"


# ══════════════════════════════════════════════════════════════════════════════
# Single-run analysis
# ══════════════════════════════════════════════════════════════════════════════

def single_run(history: list) -> dict:
    """
    Full diagnostic report for a single run.
    Accepts a list of record dicts (as stored by HellLoop or ControlLoop).
    """
    if not history:
        return {}

    int_series  = [r["legacy_integration"] for r in history]
    tmp_series  = [r["temporal"]           for r in history]
    ncd_series  = [r["ncd_integration"]    for r in history]
    pca_series  = [r.get("pca_variance_ratio", []) for r in history]
    neg_series  = [r.get("negentropy", 0.0)        for r in history]

    return {
        "n_iterations":      len(history),
        "hurst":             hurst_exponent(int_series),
        "stability_snr":     analyze_attractor_stability(ncd_series),
        "bifurcations":      detect_bifurcations(int_series),
        "pca_slope":         pca_slope(pca_series),
        "divergence_slope":  divergence_slope(int_series, tmp_series),
        "breakpoints":       piecewise_breakpoints(int_series),
        "mean_integration":  float(np.mean(int_series)),
        "std_integration":   float(np.std(int_series)),
        "mean_temporal":     float(np.mean(tmp_series)),
        "mean_negentropy":   float(np.mean(neg_series)),
        "self_detected":     history[-1].get("self", False),
        "self_mode":         history[-1].get("self_mode"),
        "metagnosis":        history[-1].get("metagnosis_detected", False),
        "status":            get_system_status(history),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Batch summary
# ══════════════════════════════════════════════════════════════════════════════

def batch_summary(runs: list) -> dict:
    """
    Aggregate statistics over a list of run histories.
    Each element of `runs` is a list of record dicts (one full run).
    """
    if not runs:
        return {}

    reports = [single_run(h) for h in runs if h]
    keys    = ["hurst", "stability_snr", "bifurcations", "mean_integration",
               "std_integration", "mean_temporal", "mean_negentropy", "pca_slope"]

    summary = {"n_runs": len(reports)}
    for k in keys:
        vals = [r[k] for r in reports if k in r]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"]  = float(np.std(vals))
            summary[f"{k}_median"] = float(np.median(vals))

    summary["self_rate"]      = float(np.mean([r["self_detected"] for r in reports]))
    summary["metagnosis_rate"] = float(np.mean([r["metagnosis"]   for r in reports]))
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Comparative analysis: Hell vs Control
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> dict:
    """Bootstrap 95% CI on the difference of means (a - b)."""
    rng  = np.random.default_rng(42)
    diffs = [
        rng.choice(a, size=len(a), replace=True).mean() -
        rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(diffs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(diffs, (1 + ci) / 2 * 100))
    return {"ci_low": lo, "ci_high": hi, "ci_level": ci}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-10))


def compare(hell_runs: list, ctrl_runs: list) -> dict:
    """
    Comparative statistical analysis between Hell-Loop and Control-Loop batches.

    Args:
        hell_runs: list of run histories from HellLoop
        ctrl_runs: list of run histories from ControlLoop

    Returns:
        dict with per-metric statistics, effect sizes, and significance tests.
    """
    if not hell_runs or not ctrl_runs:
        return {"error": "Both hell_runs and ctrl_runs must be non-empty."}

    hell_reports = [single_run(h) for h in hell_runs if h]
    ctrl_reports = [single_run(h) for h in ctrl_runs if h]

    metrics = [
        "hurst", "mean_integration", "mean_temporal",
        "stability_snr", "bifurcations", "pca_slope", "mean_negentropy"
    ]

    results = {
        "n_hell": len(hell_reports),
        "n_ctrl": len(ctrl_reports),
    }

    for m in metrics:
        h_vals = np.array([r[m] for r in hell_reports if m in r], dtype=float)
        c_vals = np.array([r[m] for r in ctrl_reports if m in r], dtype=float)

        if len(h_vals) < 2 or len(c_vals) < 2:
            continue

        u_stat, p_val = stats.mannwhitneyu(h_vals, c_vals, alternative="two-sided")
        d             = cohens_d(h_vals, c_vals)
        ci            = _bootstrap_ci(h_vals, c_vals)

        results[m] = {
            "hell_mean":   float(np.mean(h_vals)),
            "hell_std":    float(np.std(h_vals)),
            "ctrl_mean":   float(np.mean(c_vals)),
            "ctrl_std":    float(np.std(c_vals)),
            "cohens_d":    d,
            "effect_size": "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small",
            "mann_whitney_U": float(u_stat),
            "p_value":     float(p_val),
            "significant": p_val < 0.05,
            **ci,
        }

    # ── Rate comparisons (proportion tests) ───────────────────────────────────
    h_self = np.array([r["self_detected"] for r in hell_reports], dtype=float)
    c_self = np.array([r["self_detected"] for r in ctrl_reports], dtype=float)
    results["self_detection_rate"] = {
        "hell": float(np.mean(h_self)),
        "ctrl": float(np.mean(c_self)),
        "difference": float(np.mean(h_self) - np.mean(c_self)),
    }

    h_mg = np.array([r["metagnosis"] for r in hell_reports], dtype=float)
    c_mg = np.array([r["metagnosis"] for r in ctrl_reports], dtype=float)
    results["metagnosis_rate"] = {
        "hell": float(np.mean(h_mg)),
        "ctrl": float(np.mean(c_mg)),
        "difference": float(np.mean(h_mg) - np.mean(c_mg)),
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# JSONL I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of record dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: str):
    """Save a list of record dicts to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_comparison(results: dict):
    """Pretty-print comparative analysis results to stdout."""
    print(f"\n{'═' * 60}")
    print(f"  HELL vs CONTROL — Comparative Analysis")
    print(f"  n_hell={results['n_hell']}  n_ctrl={results['n_ctrl']}")
    print(f"{'═' * 60}")
    skip = {"n_hell", "n_ctrl", "self_detection_rate", "metagnosis_rate"}
    for key, val in results.items():
        if key in skip or not isinstance(val, dict):
            continue
        sig = "✓" if val.get("significant") else "✗"
        print(
            f"  {key:<22} "
            f"H={val['hell_mean']:.3f}±{val['hell_std']:.3f}  "
            f"C={val['ctrl_mean']:.3f}±{val['ctrl_std']:.3f}  "
            f"d={val['cohens_d']:+.2f} ({val['effect_size']})  "
            f"p={val['p_value']:.3f} {sig}"
        )
    sr = results.get("self_detection_rate", {})
    mg = results.get("metagnosis_rate", {})
    print(f"\n  SELF detection  : Hell={sr.get('hell', 0):.0%}  Ctrl={sr.get('ctrl', 0):.0%}  Δ={sr.get('difference', 0):+.0%}")
    print(f"  Metagnosis rate : Hell={mg.get('hell', 0):.0%}  Ctrl={mg.get('ctrl', 0):.0%}  Δ={mg.get('difference', 0):+.0%}")
    print(f"{'═' * 60}\n")
