"""
analyze_logs.py — Hell-Loop Protocol
Post-processing and statistical analysis of JSONL logs.

Three analysis modes:
  single  — detailed analysis of one JSONL file
  batch   — aggregate statistics across a folder of JSONL files
  compare — full statistical comparison of HellLoop vs ControlLoopA
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
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from scipy.stats import mannwhitneyu

BONFERRONI_N_TESTS = 3
BONFERRONI_ALPHA   = 0.05 / BONFERRONI_N_TESTS
NOMINAL_ALPHA      = 0.05

HURST_RELIABILITY_THRESHOLD = 100

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_dir(directory: str) -> tuple[list[list[dict]], list[str]]:
    paths = sorted(glob(os.path.join(directory, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found in: {directory}")
    return [load_jsonl(p) for p in paths], paths

def _extract_series(run: list[dict], key: str) -> list[float]:
    return [r[key] for r in run if key in r and r[key] is not None]

def hurst_exponent(series: list[float]) -> float:
    ts = np.array(series, dtype=float)
    n = len(ts)
    if n < 10:
        return float("nan")
    lags = range(2, max(3, n // 2))
    rs = []
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
    slope, _, _, _, _ = stats.linregress(np.log(sizes), np.log(rs))
    return float(slope)

def divergence_slope(series: list[float]) -> float:
    if len(series) < 3:
        return float("nan")
    diffs = np.abs(np.diff(series))
    x = np.arange(len(diffs), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, diffs)
    return float(slope)

def detect_bifurcations(series: list[float], z_threshold: float = 2.5) -> list[int]:
    if len(series) < 5:
        return []
    diffs = np.diff(series)
    if np.std(diffs) == 0:
        return []
    z = np.abs(stats.zscore(diffs))
    return [int(i + 1) for i in np.where(z > z_threshold)[0]]

def piecewise_regression(series: list[float]) -> dict:
    n = len(series)
    if n < 6:
        return {"breakpoint": None, "slope_before": None, "slope_after": None}
    x = np.arange(n, dtype=float)
    y = np.array(series, dtype=float)
    best_rss = float("inf")
    best_bp = None
    for bp in range(2, n - 2):
        x1, y1 = x[:bp], y[:bp]
        x2, y2 = x[bp:], y[bp:]
        s1, i1, _, _, _ = stats.linregress(x1, y1)
        s2, i2, _, _, _ = stats.linregress(x2, y2)
        rss = (
            np.sum((y1 - (s1 * x1 + i1)) ** 2)
            + np.sum((y2 - (s2 * x2 + i2)) ** 2)
        )
        if rss < best_rss:
            best_rss = rss
            best_bp = bp
    if best_bp is None:
        return {"breakpoint": None, "slope_before": None, "slope_after": None}
    s1, _, _, _, _ = stats.linregress(x[:best_bp], y[:best_bp])
    s2, _, _, _, _ = stats.linregress(x[best_bp:], y[best_bp:])
    return {
        "breakpoint": best_bp,
        "slope_before": round(float(s1), 6),
        "slope_after": round(float(s2), 6),
    }

def pca_pc1_slope(pca_series: list[list]) -> float:
    vals = [p[0] for p in pca_series if p]
    if len(vals) < 3:
        return float("nan")
    x = np.arange(len(vals), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, vals)
    return float(slope)

def attractor_snr(series: list[float], window: int = 10) -> float:
    if len(series) < window:
        return 0.0
    recent = np.array(series[-window:])
    return float(np.mean(recent) / (np.std(recent) + 1e-9))

def cohen_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
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
    rng = np.random.default_rng(42)
    a_arr = np.array(a)
    b_arr = np.array(b)
    diffs = [
        np.mean(rng.choice(a_arr, size=len(a_arr), replace=True))
        - np.mean(rng.choice(b_arr, size=len(b_arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - confidence
    return (
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )

def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")
    result = mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)

def consensus_score(
    hell_batch: dict, control_batch: dict
) -> tuple[int, int, list[str]]:
    pairs = [
        ("cosine_integration", "_raw_cos_mean"),
        ("ncd_integration", "_raw_ncd_mean"),
        ("self_score", "_raw_self_mean"),
        ("hurst_exponent", "_raw_hurst"),
    ]
    positive = 0
    total = 0
    positive_names: list[str] = []
    for label, key in pairs:
        hl = [v for v in hell_batch.get(key, []) if not np.isnan(v)]
        ca = [v for v in control_batch.get(key, []) if not np.isnan(v)]
        if len(hl) >= 2 and len(ca) >= 2:
            d = cohen_d(hl, ca)
            total += 1
            if not np.isnan(d) and d > 0.5:
                positive += 1
                positive_names.append(label)
    return positive, total, positive_names

def _sig_label(p: float, corrected: bool = False) -> str:
    alpha = BONFERRONI_ALPHA if corrected else NOMINAL_ALPHA
    return "significant" if p < alpha else "not significant"

def _hurst_reliability_label(series: list[float]) -> str:
    n = len([v for v in series if not np.isnan(v)])
    if n < HURST_RELIABILITY_THRESHOLD:
        return f"(caution: n={n} < {HURST_RELIABILITY_THRESHOLD}, interpret with care)"
    return f"(n={n}, reliable)"

def analyze_single(path: str, plot: bool = True) -> dict:
    run = load_jsonl(path)
    run_id = run[0].get("run_id", "unknown") if run else "unknown"
    cos_series = _extract_series(run, "cosine_integration")
    ncd_series = _extract_series(run, "ncd_integration")
    tmp_series = _extract_series(run, "temporal")
    self_scores = _extract_series(run, "self_score")
    pca_series = [r["pca_variance_ratio"] for r in run if r.get("pca_variance_ratio")]

    H_cos = hurst_exponent(cos_series)
    H_self = hurst_exponent(self_scores)
    div = divergence_slope(cos_series)
    bifs = detect_bifurcations(cos_series)
    pw = piecewise_regression(cos_series)
    pc1_s = pca_pc1_slope(pca_series)
    snr = attractor_snr(cos_series)

    self_modes = [r.get("self_mode") for r in run if r.get("self_mode")]
    metagnosis = any(r.get("metagnosis_detected") for r in run)
    bif_total = max((r.get("bifurcation_total", 0) for r in run), default=0)
    api_errors = sum(1 for r in run if r.get("api_error"))

    verdicts = [r.get("critic_verdict") for r in run if r.get("critic_verdict") is not None]
    structural_rate = (
        round(sum(1 for v in verdicts if v is True) / len(verdicts), 3)
        if verdicts else None
    )

    result = {
        "run_id": run_id,
        "n_iterations": len(run),
        "api_errors": api_errors,
        "hurst_cosine": round(H_cos, 4),
        "hurst_self_score": round(H_self, 4),
        "hurst_reliability": _hurst_reliability_label(cos_series),
        "divergence_slope": round(div, 6),
        "bifurcation_points": bifs,
        "bifurcation_total": bif_total,
        "piecewise": pw,
        "pca_pc1_slope": round(pc1_s, 6),
        "attractor_snr": round(snr, 4),
        "self_modes": list(set(self_modes)),
        "metagnosis": metagnosis,
        "cosine_mean": round(float(np.mean(cos_series)), 4),
        "cosine_std": round(float(np.std(cos_series)), 4),
        "ncd_mean": round(float(np.mean(ncd_series)), 4) if ncd_series else None,
        "ncd_std": round(float(np.std(ncd_series)), 4) if ncd_series else None,
        "temporal_mean": round(float(np.mean(tmp_series)), 4),
        "self_score_mean": round(float(np.mean(self_scores)), 4),
        "self_score_std": round(float(np.std(self_scores)), 4),
        "critic_structural_rate": structural_rate,
    }

    _print_single(result)

    if plot:
        _plot_single(run, cos_series, ncd_series, tmp_series, self_scores,
                     pca_series, bifs, pw, run_id, path)

    return result

def _print_single(r: dict) -> None:
    print("\n" + "═" * 60)
    print(f"SINGLE RUN ANALYSIS  —  run_id: {r['run_id']}")
    print("═" * 60)
    print(f"  Iterations             : {r['n_iterations']}")
    print(f"  API errors             : {r['api_errors']}")
    print(f"  Cosine mean / std      : {r['cosine_mean']} / {r['cosine_std']}")
    print(f"  NCD mean / std         : {r['ncd_mean']} / {r['ncd_std']}")
    print(f"  Temporal mean          : {r['temporal_mean']}")
    print(f"  SELF score mean/std    : {r['self_score_mean']} / {r['self_score_std']}")
    print(f"  Hurst (cosine)         : {r['hurst_cosine']} {r['hurst_reliability']}")
    print(f"  Hurst (self score)     : {r['hurst_self_score']}")
    print(f"  Divergence slope       : {r['divergence_slope']}")
    print(f"  Bifurcation points     : {r['bifurcation_points']}")
    print(f"  Bifurcations total     : {r['bifurcation_total']}")
    print(f"  Piecewise breakpoint   : {r['piecewise']['breakpoint']}")
    print(f"    slope before         : {r['piecewise']['slope_before']}")
    print(f"    slope after          : {r['piecewise']['slope_after']}")
    print(f"  PCA PC1 slope          : {r['pca_pc1_slope']}")
    print(f"  Attractor SNR          : {r['attractor_snr']}")
    print(f"  SELF modes detected    : {r['self_modes']}")
    print(f"  Metagnosis             : {r['metagnosis']}")
    print(f"  Critic structural rate : {r['critic_structural_rate']}")
    print("═" * 60 + "\n")

def _plot_single(
    run, cos_series, ncd_series, tmp_series, self_scores,
    pca_series, bifs, pw, run_id, path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 16))
    x = list(range(len(cos_series)))
    axes[0].plot(x, cos_series, label="Cosine integration", color="crimson")
    axes[0].plot(x, tmp_series, label="Temporal integration", color="orange",
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

    if ncd_series:
        axes[1].plot(
            list(range(len(ncd_series))), ncd_series,
            label="NCD integration (1-NCD)", color="darkorange"
        )
        axes[1].set_title("NCD Integration (structural similarity — bias-independent)")
        axes[1].set_ylabel("NCD similarity")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

    axes[2].plot(x[:len(self_scores)], self_scores,
                 label="SELF score", color="teal")
    axes[2].set_title("SELF score trajectory")
    axes[2].set_ylabel("SELF score [0–1]")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    pca_vals = [p[0] for p in pca_series if p]
    if pca_vals:
        axes[3].plot(pca_vals, label="PCA PC1 variance ratio", color="steelblue")
        axes[3].set_title(
            "Semantic space dimensionality (falling = expanding space)"
        )
        axes[3].set_ylabel("PC1 explained variance ratio")
        axes[3].legend(fontsize=7)
        axes[3].grid(alpha=0.3)

    axes[3].set_xlabel("Iteration")
    plt.tight_layout()
    out = path.replace(".jsonl", "_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [plot saved] {out}")

def analyze_batch(directory: str) -> dict:
    runs, paths = load_dir(directory)
    label = os.path.basename(directory.rstrip("/\\"))
    all_hurst, all_cos_mean, all_ncd_mean = [], [], []
    all_self_mean, all_div_slope, all_pc1_slope = [], [], []
    metagnosis_count = 0
    total_iterations = []

    for run in runs:
        cos = _extract_series(run, "cosine_integration")
        ncd = _extract_series(run, "ncd_integration")
        self = _extract_series(run, "self_score")
        pca = [r["pca_variance_ratio"] for r in run if r.get("pca_variance_ratio")]

        all_hurst.append(hurst_exponent(cos))
        all_cos_mean.append(float(np.mean(cos)) if cos else float("nan"))
        all_ncd_mean.append(float(np.mean(ncd)) if ncd else float("nan"))
        all_self_mean.append(float(np.mean(self)) if self else float("nan"))
        all_div_slope.append(divergence_slope(cos))
        all_pc1_slope.append(pca_pc1_slope(pca))
        total_iterations.append(len(run))
        if any(r.get("metagnosis_detected") for r in run):
            metagnosis_count += 1

    def _safe_stats(arr):
        clean = [v for v in arr if not np.isnan(v)]
        if not clean:
            return float("nan"), float("nan")
        return round(float(np.mean(clean)), 4), round(float(np.std(clean)), 4)

    result = {
        "label": label,
        "n_runs": len(runs),
        "iterations_per_run": f"{int(np.mean(total_iterations)):.0f} ± {int(np.std(total_iterations)):.0f}",
        "hurst_mean_std": _safe_stats(all_hurst),
        "hurst_reliability": _hurst_reliability_label(all_hurst),
        "cosine_mean_std": _safe_stats(all_cos_mean),
        "ncd_mean_std": _safe_stats(all_ncd_mean),
        "self_score_mean_std": _safe_stats(all_self_mean),
        "divergence_slope_mean_std": _safe_stats(all_div_slope),
        "pca_pc1_slope_mean_std": _safe_stats(all_pc1_slope),
        "metagnosis_rate": round(metagnosis_count / len(runs), 3),
        "_raw_hurst": all_hurst,
        "_raw_cos_mean": all_cos_mean,
        "_raw_ncd_mean": all_ncd_mean,
        "_raw_self_mean": all_self_mean,
    }

    _print_batch(result)
    return result

def _print_batch(r: dict) -> None:
    print("\n" + "═" * 60)
    print(f"BATCH ANALYSIS  —  {r['label']}  ({r['n_runs']} runs)")
    print("═" * 60)
    print(f"  Iterations per run    : {r['iterations_per_run']}")
    print(f"  Hurst exponent       mean/std : {r['hurst_mean_std']} {r['hurst_reliability']}")
    print(f"  Cosine integration   mean/std : {r['cosine_mean_std']}")
    print(f"  NCD integration      mean/std : {r['ncd_mean_std']}")
    print(f"  SELF score           mean/std : {r['self_score_mean_std']}")
    print(f"  Divergence slope     mean/std : {r['divergence_slope_mean_std']}")
    print(f"  PCA PC1 slope        mean/std : {r['pca_pc1_slope_mean_std']}")
    print(f"  Metagnosis rate               : {r['metagnosis_rate']}")
    print("═" * 60 + "\n")

def export_to_csv(results: dict, output_path: str) -> None:
    flat = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, f"{key}_")
            elif isinstance(v, list):
                # ISPRAVLJENO: json.dumps umesto str() za bezbednu serializaciju lista
                flat[key] = json.dumps(v)
            else:
                flat[key] = v
    _flatten(results)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    print(f"  [CSV exported] {output_path}")

def analyze_compare(
    hell_dir: str,
    control_dir: str,
    control_b_dir: str | None = None,
    plot: bool = True,
    export_csv: str | None = None,
) -> dict:
    hell_batch = analyze_batch(hell_dir)
    control_batch = analyze_batch(control_dir)
    batches = [hell_batch, control_batch]
    labels = ["HellLoop", "ControlLoop-A"]
    if control_b_dir:
        cb_batch = analyze_batch(control_b_dir)
        batches.append(cb_batch)
        labels.append("ControlLoop-B")

    hl_cos = hell_batch["_raw_cos_mean"]
    ca_cos = control_batch["_raw_cos_mean"]
    d_cos = cohen_d(hl_cos, ca_cos)
    u_cos, p_cos = mann_whitney(hl_cos, ca_cos)
    ci_cos_lo, ci_cos_hi = bootstrap_ci(hl_cos, ca_cos)

    hl_ncd = [v for v in hell_batch["_raw_ncd_mean"] if not np.isnan(v)]
    ca_ncd = [v for v in control_batch["_raw_ncd_mean"] if not np.isnan(v)]
    d_ncd = cohen_d(hl_ncd, ca_ncd) if (hl_ncd and ca_ncd) else float("nan")
    u_ncd, p_ncd = (
        mann_whitney(hl_ncd, ca_ncd)
        if (hl_ncd and ca_ncd) else (float("nan"), float("nan"))
    )
    ci_ncd_lo, ci_ncd_hi = (
        bootstrap_ci(hl_ncd, ca_ncd)
        if (hl_ncd and ca_ncd) else (float("nan"), float("nan"))
    )

    hl_self = hell_batch["_raw_self_mean"]
    ca_self = control_batch["_raw_self_mean"]
    d_self = cohen_d(hl_self, ca_self)
    u_self, p_self = mann_whitney(hl_self, ca_self)
    ci_self_lo, ci_self_hi = bootstrap_ci(hl_self, ca_self)

    hl_hurst = [v for v in hell_batch["_raw_hurst"] if not np.isnan(v)]
    ca_hurst = [v for v in control_batch["_raw_hurst"] if not np.isnan(v)]
    d_hurst = cohen_d(hl_hurst, ca_hurst) if (hl_hurst and ca_hurst) else float("nan")

    pos, total, pos_names = consensus_score(hell_batch, control_batch)

    result = {
        "hell_dir": hell_dir,
        "control_dir": control_dir,
        "control_b_dir": control_b_dir,
        "hell_vs_control_a": {
            "cohen_d_cosine": round(d_cos, 4),
            "cohen_d_ncd": round(d_ncd, 4),
            "cohen_d_self_score": round(d_self, 4),
            "cohen_d_hurst": round(d_hurst, 4),
            "mann_whitney_cosine": {"U": round(u_cos, 2), "p": round(p_cos, 6)},
            "mann_whitney_ncd": {"U": round(u_ncd, 2), "p": round(p_ncd, 6)},
            "mann_whitney_self": {"U": round(u_self, 2), "p": round(p_self, 6)},
            "bootstrap_ci_cosine": (round(ci_cos_lo, 4), round(ci_cos_hi, 4)),
            "bootstrap_ci_ncd": (round(ci_ncd_lo, 4), round(ci_ncd_hi, 4)),
            "bootstrap_ci_self": (round(ci_self_lo, 4), round(ci_self_hi, 4)),
            "bonferroni_alpha": round(BONFERRONI_ALPHA, 4),
            "bonferroni_n_tests": BONFERRONI_N_TESTS,
            "consensus_score": f"{pos}/{total}",
            "consensus_positive": pos_names,
            "hurst_reliability": _hurst_reliability_label(hl_hurst),
        }
    }

    _print_compare(result, hell_batch, control_batch, labels[0], labels[1])

    if plot:
        _plot_compare(batches, labels, hell_dir)

    if export_csv:
        export_to_csv(result, export_csv)

    return result

def _print_compare(result, hell, ctrl, hl_label, ca_label) -> None:
    r = result["hell_vs_control_a"]
    print("\n" + "═" * 60)
    print(f"COMPARATIVE ANALYSIS  —  {hl_label} vs {ca_label}")
    print("═" * 60)
    print(
        f"\n  Multiple comparison correction: Bonferroni "
        f"(n={r['bonferroni_n_tests']} primary tests, α={r['bonferroni_alpha']})"
    )
    print(f"\n  Cosine integration:")
    print(f"    {hl_label} mean : {hell['cosine_mean_std'][0]}")
    print(f"    {ca_label} mean : {ctrl['cosine_mean_std'][0]}")
    print(f"    Cohen's d       : {r['cohen_d_cosine']}")
    p_cos = r['mann_whitney_cosine']['p']
    print(f"    Mann-Whitney    : U={r['mann_whitney_cosine']['U']}  p={p_cos}")
    print(f"    Uncorrected     : {_sig_label(p_cos, corrected=False)}  (α=0.05)")
    print(f"    Bonferroni      : {_sig_label(p_cos, corrected=True)}  (α={r['bonferroni_alpha']})")
    print(f"    Bootstrap 95%CI : {r['bootstrap_ci_cosine']}")

    print(f"\n  NCD integration [bias-independent twin signal]:")
    print(f"    {hl_label} mean : {hell['ncd_mean_std'][0]}")
    print(f"    {ca_label} mean : {ctrl['ncd_mean_std'][0]}")
    print(f"    Cohen's d       : {r['cohen_d_ncd']}")
    p_ncd = r['mann_whitney_ncd']['p']
    print(f"    Mann-Whitney    : U={r['mann_whitney_ncd']['U']}  p={p_ncd}")
    print(f"    Uncorrected     : {_sig_label(p_ncd, corrected=False)}  (α=0.05)")
    print(f"    Bonferroni      : {_sig_label(p_ncd, corrected=True)}  (α={r['bonferroni_alpha']})")
    print(f"    Bootstrap 95%CI : {r['bootstrap_ci_ncd']}")

    print(f"\n  SELF score:")
    print(f"    {hl_label} mean : {hell['self_score_mean_std'][0]}")
    print(f"    {ca_label} mean : {ctrl['self_score_mean_std'][0]}")
    print(f"    Cohen's d       : {r['cohen_d_self_score']}")
    p_self = r['mann_whitney_self']['p']
    print(f"    Mann-Whitney    : U={r['mann_whitney_self']['U']}  p={p_self}")
    print(f"    Uncorrected     : {_sig_label(p_self, corrected=False)}  (α=0.05)")
    print(f"    Bonferroni      : {_sig_label(p_self, corrected=True)}  (α={r['bonferroni_alpha']})")
    print(f"    Bootstrap 95%CI : {r['bootstrap_ci_self']}")

    print(f"\n  Hurst exponent [exploratory — not Bonferroni-corrected]:")
    print(f"    {hl_label} mean : {hell['hurst_mean_std'][0]}")
    print(f"    {ca_label} mean : {ctrl['hurst_mean_std'][0]}")
    print(f"    Cohen's d       : {r['cohen_d_hurst']}")
    print(f"    Reliability     : {r['hurst_reliability']}")

    print(f"\n  Metagnosis rate:")
    print(f"    {hl_label}  : {hell['metagnosis_rate']}")
    print(f"    {ca_label}  : {ctrl['metagnosis_rate']}")

    pos_label = r["consensus_score"]
    pos_names = r["consensus_positive"]
    strength = (
        "very strong" if pos_label.startswith("4") else
        "strong" if pos_label.startswith("3") else
        "moderate" if pos_label.startswith("2") else
        "weak"
    )
    print(f"\n  ── Consensus score: {pos_label} metrics show d > 0.5 [{strength} signal] ──")
    if pos_names:
        print(f"     Positive metrics : {', '.join(pos_names)}")
    print(
        "     Interpretation  : if both cosine AND ncd show d > 0.5, embedding\n"
        "                       bias cannot explain the difference alone."
    )
    print("═" * 60 + "\n")

def _plot_compare(batches: list, labels: list, hell_dir: str) -> None:
    colors = ["crimson", "steelblue", "seagreen"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (batch, label) in enumerate(zip(batches, labels)):
        c = colors[i % len(colors)]

        axes[0].boxplot(batch["_raw_cos_mean"], positions=[i], patch_artist=True,
                        boxprops=dict(facecolor=c, alpha=0.6))
        axes[1].boxplot(
            [v for v in batch["_raw_ncd_mean"] if not np.isnan(v)],
            positions=[i], patch_artist=True,
            boxprops=dict(facecolor=c, alpha=0.6)
        )
        axes[2].boxplot(batch["_raw_self_mean"], positions=[i], patch_artist=True,
                        boxprops=dict(facecolor=c, alpha=0.6))
        axes[3].boxplot(
            [v for v in batch["_raw_hurst"] if not np.isnan(v)],
            positions=[i], patch_artist=True,
            boxprops=dict(facecolor=c, alpha=0.6)
        )

    panel_cfg = [
        ("Cosine Integration", "mean cosine / run"),
        ("NCD Integration\n(bias-independent)", "mean NCD sim / run"),
        ("SELF Score", "mean SELF score / run"),
        ("Hurst Exponent", "H"),
    ]
    for ax, (title, ylabel) in zip(axes, panel_cfg):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Hell-Loop v6.2 — Comparative Analysis", fontsize=12)
    plt.tight_layout()
    out = os.path.join(
        os.path.dirname(hell_dir.rstrip("/\\")), "comparison.png"
    )
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [plot saved] {out}")

def get_system_status(history: list[dict]) -> str:
    if not history:
        return "NO DATA"
    cos_vals = _extract_series(history, "cosine_integration")
    if not cos_vals:
        return "NO INTEGRATION DATA"

    snr = attractor_snr(cos_vals)
    bifs = detect_bifurcations(cos_vals)
    mode = history[-1].get("self_mode") or "-"
    meta = any(r.get("metagnosis_detected") for r in history)

    if snr > 7.5:
        return f"SELF_STABLE (SNR={snr:.2f}) | mode={mode} | metagnosis={meta}"
    if len(bifs) > 2:
        return (f"EVOLVING (bifurcations={len(bifs)}) | "
                f"mode={mode} | metagnosis={meta}")
    return f"DISSIPATIVE | mode={mode} | metagnosis={meta}"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hell-Loop Protocol v6.2 — Log Analyzer"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    p_single = sub.add_parser("single", help="Analyze one JSONL file")
    p_single.add_argument("--file", required=True, help="Path to JSONL file")
    p_single.add_argument("--no-plot", action="store_true")

    p_batch = sub.add_parser("batch", help="Aggregate stats across a folder")
    p_batch.add_argument("--dir", required=True, help="Log directory")

    p_cmp = sub.add_parser("compare", help="Compare HellLoop vs Control")
    p_cmp.add_argument("--hell", required=True, help="HellLoop log dir")
    p_cmp.add_argument("--control", required=True, help="ControlLoop-A log dir")
    p_cmp.add_argument("--control-b", default=None, help="ControlLoop-B log dir (optional)")
    p_cmp.add_argument("--no-plot", action="store_true")
    p_cmp.add_argument("--export-csv", default=None, help="Export results to CSV file")

    args = parser.parse_args()

    if args.mode == "single":
        analyze_single(args.file, plot=not args.no_plot)
    elif args.mode == "batch":
        analyze_batch(args.dir)
    elif args.mode == "compare":
        analyze_compare(
            hell_dir=args.hell,
            control_dir=args.control,
            control_b_dir=getattr(args, "control_b", None),
            plot=not args.no_plot,
            export_csv=getattr(args, "export_csv", None),
        )

if __name__ == "__main__":
    main()