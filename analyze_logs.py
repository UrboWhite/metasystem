import numpy as np
from scipy import stats


def analyze_attractor_stability(series):
    """
    Signal-to-Noise Ratio for the resonance attractor.
    High ratio = high integration with low fluctuation (Stability).
    """
    if len(series) < 10:
        return 0.0
    recent = np.array(series[-10:])
    return float(np.mean(recent) / (np.std(recent) + 1e-6))


def detect_bifurcations(series):
    """Identifies sudden ontological shifts via Z-score analysis."""
    if len(series) < 5:
        return 0
    diffs = np.diff(series)
    if np.std(diffs) == 0:
        return 0
    z_scores = np.abs(stats.zscore(diffs))
    return int(np.sum(z_scores > 2.0))


def get_system_status(history):
    """
    FIX: Now uses cosine integration (legacy_integration) — the same signal
    that HellLoop._check_self() uses for SELF detection. Previously used
    ncd_integration, which created two inconsistent definitions of 'stability'.

    ControlLoop records use key 'integration' (no 'legacy_' prefix).
    Both are handled transparently.
    """
    if not history:
        return "NO DATA"

    # Support both HellLoop and ControlLoop record formats
    first = history[0]
    if "legacy_integration" in first:
        cos_vals = [r["legacy_integration"] for r in history]
    elif "integration" in first:
        cos_vals = [r["integration"] for r in history]
    else:
        return "UNKNOWN FORMAT"

    # Keep NCD available as secondary diagnostic (not for status decision)
    ncd_vals = [r.get("ncd_integration", 0.0) for r in history]

    stability    = analyze_attractor_stability(cos_vals)
    bifurcations = detect_bifurcations(cos_vals)
    ncd_mean     = float(np.mean(ncd_vals)) if ncd_vals else 0.0

    # SELF_STABLE: high cosine coherence, low variance
    if stability > 7.5:
        return f"SELF_STABLE (Attractor Lock) | NCD_mean={ncd_mean:.3f}"

    # EVOLVING: sudden semantic phase transitions
    if bifurcations > 2:
        return f"EVOLVING (Phase Transition, bifurcations={bifurcations}) | NCD_mean={ncd_mean:.3f}"

    return f"DISSIPATIVE (High Entropy) | NCD_mean={ncd_mean:.3f}"
