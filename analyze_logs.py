import numpy as np
from scipy import stats

def analyze_attractor_stability(series):
    """Calculates Signal-to-Noise Ratio for the resonance attractor."""
    if len(series) < 10:
        return 0.0
    recent = np.array(series[-10:])
    # High ratio = high integration with low fluctuation (Stability)
    stability = np.mean(recent) / (np.std(recent) + 1e-6)
    return float(stability)

def detect_bifurcations(series):
    """Identifies sudden ontological shifts via Z-score analysis."""
    if len(series) < 5:
        return 0
    diffs = np.diff(series)
    if np.std(diffs) == 0: return 0
    z_scores = np.abs(stats.zscore(diffs))
    return int(np.sum(z_scores > 2.0))

def get_system_status(history):
    ncd_vals = [r["ncd_integration"] for r in history]
    stability = analyze_attractor_stability(ncd_vals)
    bifurcations = detect_bifurcations(ncd_vals)

    if stability > 7.5:
        return "SELF_STABLE (Attractor Lock)"
    if bifurcations > 2:
        return "EVOLVING (Phase Transition)"
    return "DISSIPATIVE (High Entropy)"
