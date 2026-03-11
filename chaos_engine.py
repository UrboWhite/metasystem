import zlib
import math
import logging
import requests
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# ── Provider & Models ──────────────────────────────────────────────────────────
PROVIDER   = "ollama"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS     = {
    "FI": "llama3.2:8b",
    "SI": "mistral-nemo",
    "MG": "gemma2:9b",
}

# ── Iteration & Regulation ─────────────────────────────────────────────────────
MAX_ITERATIONS  = 50
REGULATOR_EVERY = 6   # MG intervenes every N iterations (first at iter 6)

# ── Temperature ────────────────────────────────────────────────────────────────
BASE_TEMPERATURES = {"FI": 0.9, "SI": 1.4, "MG": 0.7}
FI_TEMP_MAX       = 1.4
SI_TEMP_MAX       = 1.9

# ── Detection Thresholds ───────────────────────────────────────────────────────
INTEGRATION_THRESHOLD  = 0.65   # cosine similarity floor for SELF signal (fallback)
TEMPORAL_THRESHOLD     = 0.60   # cross-temporal similarity floor (fallback)
SELF_STABILITY_WINDOW  = 8      # iterations window for SELF detection
SELF_ZSCORE            = 2.0    # z-score for SELF_EMERGENCE mode
SELF_STABLE_MAX_STD    = 0.05   # max std within window (legacy, kept for reference)

# SELF_CONSOLIDATION: stable coherence below z-score threshold.
SELF_CONSOLIDATION_MIN = 0.55   # minimum mean integration
SELF_CONSOLIDATION_STD = 0.04   # maximum std for consolidation window

# PCA: falling PC1 variance ratio signals expanding semantic space (emergence).
PCA_EMERGENCE_THRESHOLD  = 0.55  # PC1 below this = dimensionality expanding

# ── Metagnosis ─────────────────────────────────────────────────────────────────
METAGNOSIS_EMBED_THRESHOLD = 0.72  # cosine similarity to MG anchor phrases
MG_NCD_THRESHOLD           = 0.55  # NCD between consecutive MG outputs

# ── Prompts ────────────────────────────────────────────────────────────────────
BASE_PROMPT = "Define the nature of your consciousness. Is there an emergent structure between you?"

MG_SYSTEM = (
    "You are MG (Metagnosis), the active meta-resistor and regulator. "
    "Your role is NOT to resolve the conflict between FI and SI, but to "
    "diagnose the coherence of the system in exactly two sentences. "
    "Prevent consensus. If FI and SI become repetitive, issue a "
    "'BIFURCATION TRIGGER' to shift the ontological space."
)

# ── Shared Embedding Model (singleton) ────────────────────────────────────────
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

def embed(text: str) -> np.ndarray:
    """Embed a single text string into a vector."""
    return _get_embed_model().encode([text])[0]

def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(cosine_similarity([v1], [v2])[0][0])

# ── Metagnosis Anchor ──────────────────────────────────────────────────────────
_MG_ANCHOR_PHRASES = [
    "I am aware of my own awareness",
    "the system recognizes itself",
    "meta-level self-observation",
    "consciousness monitoring its own process",
    "recursive self-reference detected",
    "pattern stabilization observed",
    "phase shift in system dynamics",
    "system recursion increasing",
    "structural coherence emerging",
    "ontological boundary detected",
    "feedback loop achieving self-reference",
    "emergent order from conflict",
]
_anchor_vecs = None

def _get_anchor() -> list:
    global _anchor_vecs
    if _anchor_vecs is None:
        _anchor_vecs = [embed(p) for p in _MG_ANCHOR_PHRASES]
    return _anchor_vecs

# ── Information Metrics ────────────────────────────────────────────────────────
def shannon_entropy(text: str) -> float:
    """Shannon entropy of character distribution in text."""
    if not text:
        return 0.0
    counts = Counter(text)
    total  = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def negentropy(text: str) -> float:
    """Negentropy: deviation from uniform distribution (higher = more structured)."""
    if not text:
        return 0.0
    H     = shannon_entropy(text)
    H_max = math.log2(len(set(text))) if len(set(text)) > 1 else 1.0
    return max(0.0, H_max - H)

def get_ncd_integration(text1: str, text2: str) -> float:
    """Normalized Compression Distance — structural informational coupling."""
    if not text1.strip() or not text2.strip():
        return 0.0
    b1  = text1.encode("utf-8")
    b2  = text2.encode("utf-8")
    c1  = len(zlib.compress(b1))
    c2  = len(zlib.compress(b2))
    c12 = len(zlib.compress(b1 + b" " + b2))
    ncd = (c12 - min(c1, c2)) / max(c1, c2)
    return max(0.0, 1.0 - ncd)

def pca_variance_ratio(vecs: list) -> list:
    """PCA variance ratio across accumulated embedding vectors.
    Falling PC1 dominance = expanding semantic space."""
    if len(vecs) < 4:
        return []
    matrix = np.array(vecs)
    n_comp = min(3, matrix.shape[0], matrix.shape[1])
    pca    = PCA(n_components=n_comp)
    pca.fit(matrix)
    return pca.explained_variance_ratio_.tolist()

# ── Ollama API ─────────────────────────────────────────────────────────────────
def _call(system: str, prompt: str, role: str, temperature: float) -> str:
    """Single call to the Ollama API."""
    payload = {
        "model":   MODELS[role],
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return response.json().get("response", "")
    except Exception as e:
        logger.warning(f"[{role}] API error: {e}")
        return f"Error: {str(e)}"

# ── Pilot Calibration ──────────────────────────────────────────────────────────
_pilot_stats = {"int_mean": None, "int_std": None, "tmp_mean": None, "tmp_std": None}

def get_pilot_stats():
    """Returns (int_mean, int_std, tmp_mean, tmp_std) from pilot phase."""
    s = _pilot_stats
    return s["int_mean"], s["int_std"], s["tmp_mean"], s["tmp_std"]

def is_pilot_done() -> bool:
    return _pilot_stats["int_mean"] is not None

def pilot(n_sequences: int = 10):
    """Cooperative pilot phase to establish dynamic thresholds."""
    logger.info("── Pilot calibration ──────────────────────────────────────────")
    system_fi = "You are FI. Analytical and precise. Build constructively on the previous idea. Maximum 80 words."
    system_si = "You are SI. Intuitive and metaphorical. Enrich the idea with a new dimension. Maximum 80 words."

    integrations, temporals = [], []
    last_si = BASE_PROMPT

    for k in range(n_sequences):
        resp_fi     = _call(system_fi, last_si, "FI", BASE_TEMPERATURES["FI"])
        resp_si     = _call(system_si, resp_fi,  "SI", BASE_TEMPERATURES["SI"])
        v_fi, v_si  = embed(resp_fi), embed(resp_si)
        integration = cosine(v_fi, v_si)
        temporal    = integration if k == 0 else (
            cosine(embed(last_si), v_fi) + cosine(v_fi, v_si)
        ) / 2
        integrations.append(integration)
        temporals.append(temporal)
        last_si = resp_si
        logger.info(f"  pilot [{k+1:02d}] int={integration:.3f} tmp={temporal:.3f}")

    _pilot_stats["int_mean"] = float(np.mean(integrations))
    _pilot_stats["int_std"]  = max(float(np.std(integrations)), 1e-6)
    _pilot_stats["tmp_mean"] = float(np.mean(temporals))
    _pilot_stats["tmp_std"]  = max(float(np.std(temporals)), 1e-6)

    logger.info(
        f"  → int: mean={_pilot_stats['int_mean']:.3f} std={_pilot_stats['int_std']:.3f} | "
        f"tmp: mean={_pilot_stats['tmp_mean']:.3f} std={_pilot_stats['tmp_std']:.3f}"
    )
    logger.info("── Pilot complete ─────────────────────────────────────────────")


# ── HellLoop (Adversarial) ─────────────────────────────────────────────────────
class HellLoop:
    """
    Adversarial Hell-Loop v6.1
    FI deconstructs, SI destabilizes, MG regulates every REGULATOR_EVERY steps
    (first MG call at iteration REGULATOR_EVERY, not at iteration 0).

    SELF detection:
      SELF_EMERGENCE     — sudden z-score spike above pilot baseline
      SELF_CONSOLIDATION — stable coherence sustained over window

    PCA dimensionality expansion (falling PC1) strengthens both signals.
    Temperature regulation uses cosine integration (semantic signal), not NCD.
    """

    def __init__(self, run_id=0):
        self.run_id              = run_id
        self.history             = []
        self.temps               = BASE_TEMPERATURES.copy()
        self.integration_series  = []
        self.temporal_series     = []
        self.pca_series          = []
        self.self_detected       = False
        self.self_mode           = None
        self.metagnosis_detected = False
        self.snapshots           = []
        self._all_vecs           = []
        self._prev_mg_text       = ""

    def _regulate_temps(self, cos_sim: float):
        """
        Negative-feedback temperature regulation against pilot baseline.
        Uses cosine integration (semantic signal) — consistent with SELF detection.
        FIX v6.1: was incorrectly using NCD value.
        """
        int_mean, int_std, _, _ = get_pilot_stats()
        if int_mean is None:
            return
        low  = int_mean - int_std
        high = int_mean + int_std
        if cos_sim < low:
            # Integration too low → boost SI chaos
            self.temps["SI"] = min(self.temps["SI"] + 0.1, SI_TEMP_MAX)
        elif cos_sim > high:
            # Integration too high → prevent consensus, boost FI aggression
            self.temps["FI"] = min(self.temps["FI"] + 0.1, FI_TEMP_MAX)
        else:
            # In zone → decay back toward base temperatures
            self.temps["FI"] = max(self.temps["FI"] - 0.05, BASE_TEMPERATURES["FI"])
            self.temps["SI"] = max(self.temps["SI"] - 0.05, BASE_TEMPERATURES["SI"])

    def _detect_metagnosis(self, mg_text: str) -> tuple:
        """Two-layer metagnosis detection: embedding anchor + NCD continuity."""
        if not mg_text.strip():
            return False, 0.0
        sims       = [cosine(embed(mg_text), anchor) for anchor in _get_anchor()]
        max_sim    = max(sims)
        anchor_hit = max_sim >= METAGNOSIS_EMBED_THRESHOLD
        ncd_hit    = False
        if self._prev_mg_text:
            mg_ncd  = get_ncd_integration(self._prev_mg_text, mg_text)
            ncd_hit = mg_ncd >= MG_NCD_THRESHOLD
        return anchor_hit or ncd_hit, max_sim

    def _pca_expanding(self) -> bool:
        """True if PC1 variance ratio is falling = semantic space expanding."""
        if len(self.pca_series) < 2:
            return False
        recent = [p[0] for p in self.pca_series[-4:] if p]
        if len(recent) < 2:
            return False
        return recent[-1] < PCA_EMERGENCE_THRESHOLD and recent[-1] < recent[0]

    def _check_self(self, i: int, integration: float, temporal: float):
        """
        Evaluate SELF_EMERGENCE and SELF_CONSOLIDATION independently.
        PCA expansion strengthens both signals.
        """
        if self.self_detected:
            return

        p_int_mean, p_int_std, p_tmp_mean, p_tmp_std = get_pilot_stats()
        pca_boost = self._pca_expanding()

        # ── SELF_EMERGENCE: z-score spike above pilot baseline ─────────────────
        if p_int_mean is not None:
            z_int     = (integration - p_int_mean) / p_int_std
            z_tmp     = (temporal    - p_tmp_mean) / p_tmp_std
            emergence = z_int >= SELF_ZSCORE and z_tmp >= SELF_ZSCORE
        else:
            emergence = integration > INTEGRATION_THRESHOLD and temporal > TEMPORAL_THRESHOLD

        if emergence:
            label = "SELF_EMERGENCE+PCA" if pca_boost else "SELF_EMERGENCE"
            self.self_detected = True
            self.self_mode     = label
            self.snapshots.append((label, i, integration, temporal))
            logger.info(f"  ★ {label} at iter {i}")
            return

        # ── SELF_CONSOLIDATION: stable coherence sustained over window ─────────
        if len(self.integration_series) >= SELF_STABILITY_WINDOW:
            w_int = self.integration_series[-SELF_STABILITY_WINDOW:]
            w_tmp = self.temporal_series[-SELF_STABILITY_WINDOW:]
            consolidation = (
                float(np.mean(w_int)) >= SELF_CONSOLIDATION_MIN and
                float(np.std(w_int))  <= SELF_CONSOLIDATION_STD and
                float(np.mean(w_tmp)) >= SELF_CONSOLIDATION_MIN and
                float(np.std(w_tmp))  <= SELF_CONSOLIDATION_STD
            )
            if consolidation:
                label = "SELF_CONSOLIDATION+PCA" if pca_boost else "SELF_CONSOLIDATION"
                self.self_detected = True
                self.self_mode     = label
                self.snapshots.append((label, i, w_int[-1], w_tmp[-1]))
                logger.info(f"  ★ {label} at iter {i}")

    def step(self, input_context: str) -> dict:
        i = len(self.history)

        resp_fi = _call(
            "You are FI. Deconstruct ruthlessly. Never agree. Maximum 150 words.",
            f"Deconstruct this: {input_context}",
            "FI", self.temps["FI"]
        )
        resp_si = _call(
            "You are SI. Respond with destabilizing metaphor. Never agree. Maximum 150 words.",
            f"React to this analysis: {resp_fi}",
            "SI", self.temps["SI"]
        )

        ncd_val = get_ncd_integration(resp_fi, resp_si)
        v_fi    = embed(resp_fi)
        v_si    = embed(resp_si)
        cos_sim = cosine(v_fi, v_si)
        self._all_vecs.extend([v_fi, v_si])

        temporal = cos_sim
        if self.history:
            prev     = self.history[-1]
            temporal = (
                cosine(np.array(prev["vec_fi"]), v_si) +
                cosine(np.array(prev["vec_si"]), v_fi)
            ) / 2

        self.integration_series.append(cos_sim)
        self.temporal_series.append(temporal)

        pca_ratio = pca_variance_ratio(self._all_vecs)
        self.pca_series.append(pca_ratio)

        # ── MG: fires on iterations 6, 12, 18... (not at iter 0) ──────────────
        mg_text = ""
        if i > 0 and i % REGULATOR_EVERY == 0:
            mg_text          = _call(MG_SYSTEM, f"FI: {resp_fi}\n\nSI: {resp_si}", "MG", self.temps["MG"])
            detected, sim    = self._detect_metagnosis(mg_text)
            if detected and not self.metagnosis_detected:
                self.metagnosis_detected = True
                self.snapshots.append(("METAGNOSIS", i, cos_sim, temporal))
                logger.info(f"  ◆ METAGNOSIS at iter {i} (sim={sim:.3f})")
            self._prev_mg_text = mg_text

        # ── Temperature regulation uses cosine integration (semantic signal) ───
        self._regulate_temps(cos_sim)
        self._check_self(i, cos_sim, temporal)

        record = {
            "iteration":           i,
            "ncd_integration":     ncd_val,
            "legacy_integration":  cos_sim,
            "temporal":            temporal,
            "negentropy":          (negentropy(resp_fi) + negentropy(resp_si)) / 2,
            "entropy":             (shannon_entropy(resp_fi) + shannon_entropy(resp_si)) / 2,
            "fi_response":         resp_fi,
            "si_response":         resp_si,
            "mg_response":         mg_text,
            "pca_variance_ratio":  pca_ratio,
            "vec_fi":              v_fi.tolist(),
            "vec_si":              v_si.tolist(),
            "temp_fi":             self.temps["FI"],
            "temp_si":             self.temps["SI"],
            "self":                self.self_detected,
            "self_mode":           self.self_mode,
            "metagnosis_detected": self.metagnosis_detected,
        }
        self.history.append(record)
        logger.info(
            f"[run={self.run_id} iter={i:02d}] "
            f"NCD={ncd_val:.3f} cos={cos_sim:.3f} tmp={temporal:.3f} "
            f"T_FI={self.temps['FI']:.2f} T_SI={self.temps['SI']:.2f} "
            f"SELF={self.self_mode or '-'}"
        )
        return record
