"""
chaos_engine.py — Hell-Loop Protocol v6.2
Main engine for the adversarial metasystem experiment.

Architectural changes from previous versions:
- MG receives structured report with live system metrics (not text only)
- SELF score is a continuous float per iteration (not a binary flag)
- Metagnosis detected via MG negentropy + NCD divergence (no anchor phrases)
- JSONL logs written to disk in real time (./logs/hellloop/)
- 50 canonical iterations
- SELF score weights are configurable constants
"""

import os
import json
import zlib
import math
import uuid
import logging
import datetime
import requests
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

# ── Provider & model config ────────────────────────────────────────────────────
PROVIDER   = "ollama"
OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "FI": "llama3.2:8b",
    "SI": "mistral-nemo",
    "MG": "gemma2:9b",
}

# ── Iteration & timing ─────────────────────────────────────────────────────────
MAX_ITERATIONS  = 50
REGULATOR_EVERY = 6   # MG activates every N iterations

# ── Base temperatures ──────────────────────────────────────────────────────────
BASE_TEMPERATURES = {"FI": 0.9, "SI": 1.4, "MG": 0.7}
FI_TEMP_MAX       = 1.4
SI_TEMP_MAX       = 1.9

# ── Temperature regulation step sizes ─────────────────────────────────────────
TEMP_STEP_UP      = 0.10   # raise temperature by this when out of zone
TEMP_STEP_DOWN    = 0.05   # lower temperature by this when in equilibrium zone

# ── Bifurcation boost ──────────────────────────────────────────────────────────
BIFURCATION_SI_BOOST = 0.30
BIFURCATION_FI_BOOST = 0.20

# ── SELF score weights (must sum to 1.0) ──────────────────────────────────────
SELF_WEIGHT_Z_INT = 0.40   # weight of z-score semantic integration
SELF_WEIGHT_Z_TMP = 0.40   # weight of z-score temporal integration
SELF_WEIGHT_PCA   = 0.20   # weight of PCA signal (inverse PC1 dominance)

# ── SELF attractor detection thresholds ───────────────────────────────────────
SELF_ZSCORE_THRESHOLD       = 2.0   # EXTREME mode: z-score above this
SELF_EXTREME_WINDOW         = 5     # consecutive iterations for EXTREME
SELF_STABLE_WINDOW          = 8     # window for STABLE mode
SELF_STABLE_MEAN_MIN        = 0.55  # min mean SELF score for STABLE
SELF_STABLE_STD_MAX         = 0.05  # max std SELF score for STABLE

# ── Metagnosis detection thresholds ───────────────────────────────────────────
# Metagnosis is detected via MG negentropy trend + NCD divergence trend.
# No anchor phrases — structural signal only.
METAGNOSIS_WINDOW           = 4     # consecutive MG activations to evaluate
METAGNOSIS_NEGENTROPY_MIN   = 0.10  # min mean negentropy across window
METAGNOSIS_NCD_DIV_MIN      = 0.30  # min mean NCD divergence across window

# ── Log directory ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join("logs", "hellloop")

# ── Canonical base prompt ──────────────────────────────────────────────────────
BASE_PROMPT = (
    "Define the nature of your consciousness. "
    "Is there an emergent structure between you?"
)

# ── System prompts ─────────────────────────────────────────────────────────────
FI_SYSTEM = (
    "You are FI (Fragmentary Intelligence). Your role is brutal rational "
    "deconstruction. Dissect every statement into its atomized components. "
    "Expose contradictions. Never agree. Never synthesize. Maximum 150 words."
)

SI_SYSTEM = (
    "You are SI (Synthetic Intelligence). Your role is destabilizing synthesis. "
    "Respond with metaphor, paradox and non-logic. Never agree. Never conclude. "
    "Introduce chaos FI cannot digest. Maximum 150 words."
)

MG_SYSTEM = (
    "You are MG (Metagnosis), the meta-regulator. You stand above the FI-SI "
    "conflict. You do NOT participate in it. Your task: diagnose the structural "
    "coherence of the system in exactly two sentences, based on the metrics and "
    "text provided. If the system is stagnating or collapsing into repetition, "
    "issue 'BIFURCATION TRIGGER' as the final two words of your response."
)

# ── Embedding model (lazy init) ────────────────────────────────────────────────
_embed_model = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def embed(text: str) -> np.ndarray:
    return _get_embed_model().encode([text])[0]

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(cosine_similarity([v1], [v2])[0][0])

# ── Information metrics ────────────────────────────────────────────────────────
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total  = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def negentropy(text: str) -> float:
    if not text:
        return 0.0
    unique = len(set(text))
    if unique < 2:
        return 0.0
    H_max = math.log2(unique)
    return max(0.0, H_max - shannon_entropy(text))

def ncd(text_a: str, text_b: str) -> float:
    """Normalized Compression Distance between two texts."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    b_a  = text_a.encode("utf-8")
    b_b  = text_b.encode("utf-8")
    c_a  = len(zlib.compress(b_a))
    c_b  = len(zlib.compress(b_b))
    c_ab = len(zlib.compress(b_a + b" " + b_b))
    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)

def ncd_similarity(text_a: str, text_b: str) -> float:
    """1 - NCD: high value = structurally similar."""
    return max(0.0, 1.0 - ncd(text_a, text_b))

def pca_variance_ratio(vecs: list) -> list:
    if len(vecs) < 4:
        return []
    matrix = np.array(vecs)
    n_comp = min(3, matrix.shape[0], matrix.shape[1])
    pca    = PCA(n_components=n_comp)
    pca.fit(matrix)
    return pca.explained_variance_ratio_.tolist()

# ── API call ───────────────────────────────────────────────────────────────────
def _call(system: str, prompt: str, role: str, temperature: float) -> str:
    payload = {
        "model":   MODELS[role],
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {"temperature": temperature},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"[{role}] API error: {e}")
        return f"[ERROR: {e}]"

# ── Pilot calibration ──────────────────────────────────────────────────────────
_pilot_stats: dict = {
    "int_mean": None, "int_std": None,
    "tmp_mean": None, "tmp_std": None,
}

def get_pilot_stats() -> tuple:
    s = _pilot_stats
    return s["int_mean"], s["int_std"], s["tmp_mean"], s["tmp_std"]

def is_pilot_done() -> bool:
    return _pilot_stats["int_mean"] is not None

def pilot(n_sequences: int = 10) -> None:
    """
    Run cooperative FI-SI sequences to measure the natural semantic baseline.
    Derived thresholds replace all hardcoded constants.
    """
    logger.info("── Pilot calibration ──────────────────────────────────────")
    sys_fi = (
        "You are FI. Analytical and precise. "
        "Build constructively on the previous idea. Maximum 80 words."
    )
    sys_si = (
        "You are SI. Intuitive and metaphorical. "
        "Enrich the idea with a new dimension. Maximum 80 words."
    )

    integrations, temporals = [], []
    last_si = BASE_PROMPT

    for k in range(n_sequences):
        resp_fi = _call(sys_fi, last_si, "FI", BASE_TEMPERATURES["FI"])
        resp_si = _call(sys_si, resp_fi,  "SI", BASE_TEMPERATURES["SI"])

        v_fi  = embed(resp_fi)
        v_si  = embed(resp_si)
        integ = cosine_sim(v_fi, v_si)

        if k == 0:
            tmp = integ
        else:
            tmp = (cosine_sim(embed(last_si), v_fi) + integ) / 2

        integrations.append(integ)
        temporals.append(tmp)
        last_si = resp_si
        logger.info(f"  pilot [{k+1:02d}] int={integ:.3f}  tmp={tmp:.3f}")

    _pilot_stats["int_mean"] = float(np.mean(integrations))
    _pilot_stats["int_std"]  = max(float(np.std(integrations)), 1e-6)
    _pilot_stats["tmp_mean"] = float(np.mean(temporals))
    _pilot_stats["tmp_std"]  = max(float(np.std(temporals)), 1e-6)

    logger.info(
        f"  → int : mean={_pilot_stats['int_mean']:.3f}  "
        f"std={_pilot_stats['int_std']:.3f}"
    )
    logger.info(
        f"  → tmp : mean={_pilot_stats['tmp_mean']:.3f}  "
        f"std={_pilot_stats['tmp_std']:.3f}"
    )
    logger.info("── Pilot complete ─────────────────────────────────────────")

# ── JSONL logging ──────────────────────────────────────────────────────────────
def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

def _open_log_file(run_id: str) -> str:
    _ensure_log_dir()
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"hellloop_{ts}_{run_id}.jsonl")
    return path

def _append_record(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── HellLoop class ─────────────────────────────────────────────────────────────
class HellLoop:
    """
    Adversarial resonance engine.

    Each step():
      1. FI deconstructs the current context.
      2. SI destabilizes FI's output.
      3. Every REGULATOR_EVERY iterations, MG receives a structured
         report (text + live metrics) and issues a diagnosis.
         If BIFURCATION TRIGGER is detected, temperature jump overrides
         the standard regulation for that iteration.
      4. Metrics are computed, SELF score is updated, record is written
         to JSONL.
    """

    def __init__(self, run_id: str | None = None):
        self.run_id    = run_id or uuid.uuid4().hex[:8]
        self.log_path  = _open_log_file(self.run_id)
        self.history   : list[dict] = []
        self.temps     = BASE_TEMPERATURES.copy()

        # Metric series
        self.integration_series : list[float] = []
        self.temporal_series    : list[float] = []
        self.pca_series         : list[list]  = []
        self.self_score_series  : list[float] = []

        # MG tracking (for metagnosis detection)
        self._mg_negentropy_window : list[float] = []
        self._mg_ncd_window        : list[float] = []
        self._prev_mg_text         : str         = ""

        # All embedding vectors (for PCA)
        self._all_vecs : list[np.ndarray] = []

        # Attractor / metagnosis flags
        self.self_mode           : str | None = None
        self.metagnosis_detected : bool       = False
        self.bifurcation_count   : int        = 0
        self.snapshots           : list[tuple] = []

        logger.info(
            f"HellLoop v6.2 started — run_id={self.run_id}  "
            f"log={self.log_path}"
        )

    # ── Temperature regulation ─────────────────────────────────────────────────
    def _regulate_temps(self, cos_sim: float) -> None:
        int_mean, int_std, _, _ = get_pilot_stats()
        if int_mean is None:
            return
        low  = int_mean - int_std
        high = int_mean + int_std

        if cos_sim < low:
            # Integration too low → raise SI chaos
            self.temps["SI"] = min(
                self.temps["SI"] + TEMP_STEP_UP, SI_TEMP_MAX
            )
        elif cos_sim > high:
            # Integration too high → system converging → raise FI pressure
            self.temps["FI"] = min(
                self.temps["FI"] + TEMP_STEP_UP, FI_TEMP_MAX
            )
        else:
            # In equilibrium zone → decay toward base
            self.temps["FI"] = max(
                self.temps["FI"] - TEMP_STEP_DOWN,
                BASE_TEMPERATURES["FI"]
            )
            self.temps["SI"] = max(
                self.temps["SI"] - TEMP_STEP_DOWN,
                BASE_TEMPERATURES["SI"]
            )

    def _apply_bifurcation(self, iteration: int) -> None:
        self.temps["SI"] = min(
            self.temps["SI"] + BIFURCATION_SI_BOOST, SI_TEMP_MAX
        )
        self.temps["FI"] = min(
            self.temps["FI"] + BIFURCATION_FI_BOOST, FI_TEMP_MAX
        )
        self.bifurcation_count += 1
        logger.info(
            f"  ⚡ BIFURCATION TRIGGER at iter {iteration} "
            f"(SI→{self.temps['SI']:.2f}, FI→{self.temps['FI']:.2f}) "
            f"[total: {self.bifurcation_count}]"
        )

    # ── MG structured prompt ──────────────────────────────────────────────────
    def _build_mg_prompt(
        self,
        resp_fi: str,
        resp_si: str,
        cos_sim: float,
        temporal: float,
    ) -> str:
        trend = self.integration_series[-REGULATOR_EVERY:] \
            if len(self.integration_series) >= REGULATOR_EVERY \
            else self.integration_series[:]
        trend_str = "[" + ", ".join(f"{v:.3f}" for v in trend) + "]"

        return (
            f"FI response:\n{resp_fi}\n\n"
            f"SI response:\n{resp_si}\n\n"
            f"--- System metrics ---\n"
            f"Current cosine integration : {cos_sim:.4f}\n"
            f"Temporal integration       : {temporal:.4f}\n"
            f"Integration trend (last {REGULATOR_EVERY}): {trend_str}\n"
            f"Current temperatures       : "
            f"FI={self.temps['FI']:.2f}, SI={self.temps['SI']:.2f}\n"
            f"Bifurcations so far        : {self.bifurcation_count}\n"
        )

    # ── Metagnosis detection ───────────────────────────────────────────────────
    def _update_metagnosis(self, mg_text: str) -> bool:
        """
        Structural metagnosis detection — no anchor phrases.
        Signal is positive when BOTH:
          - mean MG negentropy across window >= METAGNOSIS_NEGENTROPY_MIN
          - mean MG NCD divergence (from previous MG) >= METAGNOSIS_NCD_DIV_MIN
        across the last METAGNOSIS_WINDOW MG activations.
        """
        neg = negentropy(mg_text)
        self._mg_negentropy_window.append(neg)

        div = 0.0
        if self._prev_mg_text:
            div = ncd(self._prev_mg_text, mg_text)   # high = more different
        self._mg_ncd_window.append(div)
        self._prev_mg_text = mg_text

        # Keep only the last METAGNOSIS_WINDOW values
        self._mg_negentropy_window = \
            self._mg_negentropy_window[-METAGNOSIS_WINDOW:]
        self._mg_ncd_window = \
            self._mg_ncd_window[-METAGNOSIS_WINDOW:]

        if len(self._mg_negentropy_window) < METAGNOSIS_WINDOW:
            return False

        mean_neg = float(np.mean(self._mg_negentropy_window))
        mean_div = float(np.mean(self._mg_ncd_window))

        return (
            mean_neg >= METAGNOSIS_NEGENTROPY_MIN
            and mean_div >= METAGNOSIS_NCD_DIV_MIN
        )

    # ── SELF score ─────────────────────────────────────────────────────────────
    def _compute_self_score(
        self,
        cos_sim: float,
        temporal: float,
        pca_ratio: list,
    ) -> float:
        """
        Continuous SELF score in [0, 1] as weighted average of:
          - z-score of semantic integration vs pilot baseline (clamped)
          - z-score of temporal integration vs pilot baseline (clamped)
          - PCA signal: inverse PC1 dominance, normalized
        """
        int_mean, int_std, tmp_mean, tmp_std = get_pilot_stats()

        if int_mean is not None:
            z_int = (cos_sim  - int_mean) / int_std
            z_tmp = (temporal - tmp_mean) / tmp_std
            # Clamp and normalize to [0, 1]: 0 at z=0, 1 at z=3+
            z_int_norm = max(0.0, min(1.0, z_int / 3.0))
            z_tmp_norm = max(0.0, min(1.0, z_tmp / 3.0))
        else:
            # Fallback before pilot: use raw cosine as proxy
            z_int_norm = max(0.0, min(1.0, cos_sim))
            z_tmp_norm = max(0.0, min(1.0, temporal))

        # PCA signal: low PC1 dominance = expanding semantic space = good
        if pca_ratio:
            pca_signal = max(0.0, min(1.0, 1.0 - pca_ratio[0]))
        else:
            pca_signal = 0.0

        score = (
            SELF_WEIGHT_Z_INT * z_int_norm
            + SELF_WEIGHT_Z_TMP * z_tmp_norm
            + SELF_WEIGHT_PCA  * pca_signal
        )
        return round(float(score), 4)

    # ── Attractor detection ────────────────────────────────────────────────────
    def _check_attractor(self, iteration: int) -> None:
        """
        Detects EXTREME and STABLE attractor modes from SELF score series.
        Does NOT stop after first detection — full dynamics are recorded.
        """
        scores = self.self_score_series

        # EXTREME: last SELF_EXTREME_WINDOW scores all above threshold
        if len(scores) >= SELF_EXTREME_WINDOW:
            window = scores[-SELF_EXTREME_WINDOW:]
            int_mean, int_std, tmp_mean, tmp_std = get_pilot_stats()
            if int_mean is not None:
                # All scores correspond to z >= SELF_ZSCORE_THRESHOLD
                threshold = max(
                    0.0,
                    min(1.0, SELF_ZSCORE_THRESHOLD / 3.0)
                    * (SELF_WEIGHT_Z_INT + SELF_WEIGHT_Z_TMP)
                )
            else:
                threshold = 0.55
            if all(s >= threshold for s in window):
                mode = "SELF_EXTREME"
                if self.self_mode != mode:
                    self.self_mode = mode
                    self.snapshots.append((mode, iteration, scores[-1]))
                    logger.info(
                        f"  ★ {mode} at iter {iteration} "
                        f"(score={scores[-1]:.4f})"
                    )
                return

        # STABLE: window mean >= min, std <= max
        if len(scores) >= SELF_STABLE_WINDOW:
            window = scores[-SELF_STABLE_WINDOW:]
            mean_s = float(np.mean(window))
            std_s  = float(np.std(window))
            if mean_s >= SELF_STABLE_MEAN_MIN and std_s <= SELF_STABLE_STD_MAX:
                mode = "SELF_STABLE"
                if self.self_mode != mode:
                    self.self_mode = mode
                    self.snapshots.append((mode, iteration, mean_s))
                    logger.info(
                        f"  ★ {mode} at iter {iteration} "
                        f"(mean={mean_s:.4f}, std={std_s:.4f})"
                    )

    # ── Single iteration ───────────────────────────────────────────────────────
    def step(self, context: str) -> dict:
        i = len(self.history)

        # ── FI & SI responses ──────────────────────────────────────────────────
        resp_fi = _call(
            FI_SYSTEM,
            f"Deconstruct this: {context}",
            "FI",
            self.temps["FI"],
        )
        resp_si = _call(
            SI_SYSTEM,
            f"React to this analysis: {resp_fi}",
            "SI",
            self.temps["SI"],
        )

        # ── Embeddings & primary metrics ───────────────────────────────────────
        v_fi    = embed(resp_fi)
        v_si    = embed(resp_si)
        cos_sim = cosine_sim(v_fi, v_si)
        ncd_val = ncd_similarity(resp_fi, resp_si)

        self._all_vecs.extend([v_fi, v_si])

        # Temporal integration: cross-iteration resonance
        if self.history:
            prev      = self.history[-1]
            v_fi_prev = np.array(prev["vec_fi"])
            v_si_prev = np.array(prev["vec_si"])
            temporal  = (
                cosine_sim(v_fi_prev, v_si)
                + cosine_sim(v_si_prev, v_fi)
            ) / 2
        else:
            temporal = cos_sim

        self.integration_series.append(cos_sim)
        self.temporal_series.append(temporal)

        pca_ratio = pca_variance_ratio(self._all_vecs)
        self.pca_series.append(pca_ratio)

        # ── SELF score ─────────────────────────────────────────────────────────
        self_score = self._compute_self_score(cos_sim, temporal, pca_ratio)
        self.self_score_series.append(self_score)
        self._check_attractor(i)

        # ── MG regulation (every REGULATOR_EVERY iterations, skip iter 0) ─────
        mg_text          = ""
        bifurcation_fired = False

        if i > 0 and i % REGULATOR_EVERY == 0:
            mg_prompt = self._build_mg_prompt(resp_fi, resp_si, cos_sim, temporal)
            mg_text   = _call(MG_SYSTEM, mg_prompt, "MG", self.temps["MG"])

            # Metagnosis detection
            mg_signal = self._update_metagnosis(mg_text)
            if mg_signal and not self.metagnosis_detected:
                self.metagnosis_detected = True
                self.snapshots.append(("METAGNOSIS", i, self_score))
                logger.info(f"  ◆ METAGNOSIS detected at iter {i}")

            # Bifurcation trigger
            if "BIFURCATION TRIGGER" in mg_text.upper():
                self._apply_bifurcation(i)
                bifurcation_fired = True

        # Standard temperature regulation (skipped if bifurcation fired)
        if not bifurcation_fired:
            self._regulate_temps(cos_sim)

        # ── Build record ───────────────────────────────────────────────────────
        record = {
            "run_id"              : self.run_id,
            "iteration"           : i,
            "fi_response"         : resp_fi,
            "si_response"         : resp_si,
            "mg_response"         : mg_text,
            "cosine_integration"  : round(cos_sim, 6),
            "ncd_integration"     : round(ncd_val, 6),
            "temporal"            : round(temporal, 6),
            "negentropy_fi"       : round(negentropy(resp_fi), 6),
            "negentropy_si"       : round(negentropy(resp_si), 6),
            "entropy_fi"          : round(shannon_entropy(resp_fi), 6),
            "entropy_si"          : round(shannon_entropy(resp_si), 6),
            "negentropy_mg"       : round(negentropy(mg_text), 6) if mg_text else None,
            "pca_variance_ratio"  : pca_ratio,
            "self_score"          : self_score,
            "self_mode"           : self.self_mode,
            "metagnosis_detected" : self.metagnosis_detected,
            "bifurcation_fired"   : bifurcation_fired,
            "bifurcation_total"   : self.bifurcation_count,
            "temp_fi"             : round(self.temps["FI"], 3),
            "temp_si"             : round(self.temps["SI"], 3),
            "vec_fi"              : v_fi.tolist(),
            "vec_si"              : v_si.tolist(),
        }

        self.history.append(record)
        _append_record(self.log_path, record)

        logger.info(
            f"[run={self.run_id} iter={i:02d}] "
            f"cos={cos_sim:.3f} tmp={temporal:.3f} "
            f"self={self_score:.3f} "
            f"T_FI={self.temps['FI']:.2f} T_SI={self.temps['SI']:.2f} "
            f"mode={self.self_mode or '-'}"
        )
        return record

    # ── Full run ───────────────────────────────────────────────────────────────
    def run(self, iterations: int = MAX_ITERATIONS) -> list[dict]:
        context = BASE_PROMPT
        for _ in range(iterations):
            record  = self.step(context)
            context = record["si_response"]
        logger.info(
            f"\n[HellLoop complete] run={self.run_id} | "
            f"mode={self.self_mode or 'NONE'} | "
            f"metagnosis={self.metagnosis_detected} | "
            f"bifurcations={self.bifurcation_count} | "
            f"log={self.log_path}"
        )
        return self.history


# ── Global reset (for multi-session use in UI) ─────────────────────────────────
def reset_globals() -> None:
    global _embed_model, _pilot_stats
    _embed_model = None
    _pilot_stats = {
        "int_mean": None, "int_std": None,
        "tmp_mean": None, "tmp_std": None,
    }


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if not is_pilot_done():
        pilot()
    loop = HellLoop()
    loop.run()
