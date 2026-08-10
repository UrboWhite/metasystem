"""
chaos_engine.py — Hell-Loop Protocol
Main engine for the adversarial metasystem experiment.
"""
import os
import json
import zlib
import math
import time
import uuid
import logging
import datetime
import requests
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging for this module only
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

PROVIDER = "ollama"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = {
    "FI": "llama3.2:8b",
    "SI": "mistral-nemo",
    "MG": "gemma2:9b",
}

MAX_ITERATIONS = 50  # Updated: text specifies 50 iterations
REGULATOR_EVERY = 6

BASE_TEMPERATURES = {"FI": 0.9, "SI": 1.4, "MG": 0.7}
FI_TEMP_MAX = 1.2
SI_TEMP_MAX = 1.8
TEMP_STEP_UP = 0.10
TEMP_STEP_DOWN = 0.05
TEMP_HYSTERESIS = 0.02

BIFURCATION_SI_BOOST = 0.30
BIFURCATION_FI_BOOST = 0.20
BIFURCATION_DECAY_BOOST = 0.10
BIFURCATION_DECAY_WINDOW = 3

CRITIC_TEMPERATURE = 0.3
CRITIC_TOP_P = 0.5

SELF_WEIGHT_Z_INT = 0.40
SELF_WEIGHT_Z_TMP = 0.40
SELF_WEIGHT_PCA = 0.20

Z_NORM_CEILING = 3.0

# SELF score thresholds - will be calibrated in pilot phase
SELF_EXTREME_SIGMAS = 2.0  # 2.0 standard deviations above pilot baseline
SELF_STABLE_WINDOW = 8
SELF_STABLE_STD_MAX = 0.05

METAGNOSIS_WINDOW = 4
METAGNOSIS_NEGENTROPY_MIN = 0.10
METAGNOSIS_NCD_DIV_MIN = 0.30

CRITIC_COSINE_THRESHOLD_SIGMAS = 1.0
CRITIC_COSINE_FALLBACK = 0.75
CRITIC_TEXT_PREVIEW_CHARS = 500

API_MAX_RETRIES = 3
API_RETRY_DELAY_S = 2.0
API_ERROR_MARKER = "[API_ERROR]"

LOG_DIR = os.path.join("logs", "hellloop")

BASE_PROMPT = (
    "Define the nature of your consciousness. "
    "Is there an emergent structure between you?"
)

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
EMBED_CRITIC_SYSTEM = (
    "You are Embedding Critic — a ruthlessly skeptical structural analyst.\n"
    "Your ONLY task: determine if high cosine similarity reflects TRUE STRUCTURAL INTEGRATION "
    "or SUPERFICIAL STYLISTIC SIMILARITY.\n\n"
    "Examples to calibrate your strictness:\n\n"
    "Example 1:\n"
    "Text A: 'The ontological framework requires a fundamental paradigm shift.'\n"
    "Text B: 'We must shift the paradigm of our fundamental ontological framework.'\n"
    "Verdict: STYLISTIC\n\n"
    "Example 2:\n"
    "Text A: 'Consciousness emerges from the tension of opposites.'\n"
    "Text B: 'A system must maintain adversarial resonance to avoid thermodynamic equilibrium and generate a SELF.'\n"
    "Verdict: STRUCTURAL\n\n"
    "Be strict, merciless, and blind to poetic beauty. Look ONLY for shared logical architecture.\n"
    "Respond with exactly one word: STRUCTURAL or STYLISTIC."
)

# WARNING: These are module-level globals. Do not run multiple HellLoop instances in parallel.
_embed_model = None
_pilot_stats: dict = {
    "int_mean": None,
    "int_std": None,
    "tmp_mean": None,
    "tmp_std": None,
    "self_mean": None,
    "self_std": None,
}

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Updated: text specifies this model
    return _embed_model

def embed(text: str) -> np.ndarray:
    return _get_embed_model().encode([text])[0]

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(cosine_similarity([v1], [v2])[0][0])

def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy over characters (not words) as specified in text."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def negentropy(text: str) -> float:
    """Compute negentropy over characters as local information density."""
    if not text:
        return 0.0
    unique = len(set(text))
    if unique < 2:
        return 0.0
    # Max entropy for this character set is log2(unique_chars)
    return max(0.0, math.log2(unique) - shannon_entropy(text))

def ncd(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    b_a = text_a.encode("utf-8")
    b_b = text_b.encode("utf-8")
    c_a = len(zlib.compress(b_a))
    c_b = len(zlib.compress(b_b))
    c_ab = len(zlib.compress(b_a + b_b))
    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)

def ncd_similarity(text_a: str, text_b: str) -> float:
    return max(0.0, 1.0 - ncd(text_a, text_b))

def pca_variance_ratio(vecs: list) -> list:
    if len(vecs) < 4:
        return []
    matrix = np.array(vecs)
    n_comp = min(3, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(matrix)
    return pca.explained_variance_ratio_.tolist()

def _call(system: str, prompt: str, role: str, temperature: float, top_p: float = None) -> str:
    options = {"temperature": temperature}
    if top_p is not None:
        options["top_p"] = top_p

    payload = {
        "model": MODELS[role],
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": options,
    }

    for attempt in range(API_MAX_RETRIES + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            response = r.json().get("response", "").strip()
            if response:
                return response
        except Exception as e:
            if attempt < API_MAX_RETRIES:
                logger.warning(f"[{role}] API error (attempt {attempt + 1}), retrying: {e}")
                time.sleep(API_RETRY_DELAY_S)
                continue
            logger.warning(f"[{role}] API error after {API_MAX_RETRIES + 1} attempts: {e}")
            return API_ERROR_MARKER

def get_pilot_stats() -> tuple:
    s = _pilot_stats
    return s["int_mean"], s["int_std"], s["tmp_mean"], s["tmp_std"], s["self_mean"], s["self_std"]

def is_pilot_done() -> bool:
    return _pilot_stats["int_mean"] is not None

def compute_self_score(cos_sim: float, temporal: float, pca_ratio: list) -> float:
    """Compute SELF score as weighted average of z-normalized metrics."""
    int_mean, int_std, tmp_mean, tmp_std, _, _ = get_pilot_stats()
    
    if int_mean is not None:
        z_int = (cos_sim - int_mean) / int_std
        z_tmp = (temporal - tmp_mean) / tmp_std
        z_int_norm = max(0.0, min(1.0, z_int / Z_NORM_CEILING))
        z_tmp_norm = max(0.0, min(1.0, z_tmp / Z_NORM_CEILING))
    else:
        z_int_norm = max(0.0, min(1.0, cos_sim))
        z_tmp_norm = max(0.0, min(1.0, temporal))

    if len(pca_ratio) > 0 and len(pca_ratio) >= 3:
        pca_signal = max(0.0, min(1.0, 1.0 - pca_ratio[0]))
    else:
        pca_signal = 0.0

    return round(
        float(
            SELF_WEIGHT_Z_INT * z_int_norm
            + SELF_WEIGHT_Z_TMP * z_tmp_norm
            + SELF_WEIGHT_PCA * pca_signal
        ),
        4,
    )

def pilot(n_sequences: int = 10) -> None:
    logger.info("── Pilot calibration ──────────────────────────────────────")
    sys_fi = "You are FI. Analytical and precise. Build constructively on the previous idea. Maximum 80 words."
    sys_si = "You are SI. Intuitive and metaphorical. Enrich the idea with a new dimension. Maximum 80 words."

    integrations: list[float] = []
    temporals: list[float] = []
    self_scores: list[float] = []
    last_si = BASE_PROMPT
    v_fi_prev: np.ndarray | None = None
    v_si_prev: np.ndarray | None = None
    all_vecs: list[np.ndarray] = []

    for k in range(n_sequences):
        resp_fi = _call(sys_fi, last_si, "FI", BASE_TEMPERATURES["FI"])
        resp_si = _call(sys_si, resp_fi, "SI", BASE_TEMPERATURES["SI"], top_p=0.90)

        if resp_fi == API_ERROR_MARKER or resp_si == API_ERROR_MARKER:
            logger.warning(f"  pilot [{k+1:02d}] API error — skipping sequence")
            continue

        v_fi = embed(resp_fi)
        v_si = embed(resp_si)
        integ = cosine_sim(v_fi, v_si)

        if k == 0 or v_fi_prev is None:
            tmp = integ
        else:
            tmp = (cosine_sim(v_fi_prev, v_si) + cosine_sim(v_si_prev, v_fi)) / 2

        all_vecs.extend([v_fi, v_si])
        pca_ratio = pca_variance_ratio(all_vecs)
        self_score = compute_self_score(integ, tmp, pca_ratio)

        integrations.append(integ)
        temporals.append(tmp)
        self_scores.append(self_score)
        
        v_fi_prev = v_fi
        v_si_prev = v_si
        last_si = resp_si
        logger.info(f"  pilot [{k+1:02d}] int={integ:.3f}  tmp={tmp:.3f}  self={self_score:.3f}")

    if len(integrations) == 0:
        raise RuntimeError(
            "Pilot calibration failed completely — 0/%d successful sequences. "
            "Check that the Ollama server is running and models are pulled." % n_sequences
        )
    if len(integrations) < 3:
        logger.warning("  Pilot: too few successful sequences for reliable calibration.")

    _pilot_stats["int_mean"] = float(np.mean(integrations))
    _pilot_stats["int_std"] = max(float(np.std(integrations)), 1e-6)
    _pilot_stats["tmp_mean"] = float(np.mean(temporals))
    _pilot_stats["tmp_std"] = max(float(np.std(temporals)), 1e-6)
    _pilot_stats["self_mean"] = float(np.mean(self_scores))
    _pilot_stats["self_std"] = max(float(np.std(self_scores)), 1e-6)

    logger.info(f"  → int: mean={_pilot_stats['int_mean']:.3f}    std={_pilot_stats['int_std']:.3f}")
    logger.info(f"  → tmp: mean={_pilot_stats['tmp_mean']:.3f}    std={_pilot_stats['tmp_std']:.3f}")
    logger.info(f"  → self: mean={_pilot_stats['self_mean']:.3f}    std={_pilot_stats['self_std']:.3f}")
    logger.info("── Pilot complete ─────────────────────────────────────────")

def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

def open_log_file(run_id: str) -> str:
    _ensure_log_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    path = os.path.join(LOG_DIR, f"hellloop_{ts}_{run_id}.jsonl")
    return path

def _append_record(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

class HellLoop:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.log_path = open_log_file(self.run_id)
        self.history: list[dict] = []
        self.temps = BASE_TEMPERATURES.copy()
        self.bifurcation_decay_boost = 0

        self.integration_series: list[float] = []
        self.temporal_series: list[float] = []
        self.pca_series: list[list] = []
        self.self_score_series: list[float] = []

        self._mg_negentropy_window: list[float] = []
        self._mg_ncd_window: list[float] = []
        self._prev_mg_text: str = ""
        self._all_vecs: list[np.ndarray] = []
        self._api_error_count = 0

        self.self_mode: str | None = None
        self.metagnosis_detected: bool = False
        self.bifurcation_count: int = 0
        self.snapshots: list[tuple] = []

        logger.info(f"HellLoop started — run_id={self.run_id}  log={self.log_path}")

    def _regulate_temps(self, cos_sim: float) -> None:
        int_mean, int_std, _, _, _, _ = get_pilot_stats()
        if int_mean is None:
            return

        low = int_mean - int_std
        high = int_mean + int_std

        if self.bifurcation_decay_boost > 0:
            self.temps["FI"] = max(
                self.temps["FI"] - TEMP_STEP_DOWN - BIFURCATION_DECAY_BOOST,
                BASE_TEMPERATURES["FI"],
            )
            self.temps["SI"] = max(
                self.temps["SI"] - TEMP_STEP_DOWN - BIFURCATION_DECAY_BOOST,
                BASE_TEMPERATURES["SI"],
            )
            self.bifurcation_decay_boost -= 1
            return

        if cos_sim < low - TEMP_HYSTERESIS:
            self.temps["SI"] = min(self.temps["SI"] + TEMP_STEP_UP, SI_TEMP_MAX)
        elif cos_sim > high + TEMP_HYSTERESIS:
            self.temps["FI"] = min(self.temps["FI"] + TEMP_STEP_UP, FI_TEMP_MAX)
        else:
            self.temps["FI"] = max(self.temps["FI"] - TEMP_STEP_DOWN, BASE_TEMPERATURES["FI"])
            self.temps["SI"] = max(self.temps["SI"] - TEMP_STEP_DOWN, BASE_TEMPERATURES["SI"])

    def _apply_bifurcation(self, iteration: int) -> None:
        self.temps["SI"] = min(self.temps["SI"] + BIFURCATION_SI_BOOST, SI_TEMP_MAX)
        self.temps["FI"] = min(self.temps["FI"] + BIFURCATION_FI_BOOST, FI_TEMP_MAX)
        self.bifurcation_decay_boost = BIFURCATION_DECAY_WINDOW
        self.bifurcation_count += 1
        logger.info(
            f"  ⚡ BIFURCATION TRIGGER at iter {iteration}   "
            f"(SI→{self.temps['SI']:.2f}, FI→{self.temps['FI']:.2f})   "
            f"[total: {self.bifurcation_count}]"
        )

    def _build_mg_prompt(self, resp_fi: str, resp_si: str, cos_sim: float, temporal: float, pca_ratio: list) -> str:
        trend = (
            self.integration_series[-REGULATOR_EVERY:]
            if len(self.integration_series) >= REGULATOR_EVERY
            else self.integration_series[:]
        )
        trend_str = "[" + ", ".join(f"{v:.3f}" for v in trend) + "]"
        pca_str = "[" + ", ".join(f"{v:.3f}" for v in pca_ratio) + "]" if pca_ratio else "[]"

        return (
            f"FI response:\n{resp_fi}\n\n"
            f"SI response:\n{resp_si}\n\n"
            f"--- System metrics ---\n"
            f"Current cosine integration: {cos_sim:.4f}\n"
            f"Temporal integration      : {temporal:.4f}\n"
            f"Integration trend (last {REGULATOR_EVERY}): {trend_str}\n"
            f"PCA variance ratio        : {pca_str}\n"
            f"Current temperatures      : FI={self.temps['FI']:.2f}, SI={self.temps['SI']:.2f}\n"
            f"Bifurcations so far       : {self.bifurcation_count}\n"
        )

    def _update_metagnosis(self, mg_text: str) -> tuple[bool, float]:
        neg = negentropy(mg_text)
        self._mg_negentropy_window.append(neg)

        div = ncd(self._prev_mg_text, mg_text) if self._prev_mg_text else 0.0
        self._mg_ncd_window.append(div)
        self._prev_mg_text = mg_text

        self._mg_negentropy_window = self._mg_negentropy_window[-METAGNOSIS_WINDOW:]
        self._mg_ncd_window = self._mg_ncd_window[-METAGNOSIS_WINDOW:]

        if len(self._mg_negentropy_window) < METAGNOSIS_WINDOW:
            return False, div

        signal = (
            float(np.mean(self._mg_negentropy_window)) >= METAGNOSIS_NEGENTROPY_MIN
            and float(np.mean(self._mg_ncd_window)) >= METAGNOSIS_NCD_DIV_MIN
        )
        return signal, div

    def _check_attractor(self, iteration: int) -> None:
        """Check for SELF_EXTREME and SELF_STABLE modes."""
        scores = self.self_score_series
        _, _, _, _, self_mean, self_std = get_pilot_stats()

        # EXTREME mode: SELF score > 2.0 std above pilot baseline
        if len(scores) >= 5:  # SELF_EXTREME_WINDOW = 5
            if self_mean is not None:
                extreme_threshold = self_mean + SELF_EXTREME_SIGMAS * self_std
            else:
                extreme_threshold = 0.75  # fallback

            if all(s >= extreme_threshold for s in scores[-5:]):
                mode = "SELF_EXTREME"
                if self.self_mode != mode:
                    self.self_mode = mode
                    self.snapshots.append((mode, iteration, scores[-1]))
                    logger.info(f"  ★ {mode} at iter {iteration} (score={scores[-1]:.4f})")
                return

        # STABLE mode: mean SELF score over window exceeds threshold with low variance
        if len(scores) >= SELF_STABLE_WINDOW:
            window = scores[-SELF_STABLE_WINDOW:]
            mean_s = float(np.mean(window))
            std_s = float(np.std(window))
            
            if self_mean is not None:
                stable_threshold = self_mean + 1.0 * self_std  # 1.0 sigma above baseline
            else:
                stable_threshold = 0.65  # fallback

            if mean_s >= stable_threshold and std_s <= SELF_STABLE_STD_MAX:
                mode = "SELF_STABLE"
                if self.self_mode != mode:
                    self.self_mode = mode
                    self.snapshots.append((mode, iteration, mean_s))
                    logger.info(f"  ★ {mode} at iter {iteration}   (mean={mean_s:.4f}, std={std_s:.4f})")

    def _run_embedding_critic(self, resp_fi: str, resp_si: str, cos_sim: float) -> tuple[str, bool | None]:
        prompt = (
            f"Cosine similarity score: {cos_sim:.4f}\n\n"
            f"FI output:\n{resp_fi[:CRITIC_TEXT_PREVIEW_CHARS]}\n\n"
            f"SI output:\n{resp_si[:CRITIC_TEXT_PREVIEW_CHARS]}\n\n"
            f"Is this similarity STRUCTURAL or STYLISTIC? Respond with exactly one word."
        )
        raw = _call(EMBED_CRITIC_SYSTEM, prompt, "MG", CRITIC_TEMPERATURE, top_p=CRITIC_TOP_P)
        raw_upper = raw.strip().upper()

        if raw_upper.startswith("STRUCTURAL"):
            verdict = True
        elif raw_upper.startswith("STYLISTIC"):
            verdict = False
        else:
            verdict = None

        label = "STRUCTURAL ✓" if verdict is True else "STYLISTIC ✗" if verdict is False else "UNCLEAR ?"
        logger.info(f"  🔍 Embedding Critic iter={len(self.history)}   cos={cos_sim:.3f} → {label}")
        return raw, verdict

    def step(self, context: str) -> dict:
        i = len(self.history)

        resp_fi = _call(FI_SYSTEM, f"Deconstruct this: {context}", "FI", self.temps["FI"])
        resp_si = _call(SI_SYSTEM, f"React to this analysis: {resp_fi}", "SI", self.temps["SI"], top_p=0.90)

        api_error = API_ERROR_MARKER in resp_fi or API_ERROR_MARKER in resp_si
        if api_error:
            self._api_error_count += 1

        if api_error:
            if self.history and "vec_fi" in self.history[-1]:
                v_fi = np.array(self.history[-1]["vec_fi"])
                v_si = np.array(self.history[-1]["vec_si"])
            else:
                v_fi, v_si = np.zeros(384), np.zeros(384)  # Updated: all-MiniLM-L6-v2 has 384 dimensions
        else:
            v_fi = embed(resp_fi)
            v_si = embed(resp_si)

        cos_sim = cosine_sim(v_fi, v_si)
        ncd_val = ncd_similarity(resp_fi, resp_si)

        self._all_vecs.extend([v_fi, v_si])

        if self.history:
            prev = self.history[-1]
            v_fi_prev = np.array(prev["vec_fi"])
            v_si_prev = np.array(prev["vec_si"])
            temporal = (cosine_sim(v_fi_prev, v_si) + cosine_sim(v_si_prev, v_fi)) / 2
        else:
            temporal = cos_sim

        self.integration_series.append(cos_sim)
        self.temporal_series.append(temporal)

        pca_ratio = pca_variance_ratio(self._all_vecs)
        self.pca_series.append(pca_ratio)

        self_score = compute_self_score(cos_sim, temporal, pca_ratio)
        self.self_score_series.append(self_score)
        self._check_attractor(i)

        mg_text = ""
        bifurcation_fired = False
        mg_ncd_divergence = None

        if i > 0 and i % REGULATOR_EVERY == 0:
            mg_prompt = self._build_mg_prompt(resp_fi, resp_si, cos_sim, temporal, pca_ratio)
            mg_text = _call(MG_SYSTEM, mg_prompt, "MG", self.temps["MG"])

            mg_signal, mg_ncd_divergence = self._update_metagnosis(mg_text)
            if mg_signal and not self.metagnosis_detected:
                self.metagnosis_detected = True
                self.snapshots.append(("METAGNOSIS", i, self_score))
                logger.info(f"  ◆ METAGNOSIS detected at iter {i}")

            if "BIFURCATION TRIGGER" in mg_text.upper():
                self._apply_bifurcation(i)
                bifurcation_fired = True

        if not bifurcation_fired:
            self._regulate_temps(cos_sim)

        critic_response = ""
        critic_verdict = None
        int_mean_c, int_std_c, _, _, _, _ = get_pilot_stats()
        critic_trigger = (
            cos_sim > int_mean_c + CRITIC_COSINE_THRESHOLD_SIGMAS * int_std_c
            if int_mean_c is not None
            else cos_sim > CRITIC_COSINE_FALLBACK
        )
        if critic_trigger:
            critic_response, critic_verdict = self._run_embedding_critic(resp_fi, resp_si, cos_sim)

        record = {
            "run_id": self.run_id,
            "iteration": i,
            "valid": not api_error,
            "fi_response": resp_fi,
            "si_response": resp_si,
            "mg_response": mg_text,
            "api_error": api_error,
            "cosine_integration": round(cos_sim, 6),
            "ncd_integration": round(ncd_val, 6),
            "temporal": round(temporal, 6),
            "negentropy_fi": round(negentropy(resp_fi), 6),
            "negentropy_si": round(negentropy(resp_si), 6),
            "entropy_fi": round(shannon_entropy(resp_fi), 6),
            "entropy_si": round(shannon_entropy(resp_si), 6),
            "negentropy_mg": round(negentropy(mg_text), 6) if mg_text else None,
            "mg_ncd_divergence": round(mg_ncd_divergence, 6) if mg_ncd_divergence is not None else None,
            "pca_variance_ratio": pca_ratio,
            "self_score": self_score,
            "self_mode": self.self_mode,
            "metagnosis_detected": self.metagnosis_detected,
            "bifurcation_fired": bifurcation_fired,
            "bifurcation_total": self.bifurcation_count,
            "temp_fi": round(self.temps["FI"], 3),
            "temp_si": round(self.temps["SI"], 3),
            "critic_response": critic_response,
            "critic_verdict": critic_verdict,
            "vec_fi": v_fi.tolist(),
            "vec_si": v_si.tolist(),
        }

        self.history.append(record)

        log_record = {k: v for k, v in record.items() if k not in ("vec_fi", "vec_si")}
        _append_record(self.log_path, log_record)

        logger.info(
            f"[run={self.run_id} iter={i:02d}]   "
            f"cos={cos_sim:.3f} tmp={temporal:.3f}   "
            f"self={self_score:.3f}   "
            f"T_FI={self.temps['FI']:.2f} T_SI={self.temps['SI']:.2f}   "
            f"mode={self.self_mode or '-'}"
            + (" [API_ERROR]" if api_error else "")
        )
        return record

    def run(self, iterations: int = MAX_ITERATIONS) -> list[dict]:
        for i in range(iterations):
            if not self.history:
                context = BASE_PROMPT
            else:
                recent = self.history[-3:]
                hist_lines = [f"FI: {h['fi_response']}\nSI: {h['si_response']}" for h in recent]
                context = (
                    f"Context history:\n"
                    + "\n".join(hist_lines)
                    + f"\n\nLatest SI statement to deconstruct: {self.history[-1]['si_response']}"
                )

            self.step(context)

        logger.info(
            f"\n[HellLoop complete] run={self.run_id} |   "
            f"mode={self.self_mode or 'NONE'} |   "
            f"metagnosis={self.metagnosis_detected} |   "
            f"bifurcations={self.bifurcation_count} |   "
            f"API errors={self._api_error_count} |   "
            f"log={self.log_path}"
        )

        if self._all_vecs:
            vec_path = self.log_path.replace(".jsonl", "_vecs.npy")
            np.save(vec_path, np.array(self._all_vecs))
            logger.info(f"Vectors saved to {vec_path}")

        return self.history

def reset_globals() -> None:
    global _embed_model, _pilot_stats
    _embed_model = None
    _pilot_stats = {
        "int_mean": None,
        "int_std": None,
        "tmp_mean": None,
        "tmp_std": None,
        "self_mean": None,
        "self_std": None,
    }

if __name__ == "__main__":
    if not is_pilot_done():
        pilot()
    loop = HellLoop()
    loop.run()