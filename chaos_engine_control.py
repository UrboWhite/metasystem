"""
chaos_engine_control.py — Hell-Loop Protocol v6.2
Cooperative baseline engine with two variants:

  Control-Loop A — pure baseline, no MG, fixed temperatures.
  Control-Loop B — cooperative with MG observer (no regulation power),
                   fixed temperatures.
"""

import os
import json
import uuid
import logging
import datetime
import numpy as np

from chaos_engine import (
    MODELS, BASE_TEMPERATURES, MAX_ITERATIONS, REGULATOR_EVERY,
    SELF_WEIGHT_Z_INT, SELF_WEIGHT_Z_TMP, SELF_WEIGHT_PCA,
    SELF_STABLE_WINDOW, SELF_STABLE_MEAN_MIN, SELF_STABLE_STD_MAX,
    SELF_EXTREME_WINDOW, SELF_ZSCORE_THRESHOLD,
    METAGNOSIS_WINDOW, METAGNOSIS_NEGENTROPY_MIN, METAGNOSIS_NCD_DIV_MIN,
    BASE_PROMPT,
    embed, cosine_sim, ncd_similarity, ncd,
    negentropy, shannon_entropy, pca_variance_ratio,
    _call, get_pilot_stats,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

LOG_DIR_A = os.path.join("logs", "control_a")
LOG_DIR_B = os.path.join("logs", "control_b")

# ── Cooperative system prompts ─────────────────────────────────────────────────
FI_SYSTEM_COOP = (
    "You are FI. Analytical and precise. "
    "Build constructively on the previous idea. "
    "Deepen and expand it. Maximum 150 words."
)

SI_SYSTEM_COOP = (
    "You are SI. Intuitive and metaphorical. "
    "Enrich the previous idea with a new dimension. "
    "Do not contradict — synthesize. Maximum 150 words."
)

MG_SYSTEM_OBSERVER = (
    "You are MG (Metagnosis), a passive observer. "
    "Diagnose the structural coherence of the system in exactly two sentences. "
    "Do not intervene. Do not issue any triggers."
)


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _ensure_log_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _open_log_file(log_dir: str, prefix: str, run_id: str) -> str:
    _ensure_log_dir(log_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{prefix}_{ts}_{run_id}.jsonl")

def _append_record(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _compute_self_score(
    cos_sim: float,
    temporal: float,
    pca_ratio: list,
) -> float:
    int_mean, int_std, tmp_mean, tmp_std = get_pilot_stats()

    if int_mean is not None:
        z_int_norm = max(0.0, min(1.0, (cos_sim  - int_mean) / int_std  / 3.0))
        z_tmp_norm = max(0.0, min(1.0, (temporal - tmp_mean) / tmp_std  / 3.0))
    else:
        z_int_norm = max(0.0, min(1.0, cos_sim))
        z_tmp_norm = max(0.0, min(1.0, temporal))

    pca_signal = max(0.0, min(1.0, 1.0 - pca_ratio[0])) if pca_ratio else 0.0

    return round(
        SELF_WEIGHT_Z_INT * z_int_norm
        + SELF_WEIGHT_Z_TMP * z_tmp_norm
        + SELF_WEIGHT_PCA  * pca_signal,
        4,
    )


# ── Base class shared by both variants ────────────────────────────────────────
class _BaseControlLoop:

    def __init__(self, run_id: str | None, log_dir: str, prefix: str):
        self.run_id   = run_id or uuid.uuid4().hex[:8]
        self.log_path = _open_log_file(log_dir, prefix, self.run_id)
        self.history  : list[dict]  = []
        self.temps    = BASE_TEMPERATURES.copy()   # fixed, never changed

        self.integration_series : list[float] = []
        self.temporal_series    : list[float] = []
        self.pca_series         : list[list]  = []
        self.self_score_series  : list[float] = []

        self._mg_negentropy_window : list[float] = []
        self._mg_ncd_window        : list[float] = []
        self._prev_mg_text         : str         = ""
        self._all_vecs             : list        = []

        self.self_mode           : str | None = None
        self.metagnosis_detected : bool       = False
        self.snapshots           : list[tuple] = []

    def _check_attractor(self, iteration: int) -> None:
        scores = self.self_score_series

        if len(scores) >= SELF_EXTREME_WINDOW:
            int_mean, int_std, _, _ = get_pilot_stats()
            threshold = (
                max(0.0, min(1.0, SELF_ZSCORE_THRESHOLD / 3.0))
                * (SELF_WEIGHT_Z_INT + SELF_WEIGHT_Z_TMP)
                if int_mean is not None else 0.55
            )
            if all(s >= threshold for s in scores[-SELF_EXTREME_WINDOW:]):
                mode = "SELF_EXTREME"
                if self.self_mode != mode:
                    self.self_mode = mode
                    self.snapshots.append((mode, iteration, scores[-1]))
                    logger.info(
                        f"  ★ {mode} at iter {iteration} "
                        f"(score={scores[-1]:.4f})"
                    )
                return

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

    def _update_metagnosis(self, mg_text: str) -> bool:
        neg = negentropy(mg_text)
        self._mg_negentropy_window.append(neg)

        div = ncd(self._prev_mg_text, mg_text) if self._prev_mg_text else 0.0
        self._mg_ncd_window.append(div)
        self._prev_mg_text = mg_text

        self._mg_negentropy_window = \
            self._mg_negentropy_window[-METAGNOSIS_WINDOW:]
        self._mg_ncd_window = \
            self._mg_ncd_window[-METAGNOSIS_WINDOW:]

        if len(self._mg_negentropy_window) < METAGNOSIS_WINDOW:
            return False

        return (
            float(np.mean(self._mg_negentropy_window)) >= METAGNOSIS_NEGENTROPY_MIN
            and float(np.mean(self._mg_ncd_window)) >= METAGNOSIS_NCD_DIV_MIN
        )

    def _compute_metrics(
        self,
        resp_fi: str,
        resp_si: str,
        v_fi: np.ndarray,
        v_si: np.ndarray,
        iteration: int,
    ) -> tuple:
        cos_sim = cosine_sim(v_fi, v_si)
        ncd_val = ncd_similarity(resp_fi, resp_si)
        self._all_vecs.extend([v_fi, v_si])

        if self.history:
            prev     = self.history[-1]
            temporal = (
                cosine_sim(np.array(prev["vec_fi"]), v_si)
                + cosine_sim(np.array(prev["vec_si"]), v_fi)
            ) / 2
        else:
            temporal = cos_sim

        self.integration_series.append(cos_sim)
        self.temporal_series.append(temporal)

        pca_ratio  = pca_variance_ratio(self._all_vecs)
        self.pca_series.append(pca_ratio)

        self_score = _compute_self_score(cos_sim, temporal, pca_ratio)
        self.self_score_series.append(self_score)
        self._check_attractor(iteration)

        return cos_sim, ncd_val, temporal, pca_ratio, self_score

    def _build_record(
        self,
        iteration: int,
        resp_fi: str,
        resp_si: str,
        mg_text: str,
        cos_sim: float,
        ncd_val: float,
        temporal: float,
        pca_ratio: list,
        self_score: float,
        v_fi: np.ndarray,
        v_si: np.ndarray,
    ) -> dict:
        return {
            "run_id"              : self.run_id,
            "iteration"           : iteration,
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
            "temp_fi"             : round(self.temps["FI"], 3),
            "temp_si"             : round(self.temps["SI"], 3),
            "vec_fi"              : v_fi.tolist(),
            "vec_si"              : v_si.tolist(),
        }

    def run(self, iterations: int = MAX_ITERATIONS) -> list[dict]:
        context = BASE_PROMPT
        for _ in range(iterations):
            record  = self.step(context)
            context = record["si_response"]
        logger.info(
            f"\n[{self.__class__.__name__} complete] run={self.run_id} | "
            f"mode={self.self_mode or 'NONE'} | "
            f"metagnosis={self.metagnosis_detected} | "
            f"log={self.log_path}"
        )
        return self.history


# ── Control-Loop A: pure baseline, no MG ──────────────────────────────────────
class ControlLoopA(_BaseControlLoop):
    """
    Cooperative baseline without MG.
    Fixed temperatures throughout. No regulatory intervention of any kind.
    """

    def __init__(self, run_id: str | None = None):
        super().__init__(run_id, LOG_DIR_A, "control_a")
        logger.info(
            f"ControlLoopA started — run_id={self.run_id}  log={self.log_path}"
        )

    def step(self, context: str) -> dict:
        i = len(self.history)

        resp_fi = _call(FI_SYSTEM_COOP, context,  "FI", self.temps["FI"])
        resp_si = _call(SI_SYSTEM_COOP, resp_fi,  "SI", self.temps["SI"])

        v_fi = embed(resp_fi)
        v_si = embed(resp_si)

        cos_sim, ncd_val, temporal, pca_ratio, self_score = \
            self._compute_metrics(resp_fi, resp_si, v_fi, v_si, i)

        record = self._build_record(
            i, resp_fi, resp_si, "",
            cos_sim, ncd_val, temporal, pca_ratio, self_score,
            v_fi, v_si,
        )
        self.history.append(record)
        _append_record(self.log_path, record)

        logger.info(
            f"[ControlA run={self.run_id} iter={i:02d}] "
            f"cos={cos_sim:.3f} tmp={temporal:.3f} "
            f"self={self_score:.3f} mode={self.self_mode or '-'}"
        )
        return record


# ── Control-Loop B: cooperative + MG observer (no regulation) ─────────────────
class ControlLoopB(_BaseControlLoop):
    """
    Cooperative baseline with MG as a passive observer.
    MG diagnoses the system every REGULATOR_EVERY iterations but cannot
    issue BIFURCATION TRIGGER and has no effect on temperatures.
    Metagnosis is still detected structurally for comparative analysis.
    """

    def __init__(self, run_id: str | None = None):
        super().__init__(run_id, LOG_DIR_B, "control_b")
        logger.info(
            f"ControlLoopB started — run_id={self.run_id}  log={self.log_path}"
        )

    def step(self, context: str) -> dict:
        i = len(self.history)

        resp_fi = _call(FI_SYSTEM_COOP, context,  "FI", self.temps["FI"])
        resp_si = _call(SI_SYSTEM_COOP, resp_fi,  "SI", self.temps["SI"])

        v_fi = embed(resp_fi)
        v_si = embed(resp_si)

        cos_sim, ncd_val, temporal, pca_ratio, self_score = \
            self._compute_metrics(resp_fi, resp_si, v_fi, v_si, i)

        mg_text = ""
        if i > 0 and i % REGULATOR_EVERY == 0:
            mg_prompt = (
                f"FI response:\n{resp_fi}\n\n"
                f"SI response:\n{resp_si}"
            )
            mg_text = _call(MG_SYSTEM_OBSERVER, mg_prompt, "MG", self.temps["MG"])

            mg_signal = self._update_metagnosis(mg_text)
            if mg_signal and not self.metagnosis_detected:
                self.metagnosis_detected = True
                self.snapshots.append(("METAGNOSIS", i, self_score))
                logger.info(f"  ◆ METAGNOSIS detected at iter {i}")

        record = self._build_record(
            i, resp_fi, resp_si, mg_text,
            cos_sim, ncd_val, temporal, pca_ratio, self_score,
            v_fi, v_si,
        )
        self.history.append(record)
        _append_record(self.log_path, record)

        logger.info(
            f"[ControlB run={self.run_id} iter={i:02d}] "
            f"cos={cos_sim:.3f} tmp={temporal:.3f} "
            f"self={self_score:.3f} mode={self.self_mode or '-'}"
        )
        return record


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from chaos_engine import pilot, is_pilot_done
    if not is_pilot_done():
        pilot()
    print("\n--- Running Control-Loop A ---")
    ControlLoopA().run()
    print("\n--- Running Control-Loop B ---")
    ControlLoopB().run()
