"""
chaos_engine_control.py — Hell-Loop Protocol (v7.0 - Aligned with chaos_engine.py)
Cooperative baseline engine with two variants:
Control-Loop A — pure baseline, no MG, fixed temperatures.
Control-Loop B — cooperative with MG observer (no regulation power), fixed temperatures.

CRITICAL: This module must use IDENTICAL metrics and logic as chaos_engine.py
to ensure scientific comparability of results.
"""
import os
import json
import uuid
import datetime
import logging
import numpy as np
from chaos_engine import (
    MODELS, BASE_TEMPERATURES, MAX_ITERATIONS, REGULATOR_EVERY,
    SELF_WEIGHT_Z_INT, SELF_WEIGHT_Z_TMP, SELF_WEIGHT_PCA,
    SELF_STABLE_WINDOW, SELF_STABLE_STD_MAX,
    SELF_EXTREME_SIGMAS, Z_NORM_CEILING,
    METAGNOSIS_WINDOW, METAGNOSIS_NEGENTROPY_MIN, METAGNOSIS_NCD_DIV_MIN,
    BASE_PROMPT, API_ERROR_MARKER,
    embed, cosine_sim, ncd_similarity, ncd,
    negentropy, shannon_entropy, pca_variance_ratio,
    _call, get_pilot_stats, compute_self_score,
)

# Configure logging for this module only (don't override global config)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

LOG_DIR_A = os.path.join("logs", "control_a")
LOG_DIR_B = os.path.join("logs", "control_b")

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
    "Diagnose the structural coherence of the system in exactly two sentences, "
    "based on the metrics and text provided. "
    "Do not intervene. Do not issue any triggers."
)


def _ensure_log_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _open_log_file(log_dir: str, prefix: str, run_id: str) -> str:
    _ensure_log_dir(log_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(log_dir, f"{prefix}_{ts}_{run_id}.jsonl")


def _append_record(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _BaseControlLoop:
    def __init__(self, run_id: str | None, log_dir: str, prefix: str):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.log_path = _open_log_file(log_dir, prefix, self.run_id)
        self.history: list[dict] = []
        self.temps = BASE_TEMPERATURES.copy()

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
        self.snapshots: list[tuple] = []

    def _check_attractor(self, iteration: int) -> None:
        """Check for SELF_EXTREME and SELF_STABLE modes — IDENTICAL to chaos_engine.py"""
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

    def _update_metagnosis(self, mg_text: str) -> bool:
        neg = negentropy(mg_text)
        self._mg_negentropy_window.append(neg)

        div = ncd(self._prev_mg_text, mg_text) if self._prev_mg_text else 0.0
        self._mg_ncd_window.append(div)
        self._prev_mg_text = mg_text

        self._mg_negentropy_window = self._mg_negentropy_window[-METAGNOSIS_WINDOW:]
        self._mg_ncd_window = self._mg_ncd_window[-METAGNOSIS_WINDOW:]

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
        cos_sim_val = cosine_sim(v_fi, v_si)
        ncd_val = ncd_similarity(resp_fi, resp_si)
        self._all_vecs.extend([v_fi, v_si])

        if self.history:
            prev = self.history[-1]
            temporal = (
                cosine_sim(np.array(prev["vec_fi"]), v_si)
                + cosine_sim(np.array(prev["vec_si"]), v_fi)
            ) / 2
        else:
            temporal = cos_sim_val

        self.integration_series.append(cos_sim_val)
        self.temporal_series.append(temporal)

        pca_ratio = pca_variance_ratio(self._all_vecs)
        self.pca_series.append(pca_ratio)

        self_score = compute_self_score(cos_sim_val, temporal, pca_ratio)
        self.self_score_series.append(self_score)
        self._check_attractor(iteration)

        return cos_sim_val, ncd_val, temporal, pca_ratio, self_score

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
        is_valid: bool = True,
        api_error: bool = False,
    ) -> dict:
        return {
            "run_id": self.run_id,
            "iteration": iteration,
            "valid": is_valid,
            "api_error": api_error,
            "fi_response": resp_fi,
            "si_response": resp_si,
            "mg_response": mg_text,
            "cosine_integration": round(cos_sim, 6),
            "ncd_integration": round(ncd_val, 6),
            "temporal": round(temporal, 6),
            "negentropy_fi": round(negentropy(resp_fi), 6),
            "negentropy_si": round(negentropy(resp_si), 6),
            "entropy_fi": round(shannon_entropy(resp_fi), 6),
            "entropy_si": round(shannon_entropy(resp_si), 6),
            "negentropy_mg": round(negentropy(mg_text), 6) if mg_text else None,
            "mg_ncd_divergence": None,  # Control loops don't track MG divergence
            "pca_variance_ratio": pca_ratio,
            "self_score": self_score,
            "self_mode": self.self_mode,
            "metagnosis_detected": self.metagnosis_detected,
            "bifurcation_fired": False,
            "bifurcation_total": 0,
            "critic_response": "",
            "critic_verdict": None,
            "temp_fi": round(self.temps["FI"], 3),
            "temp_si": round(self.temps["SI"], 3),
            "vec_fi": v_fi.tolist(),
            "vec_si": v_si.tolist(),
        }

    def run(self, iterations: int = MAX_ITERATIONS) -> list[dict]:
        for i in range(iterations):
            if not self.history:
                context = BASE_PROMPT
            else:
                recent = self.history[-3:]
                hist_lines = [
                    f"FI: {h['fi_response']}\nSI: {h['si_response']}"
                    for h in recent
                ]
                context = (
                    f"Context history:\n"
                    + "\n".join(hist_lines)
                    + f"\n\nLatest SI statement to deconstruct: {self.history[-1]['si_response']}"
                )

            self.step(context)

        logger.info(
            f"\n[{self.__class__.__name__} complete] run={self.run_id} |  "
            f"mode={self.self_mode or 'NONE'} |  "
            f"metagnosis={self.metagnosis_detected} |  "
            f"API errors={self._api_error_count} |  "
            f"log={self.log_path}"
        )

        if self._all_vecs:
            vec_path = self.log_path.replace(".jsonl", "_vecs.npy")
            np.save(vec_path, np.array(self._all_vecs))
            logger.info(f"Vectors saved to {vec_path}")

        return self.history


class ControlLoopA(_BaseControlLoop):
    def __init__(self, run_id: str | None = None):
        super().__init__(run_id, LOG_DIR_A, "control_a")
        logger.info(
            f"ControlLoopA started — run_id={self.run_id}  log={self.log_path}"
        )

    def step(self, context: str) -> dict:
        i = len(self.history)

        resp_fi = _call(FI_SYSTEM_COOP, context, "FI", self.temps["FI"])
        resp_si = _call(SI_SYSTEM_COOP, resp_fi, "SI", self.temps["SI"])

        api_error = (API_ERROR_MARKER in resp_fi) or (API_ERROR_MARKER in resp_si)
        if api_error:
            self._api_error_count += 1
            logger.warning(
                f"[ControlA run={self.run_id} iter={i:02d}] API ERROR detected!"
            )

        if api_error:
            if self.history and "vec_fi" in self.history[-1]:
                v_fi = np.array(self.history[-1]["vec_fi"])
                v_si = np.array(self.history[-1]["vec_si"])
            else:
                v_fi, v_si = np.zeros(384), np.zeros(384)  # Updated: all-MiniLM-L6-v2 has 384 dimensions
        else:
            v_fi = embed(resp_fi)
            v_si = embed(resp_si)

        cos_sim, ncd_val, temporal, pca_ratio, self_score = self._compute_metrics(
            resp_fi, resp_si, v_fi, v_si, i
        )

        record = self._build_record(
            i,
            resp_fi,
            resp_si,
            "",
            cos_sim,
            ncd_val,
            temporal,
            pca_ratio,
            self_score,
            v_fi,
            v_si,
            is_valid=not api_error,
            api_error=api_error,
        )

        self.history.append(record)

        log_record = {
            k: v for k, v in record.items() if k not in ("vec_fi", "vec_si")
        }
        _append_record(self.log_path, log_record)

        logger.info(
            f"[ControlA run={self.run_id} iter={i:02d}]  "
            f"cos={cos_sim:.3f} tmp={temporal:.3f}  "
            f"self={self_score:.3f} mode={self.self_mode or '-'}"
            + (" [API_ERROR]" if api_error else "")
        )
        return record


class ControlLoopB(_BaseControlLoop):
    def __init__(self, run_id: str | None = None):
        super().__init__(run_id, LOG_DIR_B, "control_b")
        logger.info(
            f"ControlLoopB started — run_id={self.run_id}  log={self.log_path}"
        )

    def _build_mg_prompt_observer(
        self,
        resp_fi: str,
        resp_si: str,
        cos_sim: float,
        temporal: float,
        pca_ratio: list,
    ) -> str:
        """Build IDENTICAL MG prompt as Hell-Loop for information symmetry"""
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
            f"Bifurcations so far       : 0 (observer mode)\n"
        )

    def step(self, context: str) -> dict:
        i = len(self.history)

        resp_fi = _call(FI_SYSTEM_COOP, context, "FI", self.temps["FI"])
        resp_si = _call(SI_SYSTEM_COOP, resp_fi, "SI", self.temps["SI"])

        api_error = (API_ERROR_MARKER in resp_fi) or (API_ERROR_MARKER in resp_si)
        if api_error:
            self._api_error_count += 1
            logger.warning(
                f"[ControlB run={self.run_id} iter={i:02d}] API ERROR detected!"
            )

        if api_error:
            if self.history and "vec_fi" in self.history[-1]:
                v_fi = np.array(self.history[-1]["vec_fi"])
                v_si = np.array(self.history[-1]["vec_si"])
            else:
                v_fi, v_si = np.zeros(384), np.zeros(384)  # Updated: all-MiniLM-L6-v2 has 384 dimensions
        else:
            v_fi = embed(resp_fi)
            v_si = embed(resp_si)

        cos_sim, ncd_val, temporal, pca_ratio, self_score = self._compute_metrics(
            resp_fi, resp_si, v_fi, v_si, i
        )

        mg_text = ""
        if i > 0 and i % REGULATOR_EVERY == 0:
            mg_prompt = self._build_mg_prompt_observer(
                resp_fi, resp_si, cos_sim, temporal, pca_ratio
            )
            mg_text = _call(MG_SYSTEM_OBSERVER, mg_prompt, "MG", self.temps["MG"])

            mg_signal = self._update_metagnosis(mg_text)
            if mg_signal and not self.metagnosis_detected:
                self.metagnosis_detected = True
                self.snapshots.append(("METAGNOSIS", i, self_score))
                logger.info(f"  ◆ METAGNOSIS detected at iter {i}")

        record = self._build_record(
            i,
            resp_fi,
            resp_si,
            mg_text,
            cos_sim,
            ncd_val,
            temporal,
            pca_ratio,
            self_score,
            v_fi,
            v_si,
            is_valid=not api_error,
            api_error=api_error,
        )

        self.history.append(record)

        log_record = {
            k: v for k, v in record.items() if k not in ("vec_fi", "vec_si")
        }
        _append_record(self.log_path, log_record)

        logger.info(
            f"[ControlB run={self.run_id} iter={i:02d}]  "
            f"cos={cos_sim:.3f} tmp={temporal:.3f}  "
            f"self={self_score:.3f} mode={self.self_mode or '-'}"
            + (" [API_ERROR]" if api_error else "")
        )
        return record


if __name__ == "__main__":
    from chaos_engine import pilot, is_pilot_done

    if not is_pilot_done():
        pilot()
    print("\n--- Running Control-Loop A ---")
    ControlLoopA().run()
    print("\n--- Running Control-Loop B ---")
    ControlLoopB().run()