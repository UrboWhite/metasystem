"""
chaos_engine_control.py — Hell-Loop Protocol
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
    SELF_STABLE_WINDOW, SELF_STABLE_STD_MAX, SELF_STABLE_SIGMAS,
    SELF_EXTREME_SIGMAS, Z_NORM_CEILING,
    METAGNOSIS_WINDOW,
    BASE_PROMPT, API_ERROR_MARKER,
    embed, cosine_sim, ncd_similarity, ncd,
    negentropy, shannon_entropy, pca_variance_ratio,
    _call, get_pilot_stats, compute_self_score,
)

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
        scores = self.self_score_series
        _, _, _, _, self_mean, self_std = get_pilot_stats()

        is_extreme = False
        if len(scores) >= 5:
            extreme_threshold = (
                self_mean + SELF_EXTREME_SIGMAS * self_std if self_mean is not None else 0.75
            )
            is_extreme = all(s >= extreme_threshold for s in scores[-5:])

        is_stable = False
        if len(scores) >= SELF_STABLE_WINDOW:
            window = scores[-SELF_STABLE_WINDOW:]
            mean_s = float(np.mean(window))
            std_s = float(np.std(window))
            stable_threshold = (
                self_mean + SELF_STABLE_SIGMAS * self_std if self_mean is not None else 0.65
            )
            is_stable = mean_s >= stable_threshold and std_s <= SELF_STABLE_STD_MAX

        if is_extreme:
            new_mode = "SELF_EXTREME"
        elif is_stable:
            new_mode = "SELF_STABLE"
        else:
            new_mode = None

        if new_mode != self.self_mode:
            if new_mode is not None:
                self.snapshots.append((new_mode, iteration, scores[-1]))
                logger.info(f"  ★ {new_mode} at iter {iteration} (score={scores[-1]:.4f})")
            else:
                self.snapshots.append(("SELF_DROP", iteration, scores[-1]))
                logger.info(f"  ▼ SELF_DROP at iter {iteration} (score={scores[-1]:.4f}, was={self.self_mode})")
            self.self_mode = new_mode

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

        trend_neg = self._mg_negentropy_window[-1] > self._mg_negentropy_window[0]
        trend_ncd = self._mg_ncd_window[-1] > self._mg_ncd_window[0]

        signal = trend_neg and trend_ncd
        return signal, div

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
            prev_valid = None
            for h in reversed(self.history):
                if h.get("valid", True) and h.get("vec_fi") is not None:
                    prev_valid = h
                    break
            if prev_valid is not None:
                temporal = (
                    cosine_sim(np.array(prev_valid["vec_fi"]), v_si)
                    + cosine_sim(np.array(prev_valid["vec_si"]), v_fi)
                ) / 2
            else:
                temporal = cos_sim_val
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
        mg_ncd_divergence: float | None = None,
    ) -> dict:
        return {
            "run_id": self.run_id,
            "iteration": iteration,
            "valid": is_valid,
            "api_error": api_error,
            "fi_response": resp_fi,
            "si_response": resp_si,
            "mg_response": mg_text,
            "cosine_integration": round(cos_sim, 6) if cos_sim is not None else None,
            "ncd_integration": round(ncd_val, 6) if ncd_val is not None else None,
            "temporal": round(temporal, 6) if temporal is not None else None,
            "negentropy_fi": round(negentropy(resp_fi), 6) if resp_fi and resp_fi != API_ERROR_MARKER else None,
            "negentropy_si": round(negentropy(resp_si), 6) if resp_si and resp_si != API_ERROR_MARKER else None,
            "entropy_fi": round(shannon_entropy(resp_fi), 6) if resp_fi and resp_fi != API_ERROR_MARKER else None,
            "entropy_si": round(shannon_entropy(resp_si), 6) if resp_si and resp_si != API_ERROR_MARKER else None,
            "negentropy_mg": round(negentropy(mg_text), 6) if mg_text else None,
            "mg_ncd_divergence": round(mg_ncd_divergence, 6) if mg_ncd_divergence is not None else None,
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
            "vec_fi": v_fi.tolist() if v_fi is not None else None,
            "vec_si": v_si.tolist() if v_si is not None else None,
        }

    def run(self, iterations: int = MAX_ITERATIONS) -> list[dict]:
        for i in range(iterations):
            if not self.history:
                context = BASE_PROMPT
            else:
                context = self.history[-1]["si_response"]
                if self.history[-1].get("si_response", "") == API_ERROR_MARKER:
                    for h in reversed(self.history[:-1]):
                        if h.get("valid", True) and h.get("si_response") != API_ERROR_MARKER:
                            context = h["si_response"]
                            break

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

            record = {
                "run_id": self.run_id,
                "iteration": i,
                "valid": False,
                "api_error": True,
                "fi_response": resp_fi,
                "si_response": resp_si,
                "mg_response": "",
                "cosine_integration": None,
                "ncd_integration": None,
                "temporal": None,
                "negentropy_fi": None,
                "negentropy_si": None,
                "entropy_fi": None,
                "entropy_si": None,
                "negentropy_mg": None,
                "mg_ncd_divergence": None,
                "pca_variance_ratio": [],
                "self_score": None,
                "self_mode": self.self_mode,
                "metagnosis_detected": self.metagnosis_detected,
                "bifurcation_fired": False,
                "bifurcation_total": 0,
                "critic_response": "",
                "critic_verdict": None,
                "temp_fi": round(self.temps["FI"], 3),
                "temp_si": round(self.temps["SI"], 3),
                "vec_fi": None,
                "vec_si": None,
            }
            self.history.append(record)
            _append_record(self.log_path, {k: v for k, v in record.items() if k not in ("vec_fi", "vec_si")})
            return record

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
            is_valid=True,
            api_error=False,
            mg_ncd_divergence=None,
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

            record = {
                "run_id": self.run_id,
                "iteration": i,
                "valid": False,
                "api_error": True,
                "fi_response": resp_fi,
                "si_response": resp_si,
                "mg_response": "",
                "cosine_integration": None,
                "ncd_integration": None,
                "temporal": None,
                "negentropy_fi": None,
                "negentropy_si": None,
                "entropy_fi": None,
                "entropy_si": None,
                "negentropy_mg": None,
                "mg_ncd_divergence": None,
                "pca_variance_ratio": [],
                "self_score": None,
                "self_mode": self.self_mode,
                "metagnosis_detected": self.metagnosis_detected,
                "bifurcation_fired": False,
                "bifurcation_total": 0,
                "critic_response": "",
                "critic_verdict": None,
                "temp_fi": round(self.temps["FI"], 3),
                "temp_si": round(self.temps["SI"], 3),
                "vec_fi": None,
                "vec_si": None,
            }
            self.history.append(record)
            _append_record(self.log_path, {k: v for k, v in record.items() if k not in ("vec_fi", "vec_si")})
            return record

        v_fi = embed(resp_fi)
        v_si = embed(resp_si)

        cos_sim, ncd_val, temporal, pca_ratio, self_score = self._compute_metrics(
            resp_fi, resp_si, v_fi, v_si, i
        )

        mg_text = ""
        mg_ncd_divergence = None
        if i > 0 and i % REGULATOR_EVERY == 0:
            mg_prompt = self._build_mg_prompt_observer(
                resp_fi, resp_si, cos_sim, temporal, pca_ratio
            )
            mg_text = _call(MG_SYSTEM_OBSERVER, mg_prompt, "MG", self.temps["MG"])

            if not mg_text or mg_text == API_ERROR_MARKER:
                logger.warning(
                    f"[ControlB run={self.run_id} iter={i:02d}] MG returned empty/invalid response."
                )
                mg_text = ""
            else:
                mg_signal, mg_ncd_divergence = self._update_metagnosis(mg_text)
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
            is_valid=True,
            api_error=False,
            mg_ncd_divergence=mg_ncd_divergence,
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
