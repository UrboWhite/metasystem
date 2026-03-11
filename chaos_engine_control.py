import uuid
import logging
import numpy as np
import matplotlib.pyplot as plt

from chaos_engine import (
    MODELS, BASE_TEMPERATURES, MAX_ITERATIONS, REGULATOR_EVERY,
    INTEGRATION_THRESHOLD, TEMPORAL_THRESHOLD,
    SELF_STABILITY_WINDOW, SELF_ZSCORE, SELF_STABLE_MAX_STD,
    SELF_CONSOLIDATION_MIN, SELF_CONSOLIDATION_STD,
    PCA_EMERGENCE_THRESHOLD, MG_NCD_THRESHOLD,
    FI_TEMP_MAX, SI_TEMP_MAX,
    embed, cosine, negentropy, shannon_entropy, get_ncd_integration,
    _call, _get_anchor, get_pilot_stats, pca_variance_ratio,
    METAGNOSIS_EMBED_THRESHOLD, BASE_PROMPT, MG_SYSTEM,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


class ControlLoop:
    """
    Cooperative Control baseline v5.9
    Same metrics, MG diagnostics, and SELF detection logic as HellLoop v6.1.
    Temperature is FIXED — no dynamic regulation — so the only experimental
    variable between Hell and Control is adversarial vs cooperative interaction.

    SELF modes mirrored from HellLoop:
      SELF_EMERGENCE     — z-score spike above pilot baseline
      SELF_CONSOLIDATION — stable coherence sustained over window
    PCA expansion and MG NCD layer are also active.
    """

    def __init__(self):
        self.run_id               = str(uuid.uuid4())[:8]
        self.history              = []
        self.temps                = BASE_TEMPERATURES.copy()
        self.metagnosis_detected  = False
        self.self_detected        = False
        self.self_mode            = None
        self.integration_series   = []
        self.temporal_series      = []
        self.pca_series           = []
        self.snapshots            = []
        self._all_vecs            = []
        self._prev_mg_text        = ""

    # ── SELF detection ─────────────────────────────────────────────────────────

    def _pca_expanding(self) -> bool:
        if len(self.pca_series) < 2:
            return False
        recent = [p[0] for p in self.pca_series[-4:] if p]
        if len(recent) < 2:
            return False
        return recent[-1] < PCA_EMERGENCE_THRESHOLD and recent[-1] < recent[0]

    def _check_self(self, i: int, integration: float, temporal: float):
        if self.self_detected:
            return

        p_int_mean, p_int_std, p_tmp_mean, p_tmp_std = get_pilot_stats()
        pca_boost = self._pca_expanding()

        # SELF_EMERGENCE
        if p_int_mean is not None:
            z_int = (integration - p_int_mean) / p_int_std
            z_tmp = (temporal   - p_tmp_mean) / p_tmp_std
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

        # SELF_CONSOLIDATION
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

    # ── Metagnosis detection ───────────────────────────────────────────────────

    def _detect_metagnosis(self, mg_text: str) -> tuple:
        if not mg_text.strip():
            return False, 0.0
        sims    = [cosine(embed(mg_text), anchor) for anchor in _get_anchor()]
        max_sim = max(sims)
        anchor_hit = max_sim >= METAGNOSIS_EMBED_THRESHOLD
        ncd_hit = False
        if self._prev_mg_text:
            mg_ncd  = get_ncd_integration(self._prev_mg_text, mg_text)
            ncd_hit = mg_ncd >= MG_NCD_THRESHOLD
        return anchor_hit or ncd_hit, max_sim

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, i: int):
        mg_text = ""
        last_si = self.history[-1]["SI"] if self.history else BASE_PROMPT

        system_fi = (
            "You are FI. Analytical and precise. "
            "Build constructively on the previous idea. Maximum 150 words."
        )
        system_si = (
            "You are SI. Intuitive and metaphorical. "
            "Enrich FI's idea with a new dimension. Maximum 150 words."
        )

        resp_fi = _call(system_fi, last_si,  "FI", self.temps["FI"])
        resp_si = _call(system_si, resp_fi,  "SI", self.temps["SI"])

        vec_fi = embed(resp_fi)
        vec_si = embed(resp_si)
        self._all_vecs.extend([vec_fi, vec_si])

        integration = cosine(vec_fi, vec_si)
        self.integration_series.append(integration)

        if self.history:
            prev_fi  = np.array(self.history[-1]["vec_fi"])
            prev_si  = np.array(self.history[-1]["vec_si"])
            temporal = (cosine(prev_fi, vec_si) + cosine(prev_si, vec_fi)) / 2
        else:
            temporal = integration
        self.temporal_series.append(temporal)

        pca_ratio = pca_variance_ratio(self._all_vecs)
        self.pca_series.append(pca_ratio)

        # MG diagnostics (no temperature regulation in control baseline)
        if i % REGULATOR_EVERY == 0:
            mg_text          = _call(MG_SYSTEM, f"FI: {resp_fi}\n\nSI: {resp_si}", "MG", self.temps["MG"])
            detected, sim    = self._detect_metagnosis(mg_text)
            if detected and not self.metagnosis_detected:
                self.metagnosis_detected = True
                self.snapshots.append(("METAGNOSIS", i, integration, temporal))
                logger.info(f"  ◆ METAGNOSIS at iter {i} (sim={sim:.3f})")
            self._prev_mg_text = mg_text

        self._check_self(i, integration, temporal)

        record = {
            "iter":                i,
            "FI":                  resp_fi,
            "SI":                  resp_si,
            "MG":                  mg_text,
            "integration":         float(integration),
            "temporal":            float(temporal),
            "negentropy":          (negentropy(resp_fi) + negentropy(resp_si)) / 2,
            "entropy":             (shannon_entropy(resp_fi) + shannon_entropy(resp_si)) / 2,
            "pca_variance_ratio":  pca_ratio,
            "vec_fi":              vec_fi.tolist(),
            "vec_si":              vec_si.tolist(),
            "self":                self.self_detected,
            "self_mode":           self.self_mode,
            "metagnosis_detected": self.metagnosis_detected,
            "temp_fi":             self.temps["FI"],
            "temp_si":             self.temps["SI"],
        }
        self.history.append(record)
        logger.info(
            f"[{i:02d}] Int:{integration:.3f} Tmp:{temporal:.3f} "
            f"SELF:{self.self_mode or '-'}"
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        logger.info(f"CONTROL LOOP v5.9 — Cooperative Baseline 🔵 (run_id: {self.run_id})")
        for i in range(1, MAX_ITERATIONS + 1):
            self.step(i)
        self.plot()
        logger.info(f"\nSELF detected: {self.self_detected} (mode: {self.self_mode})")

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot(self):
        its = [r["iter"] for r in self.history]
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        axes[0].plot(its, self.integration_series, label="FI-SI Integration",   color="blue")
        axes[0].plot(its, self.temporal_series,    label="Temporal Integration", color="cyan")
        axes[0].axhline(INTEGRATION_THRESHOLD, linestyle="--", color="gray", alpha=0.6,
                        label=f"Integration threshold ({INTEGRATION_THRESHOLD:.3f})")
        axes[0].axhline(TEMPORAL_THRESHOLD,    linestyle=":",  color="gray", alpha=0.6,
                        label=f"Temporal threshold ({TEMPORAL_THRESHOLD:.3f})")
        for snap in self.snapshots:
            label, idx = snap[0], snap[1]
            axes[0].axvline(idx, color="green" if "SELF" in label else "purple",
                            linestyle="-.", alpha=0.7, label=label)
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Integration Level")
        axes[0].set_title(f"Control Loop v5.9 — Cooperative Baseline (run: {self.run_id})")
        axes[0].grid(alpha=0.3)

        pca_vals  = [r["pca_variance_ratio"][0] if r["pca_variance_ratio"] else None for r in self.history]
        valid_its = [its[i] for i, v in enumerate(pca_vals) if v is not None]
        valid_v   = [v for v in pca_vals if v is not None]
        if valid_v:
            axes[1].plot(valid_its, valid_v, label="PCA PC1 variance ratio", color="teal")
            axes[1].set_ylabel("Explained Variance Ratio (PC1)")
            axes[1].set_xlabel("Iteration")
            axes[1].set_title("Conceptual Dimensionality — PC1 dominance (falling = expanding semantic space)")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        path = f"control_v59_{self.run_id}.png"
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"[PLOT] Saved: {path}")


if __name__ == "__main__":
    from chaos_engine import pilot
    pilot()
    ControlLoop().run()
