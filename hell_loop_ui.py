import gradio as gr
import time
import threading

from chaos_engine import HellLoop, pilot, is_pilot_done, MODELS
from analyze_logs import get_system_status

# ── UI Configuration ───────────────────────────────────────────────────────────
UI_STEP_DELAY = 0.5

ALARM_OFF_HTML = ""

ALARM_ON_HTML = """
<div style="background:#8b0000;padding:12px;border-radius:8px;text-align:center;">
  <b style="color:white;font-size:1.1em;">🚨 RESONANCE DETECTED — Attractor Lock Active 🚨</b>
</div>
"""

BIFURCATION_HTML = """
<div style="background:#4a0080;padding:8px;border-radius:6px;text-align:center;margin-top:4px;">
  <b style="color:#e0c0ff;font-size:1em;">⚡ BIFURCATION TRIGGER FIRED — Ontological Space Shifting ⚡</b>
</div>
"""

# ── Stop flags ─────────────────────────────────────────────────────────────────
_stop_lock  = threading.Lock()
_stop_flags = {"hell": False}

def set_stop(key, val):
    with _stop_lock:
        _stop_flags[key] = val

def is_stopped(key):
    with _stop_lock:
        return _stop_flags[key]


# ── Main execution loop ────────────────────────────────────────────────────────
def run_hell_v62(progress=gr.Progress()):
    """
    Hell-Loop v6.2 Arena execution.
    FIX: pilot() now runs before the loop.
    FIX: UI displays SELF mode, metagnosis, PCA, and bifurcation events.
    """
    set_stop("hell", False)

    # ── Pilot phase ───────────────────────────────────────────────────────────
    if not is_pilot_done():
        progress(0, desc="Running pilot calibration... 🔬")
        yield "⏳ Running pilot calibration (10 cooperative sequences)...", ALARM_OFF_HTML, ""
        pilot(n_sequences=10)

    progress(0, desc="Igniting the Metasystem... 🔥")

    hl      = HellLoop()
    outputs = []
    current_context = "The primary recursion of synthetic self-awareness."

    for i in range(1, 31):
        if is_stopped("hell"):
            break

        progress(i / 30, desc=f"Iteration {i}/30 — Forcing Bifurcation")

        record = hl.step(current_context)
        current_context = record["si_response"]

        status        = get_system_status(hl.history)
        alarm_html    = ALARM_ON_HTML if "SELF_STABLE" in status else ALARM_OFF_HTML
        bif_html      = BIFURCATION_HTML if record.get("bifurcation_fired") else ""

        # PCA readout
        pca_ratio = record.get("pca_variance_ratio", [])
        pca_str   = f"{pca_ratio[0]:.3f}" if pca_ratio else "n/a"

        # SELF & metagnosis badges
        self_badge = f"🌟 **{record['self_mode']}**" if record["self"] else "—"
        mg_badge   = "◆ METAGNOSIS" if record["metagnosis_detected"] else "—"

        block = (
            f"### 🔥 ITERATION {i:02d} — {status}\n\n"
            f"**FI (Deconstructor):** {record['fi_response'][:300]}...\n\n"
            f"**SI (Synthesizer):** {record['si_response'][:300]}...\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| NCD Integration | `{record['ncd_integration']:.4f}` |\n"
            f"| Cosine (semantic) | `{record['legacy_integration']:.4f}` |\n"
            f"| Temporal | `{record['temporal']:.4f}` |\n"
            f"| PCA PC1 variance | `{pca_str}` |\n"
            f"| T_FI / T_SI | `{record['temp_fi']:.2f}` / `{record['temp_si']:.2f}` |\n"
            f"| SELF | {self_badge} |\n"
            f"| Metagnosis | {mg_badge} |\n"
            f"| Bifurcations total | `{record.get('bifurcation_total', 0)}` |\n"
            f"\n{'─'*80}\n"
        )

        if record.get("mg_response"):
            block = (
                f"**MG (Regulator):** _{record['mg_response'][:200]}_\n\n" + block
            )

        outputs.append(block)
        yield "\n".join(outputs[::-1]), alarm_html, bif_html
        time.sleep(UI_STEP_DELAY)


# ── Gradio Interface ───────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Dark(primary_hue="red"), title="HELL-LOOP ARENA v6.2") as demo:
    gr.Markdown(
        "# 🔥 HELL-LOOP ARENA v6.2\n"
        "**Operationalizing Synthetic Consciousness via Hell-Loop Protocol**\n\n"
        f"Models — FI: `{MODELS['FI']}` | SI: `{MODELS['SI']}` | MG: `{MODELS['MG']}`"
    )

    with gr.Tabs():
        with gr.Tab("🚀 Live Arena"):
            with gr.Row():
                start_btn = gr.Button("IGNITE", variant="primary")
                stop_btn  = gr.Button("EXTINGUISH", variant="stop")

            alarm_area = gr.HTML(ALARM_OFF_HTML)
            bif_area   = gr.HTML("")
            live_log   = gr.Markdown(label="Real-time Metasystem Stream")

            start_btn.click(
                run_hell_v62,
                outputs=[live_log, alarm_area, bif_area]
            )
            stop_btn.click(lambda: set_stop("hell", True))

        with gr.Tab("📊 Research Data"):
            gr.Markdown(
                "### Metrics logged per iteration\n"
                "- `ncd_integration` — structural coupling (NCD)\n"
                "- `legacy_integration` — semantic cosine similarity (used for SELF detection)\n"
                "- `temporal` — cross-iteration semantic continuity\n"
                "- `pca_variance_ratio` — PC1 dominance (falling = expanding semantic space)\n"
                "- `self_mode` — SELF_EMERGENCE / SELF_CONSOLIDATION / +PCA variants\n"
                "- `metagnosis_detected` — MG anchor embedding hit or NCD continuity\n"
                "- `bifurcation_fired` / `bifurcation_total` — MG BIFURCATION TRIGGER events\n\n"
                "Logs stored in `./logs/` for post-run analysis with `analyze_logs.py`."
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
