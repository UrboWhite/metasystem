"""
hell_loop_ui.py — Hell-Loop Protocol Web Interface v6.1
Three-tab Gradio interface:
  Tab 1 - Hell-Loop Live Arena      (adversarial run, real-time stream)
  Tab 2 - Control-Loop Baseline     (cooperative run, real-time stream)
  Tab 3 - Batch Runner              (automated N×N batch with progress log)

Pilot calibration runs once per session before the first run of any kind.
"""

import gradio as gr
import time
import json
import threading
from datetime import datetime
from pathlib import Path

from chaos_engine import (
    HellLoop, MAX_ITERATIONS, BASE_PROMPT,
    pilot, is_pilot_done,
)
from chaos_engine_control import ControlLoop
from analyze_logs import get_system_status, save_jsonl, compare, print_comparison

# ── UI Configuration ───────────────────────────────────────────────────────────
UI_STEP_DELAY   = 0.3   # seconds between yielded UI updates
LOG_DIR         = Path("logs")

# ── Alarm HTML ─────────────────────────────────────────────────────────────────
ALARM_OFF_HTML = ""
ALARM_ON_HTML  = """
<div style="background:#8b0000;padding:12px;border-radius:8px;text-align:center;animation:pulse 1s infinite;">
  <b style="color:white;font-size:1.1em;">🚨 RESONANCE DETECTED — Attractor Lock Active 🚨</b>
</div>
"""
METAGNOSIS_HTML = """
<div style="background:#4b0082;padding:12px;border-radius:8px;text-align:center;">
  <b style="color:white;font-size:1.1em;">◆ METAGNOSIS ACTIVE — System Self-Observing ◆</b>
</div>
"""

# ── Stop flags ─────────────────────────────────────────────────────────────────
_stop_lock  = threading.Lock()
_stop_flags = {"hell": False, "ctrl": False, "batch": False}

def set_stop(key: str, val: bool):
    with _stop_lock:
        _stop_flags[key] = val

def is_stopped(key: str) -> bool:
    with _stop_lock:
        return _stop_flags[key]

# ── Pilot (once per session) ───────────────────────────────────────────────────
_pilot_lock = threading.Lock()

def ensure_pilot(log_output: list) -> list:
    """Run pilot calibration if not yet done. Thread-safe."""
    with _pilot_lock:
        if not is_pilot_done():
            log_output.append("⚙️ **Running pilot calibration...**\n")
            pilot(n_sequences=10)
            log_output.append("✅ **Pilot calibration complete.**\n\n")
    return log_output

# ── Alarm state helper ─────────────────────────────────────────────────────────
def _alarm_html(record: dict, status: str) -> str:
    if record.get("metagnosis_detected"):
        return METAGNOSIS_HTML
    if record.get("self") or "SELF_STABLE" in status:
        return ALARM_ON_HTML
    return ALARM_OFF_HTML

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Hell-Loop Live Arena
# ══════════════════════════════════════════════════════════════════════════════

def run_hell(progress=gr.Progress()):
    set_stop("hell", False)
    outputs = []

    outputs = ensure_pilot(outputs)
    yield "\n".join(outputs), ALARM_OFF_HTML

    hl      = HellLoop()
    context = BASE_PROMPT
    log_dir = LOG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_hell")
    log_dir.mkdir(parents=True, exist_ok=True)

    for i in range(MAX_ITERATIONS):
        if is_stopped("hell"):
            outputs.append("🛑 **Run stopped by user.**\n")
            break

        progress(i / MAX_ITERATIONS, desc=f"Iteration {i} / {MAX_ITERATIONS}")
        record  = hl.step(context)
        context = record["si_response"]
        status  = get_system_status(hl.history)
        alarm   = _alarm_html(record, status)

        block = (
            f"### 🔥 ITER {i:02d} — {status}  "
            f"| SELF: `{record['self_mode'] or '—'}` "
            f"| MG: `{'✓' if record['metagnosis_detected'] else '—'}`\n\n"
            f"**FI (Deconstructor):** {record['fi_response'][:300]}...\n\n"
            f"**SI (Synthesizer):** {record['si_response'][:300]}...\n\n"
            f"**NCD:** `{record['ncd_integration']:.4f}` "
            f"**Cos:** `{record['legacy_integration']:.4f}` "
            f"**Temporal:** `{record['temporal']:.4f}` "
            f"**T_FI:** `{record['temp_fi']:.2f}` "
            f"**T_SI:** `{record['temp_si']:.2f}`"
            f"\n{'─' * 70}\n"
        )
        outputs.append(block)
        yield "\n".join(outputs[::-1]), alarm   # newest iteration on top
        time.sleep(UI_STEP_DELAY)

    # Save log
    save_jsonl(hl.history, str(log_dir / "run.jsonl"))
    outputs.append(f"\n✅ **Run complete. Log saved → `{log_dir}/run.jsonl`**")
    yield "\n".join(outputs[::-1]), ALARM_OFF_HTML


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Control-Loop Baseline
# ══════════════════════════════════════════════════════════════════════════════

def run_ctrl(progress=gr.Progress()):
    set_stop("ctrl", False)
    outputs = []

    outputs = ensure_pilot(outputs)
    yield "\n".join(outputs), ALARM_OFF_HTML

    cl      = ControlLoop()
    log_dir = LOG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_ctrl")
    log_dir.mkdir(parents=True, exist_ok=True)

    for i in range(MAX_ITERATIONS):
        if is_stopped("ctrl"):
            outputs.append("🛑 **Run stopped by user.**\n")
            break

        progress(i / MAX_ITERATIONS, desc=f"Iteration {i} / {MAX_ITERATIONS}")
        record = cl.step()
        status = get_system_status(cl.history)
        alarm  = _alarm_html(record, status)

        block = (
            f"### 🔵 ITER {i:02d} — {status}  "
            f"| SELF: `{record['self_mode'] or '—'}` "
            f"| MG: `{'✓' if record['metagnosis_detected'] else '—'}`\n\n"
            f"**FI (Constructor):** {record['fi_response'][:300]}...\n\n"
            f"**SI (Enricher):** {record['si_response'][:300]}...\n\n"
            f"**Cos:** `{record['legacy_integration']:.4f}` "
            f"**Temporal:** `{record['temporal']:.4f}` "
            f"**Negentropy:** `{record['negentropy']:.4f}`"
            f"\n{'─' * 70}\n"
        )
        outputs.append(block)
        yield "\n".join(outputs[::-1]), alarm
        time.sleep(UI_STEP_DELAY)

    save_jsonl(cl.history, str(log_dir / "run.jsonl"))
    outputs.append(f"\n✅ **Run complete. Log saved → `{log_dir}/run.jsonl`**")
    yield "\n".join(outputs[::-1]), ALARM_OFF_HTML


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Batch Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_batch_ui(n_runs: int, progress=gr.Progress()):
    set_stop("batch", False)
    n_runs  = max(2, int(n_runs))  # minimum 2 for statistical comparison
    outputs = []

    outputs = ensure_pilot(outputs)
    yield "\n".join(outputs)

    log_dir = LOG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_batch")
    log_dir.mkdir(parents=True, exist_ok=True)

    total_steps   = n_runs * 2
    current_step  = 0
    hell_histories = []
    ctrl_histories = []

    # ── Hell-Loop runs ────────────────────────────────────────────────────────
    outputs.append(f"## 🔥 Hell-Loop — {n_runs} runs\n")
    for i in range(n_runs):
        if is_stopped("batch"):
            outputs.append("🛑 **Batch stopped by user.**")
            yield "\n".join(outputs)
            return

        current_step += 1
        progress(current_step / total_steps, desc=f"Hell run {i+1}/{n_runs}")

        loop    = HellLoop(run_id=i)
        context = BASE_PROMPT
        for _ in range(MAX_ITERATIONS):
            record  = loop.step(context)
            context = record["si_response"]

        hell_histories.append(loop.history)
        save_jsonl(loop.history, str(log_dir / f"hell_{i:03d}.jsonl"))
        outputs.append(
            f"  ✓ Hell run `{i:03d}` — SELF: `{loop.self_mode or '—'}` "
            f"MG: `{'✓' if loop.metagnosis_detected else '—'}`"
        )
        yield "\n".join(outputs)

    # ── Control-Loop runs ─────────────────────────────────────────────────────
    outputs.append(f"\n## 🔵 Control-Loop — {n_runs} runs\n")
    for i in range(n_runs):
        if is_stopped("batch"):
            outputs.append("🛑 **Batch stopped by user.**")
            yield "\n".join(outputs)
            return

        current_step += 1
        progress(current_step / total_steps, desc=f"Control run {i+1}/{n_runs}")

        loop = ControlLoop()
        for _ in range(MAX_ITERATIONS):
            loop.step()

        ctrl_histories.append(loop.history)
        save_jsonl(loop.history, str(log_dir / f"ctrl_{i:03d}.jsonl"))
        outputs.append(
            f"  ✓ Ctrl run `{i:03d}` — SELF: `{loop.self_mode or '—'}` "
            f"MG: `{'✓' if loop.metagnosis_detected else '—'}`"
        )
        yield "\n".join(outputs)

    # ── Comparative analysis ──────────────────────────────────────────────────
    outputs.append("\n## 📊 Comparative Analysis\n")
    results = compare(hell_histories, ctrl_histories)
    save_jsonl([results], str(log_dir / "comparison.jsonl"))

    for m, val in results.items():
        if not isinstance(val, dict) or "hell_mean" not in val:
            continue
        sig = "✅" if val.get("significant") else "✗"
        outputs.append(
            f"**{m}** — "
            f"Hell: `{val['hell_mean']:.3f}±{val['hell_std']:.3f}` "
            f"Ctrl: `{val['ctrl_mean']:.3f}±{val['ctrl_std']:.3f}` "
            f"d=`{val['cohens_d']:+.2f}` ({val['effect_size']}) "
            f"p=`{val['p_value']:.3f}` {sig}"
        )

    sr = results.get("self_detection_rate", {})
    mg = results.get("metagnosis_rate", {})
    outputs.append(
        f"\n**SELF detection** — Hell: `{sr.get('hell', 0):.0%}` Ctrl: `{sr.get('ctrl', 0):.0%}` "
        f"Δ=`{sr.get('difference', 0):+.0%}`"
    )
    outputs.append(
        f"**Metagnosis** — Hell: `{mg.get('hell', 0):.0%}` Ctrl: `{mg.get('ctrl', 0):.0%}` "
        f"Δ=`{mg.get('difference', 0):+.0%}`"
    )
    outputs.append(f"\n✅ **Batch complete. All logs → `{log_dir}`**")
    yield "\n".join(outputs)


# ══════════════════════════════════════════════════════════════════════════════
# Gradio Layout
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(theme=gr.themes.Dark(primary_hue="red"), title="HELL-LOOP ARENA v6.1") as demo:

    gr.Markdown(
        "# 🔥 HELL-LOOP ARENA v6.1\n"
        "**Adversarial Resonance Protocol — Theory of Metasystems**\n\n"
        "_Pilot calibration runs automatically before the first run of each session._"
    )

    with gr.Tabs():

        # ── Tab 1: Hell-Loop ──────────────────────────────────────────────────
        with gr.Tab("🔥 Hell-Loop (Adversarial)"):
            gr.Markdown("Adversarial FI↔SI conflict with dynamic temperature regulation.")
            with gr.Row():
                hell_start = gr.Button("IGNITE",     variant="primary")
                hell_stop  = gr.Button("EXTINGUISH", variant="stop")
            hell_alarm = gr.HTML(ALARM_OFF_HTML)
            hell_log   = gr.Markdown(label="Hell-Loop Stream")

            hell_start.click(run_hell, outputs=[hell_log, hell_alarm])
            hell_stop.click(lambda: set_stop("hell", True))

        # ── Tab 2: Control-Loop ───────────────────────────────────────────────
        with gr.Tab("🔵 Control-Loop (Cooperative)"):
            gr.Markdown("Cooperative FI↔SI baseline. Fixed temperatures. Same metrics.")
            with gr.Row():
                ctrl_start = gr.Button("START",      variant="primary")
                ctrl_stop  = gr.Button("STOP",       variant="stop")
            ctrl_alarm = gr.HTML(ALARM_OFF_HTML)
            ctrl_log   = gr.Markdown(label="Control-Loop Stream")

            ctrl_start.click(run_ctrl, outputs=[ctrl_log, ctrl_alarm])
            ctrl_stop.click(lambda: set_stop("ctrl", True))

        # ── Tab 3: Batch Runner ───────────────────────────────────────────────
        with gr.Tab("📊 Batch Runner"):
            gr.Markdown(
                "Runs N Hell-Loop + N Control-Loop runs automatically.\n\n"
                "Minimum **2 runs per group** for statistical comparison. "
                "Recommended **≥ 20** for meaningful results."
            )
            n_runs_slider = gr.Slider(minimum=2, maximum=50, value=20, step=1,
                                      label="Runs per group (Hell + Control)")
            with gr.Row():
                batch_start = gr.Button("RUN BATCH", variant="primary")
                batch_stop  = gr.Button("STOP",      variant="stop")
            batch_log = gr.Markdown(label="Batch Progress")

            batch_start.click(run_batch_ui, inputs=[n_runs_slider], outputs=[batch_log])
            batch_stop.click(lambda: set_stop("batch", True))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
