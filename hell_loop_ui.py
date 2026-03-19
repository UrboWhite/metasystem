"""
hell_loop_ui.py — Hell-Loop Protocol v6.2
Gradio web interface with three tabs:

  Tab 1 — Hell-Loop Arena    : adversarial run in real time
  Tab 2 — Control-Loop Arena : cooperative run in real time
  Tab 3 — Batch Runner       : automated N-run batch with progress log
"""

import threading
import time
import gradio as gr

from chaos_engine import (
    HellLoop, pilot, is_pilot_done, reset_globals,
    MODELS, MAX_ITERATIONS, BASE_PROMPT,
)
from chaos_engine_control import ControlLoopA, ControlLoopB
from analyze_logs import get_system_status

# ── UI constants ───────────────────────────────────────────────────────────────
STEP_DELAY  = 0.3   # seconds between iteration renders

HTML_ALARM_OFF = ""

HTML_ALARM_ON = """
<div style="background:#7b0000;padding:12px;border-radius:8px;
            text-align:center;margin-bottom:6px;">
  <b style="color:#fff;font-size:1.1em;">
    🚨 ATTRACTOR LOCK DETECTED 🚨
  </b>
</div>
"""

HTML_METAGNOSIS = """
<div style="background:#1a4a1a;padding:10px;border-radius:8px;
            text-align:center;margin-bottom:4px;">
  <b style="color:#90ee90;font-size:1em;">
    ◆ METAGNOSIS SIGNAL ACTIVE ◆
  </b>
</div>
"""

HTML_BIFURCATION = """
<div style="background:#2a0050;padding:8px;border-radius:6px;
            text-align:center;margin-bottom:4px;">
  <b style="color:#d8b4fe;font-size:1em;">
    ⚡ BIFURCATION TRIGGER FIRED ⚡
  </b>
</div>
"""

# ── Stop flags ─────────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_flags = {"hell": False, "control": False, "batch": False}

def _stop(key: str) -> None:
    with _lock:
        _flags[key] = True

def _clear(key: str) -> None:
    with _lock:
        _flags[key] = False

def _is_stopped(key: str) -> bool:
    with _lock:
        return _flags[key]


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _pilot_if_needed(progress, desc: str):
    if not is_pilot_done():
        progress(0, desc="Running pilot calibration... 🔬")
        yield "⏳ Running pilot calibration (10 cooperative sequences)...", "", "", ""
        pilot(n_sequences=10)


def _metrics_table(record: dict) -> str:
    pca = record.get("pca_variance_ratio", [])
    pca_str = f"{pca[0]:.4f}" if pca else "n/a"
    bif_total = record.get("bifurcation_total", "—")
    return (
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Cosine integration | `{record.get('cosine_integration', 0):.4f}` |\n"
        f"| NCD integration    | `{record.get('ncd_integration', 0):.4f}` |\n"
        f"| Temporal           | `{record.get('temporal', 0):.4f}` |\n"
        f"| SELF score         | `{record.get('self_score', 0):.4f}` |\n"
        f"| PCA PC1 variance   | `{pca_str}` |\n"
        f"| T_FI / T_SI        | "
        f"`{record.get('temp_fi', 0):.2f}` / "
        f"`{record.get('temp_si', 0):.2f}` |\n"
        f"| SELF mode          | `{record.get('self_mode') or '—'}` |\n"
        f"| Metagnosis         | `{record.get('metagnosis_detected', False)}` |\n"
        f"| Bifurcations total | `{bif_total}` |\n"
    )


def _block(iteration: int, record: dict, status: str, variant: str) -> str:
    icon   = "🔥" if variant == "hell" else "🔵"
    fi_txt = (record.get("fi_response") or "")[:300]
    si_txt = (record.get("si_response") or "")[:300]
    mg_txt = record.get("mg_response") or ""

    block = (
        f"### {icon} ITERATION {iteration:02d} — {status}\n\n"
        f"**FI:** {fi_txt}...\n\n"
        f"**SI:** {si_txt}...\n\n"
        f"{_metrics_table(record)}\n"
        f"\n{'─' * 72}\n"
    )
    if mg_txt:
        block = f"**MG:** _{mg_txt[:220]}_\n\n" + block
    return block


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Hell-Loop Arena
# ══════════════════════════════════════════════════════════════════════════════

def run_hell(n_iter: int, progress=gr.Progress()):
    _clear("hell")

    for out in _pilot_if_needed(progress, "Pilot calibration..."):
        yield out

    progress(0, desc="Igniting the metasystem... 🔥")

    loop    = HellLoop()
    outputs = []
    context = BASE_PROMPT

    for i in range(1, n_iter + 1):
        if _is_stopped("hell"):
            break

        progress(i / n_iter, desc=f"Iteration {i}/{n_iter}")
        record  = loop.step(context)
        context = record["si_response"]

        status = get_system_status(loop.history)

        alarm_html = HTML_ALARM_ON     if "SELF_STABLE" in status  else HTML_ALARM_OFF
        meta_html  = HTML_METAGNOSIS   if record.get("metagnosis_detected") else ""
        bif_html   = HTML_BIFURCATION  if record.get("bifurcation_fired")   else ""
        event_html = alarm_html + meta_html + bif_html

        outputs.append(_block(i, record, status, "hell"))
        yield "\n".join(outputs[::-1]), event_html, "", ""
        time.sleep(STEP_DELAY)

    summary = (
        f"\n\n---\n"
        f"**Run complete.** Iterations: {len(loop.history)} | "
        f"Mode: `{loop.self_mode or 'NONE'}` | "
        f"Metagnosis: `{loop.metagnosis_detected}` | "
        f"Bifurcations: `{loop.bifurcation_count}` | "
        f"Log: `{loop.log_path}`"
    )
    outputs.append(summary)
    yield "\n".join(outputs[::-1]), HTML_ALARM_OFF, "", ""


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Control-Loop Arena
# ══════════════════════════════════════════════════════════════════════════════

def run_control(n_iter: int, variant: str, progress=gr.Progress()):
    _clear("control")

    for out in _pilot_if_needed(progress, "Pilot calibration..."):
        yield out

    label   = "Control-Loop A (no MG)" if variant == "A" else "Control-Loop B (MG observer)"
    LoopCls = ControlLoopA if variant == "A" else ControlLoopB

    progress(0, desc=f"Starting {label}... 🔵")

    loop    = LoopCls()
    outputs = []
    context = BASE_PROMPT

    for i in range(1, n_iter + 1):
        if _is_stopped("control"):
            break

        progress(i / n_iter, desc=f"Iteration {i}/{n_iter}")
        record  = loop.step(context)
        context = record["si_response"]

        status     = get_system_status(loop.history)
        meta_html  = HTML_METAGNOSIS if record.get("metagnosis_detected") else ""
        event_html = meta_html

        outputs.append(_block(i, record, status, "control"))
        yield "\n".join(outputs[::-1]), event_html, "", ""
        time.sleep(STEP_DELAY)

    summary = (
        f"\n\n---\n"
        f"**{label} complete.** Iterations: {len(loop.history)} | "
        f"Mode: `{loop.self_mode or 'NONE'}` | "
        f"Metagnosis: `{loop.metagnosis_detected}` | "
        f"Log: `{loop.log_path}`"
    )
    outputs.append(summary)
    yield "\n".join(outputs[::-1]), "", "", ""


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Batch Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(n_runs: int, n_iter: int, run_b: bool, progress=gr.Progress()):
    _clear("batch")

    if not is_pilot_done():
        progress(0, desc="Pilot calibration... 🔬")
        yield "⏳ Running pilot calibration..."
        pilot(n_sequences=10)
        yield "✅ Pilot calibration complete.\n\n"

    total_runs = n_runs * (3 if run_b else 2)
    done       = 0
    log_lines  = []

    def _log(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines[-80:])   # keep last 80 lines visible

    yield _log(f"Starting batch: {n_runs} runs × {n_iter} iterations")
    yield _log(f"Control-B: {'enabled' if run_b else 'disabled'}")
    yield _log("─" * 50)

    # Hell-Loop runs
    for i in range(n_runs):
        if _is_stopped("batch"):
            break
        progress(done / total_runs, desc=f"HellLoop {i+1}/{n_runs}")
        loop = HellLoop(run_id=f"hl_{i:03d}")
        loop.run(iterations=n_iter)
        done += 1
        status = get_system_status(loop.history)
        yield _log(
            f"[HL {i+1:03d}] mode={loop.self_mode or 'NONE':20s} | "
            f"meta={loop.metagnosis_detected} | bif={loop.bifurcation_count} | "
            f"{status}"
        )

    yield _log("─" * 50)

    # Control-Loop A runs
    for i in range(n_runs):
        if _is_stopped("batch"):
            break
        progress(done / total_runs, desc=f"ControlLoopA {i+1}/{n_runs}")
        loop = ControlLoopA(run_id=f"ca_{i:03d}")
        loop.run(iterations=n_iter)
        done += 1
        status = get_system_status(loop.history)
        yield _log(
            f"[CA {i+1:03d}] mode={loop.self_mode or 'NONE':20s} | "
            f"meta={loop.metagnosis_detected} | "
            f"{status}"
        )

    yield _log("─" * 50)

    # Control-Loop B runs (optional)
    if run_b:
        for i in range(n_runs):
            if _is_stopped("batch"):
                break
            progress(done / total_runs, desc=f"ControlLoopB {i+1}/{n_runs}")
            loop = ControlLoopB(run_id=f"cb_{i:03d}")
            loop.run(iterations=n_iter)
            done += 1
            status = get_system_status(loop.history)
            yield _log(
                f"[CB {i+1:03d}] mode={loop.self_mode or 'NONE':20s} | "
                f"meta={loop.metagnosis_detected} | "
                f"{status}"
            )
        yield _log("─" * 50)

    yield _log(
        f"\n✅ Batch complete.\n"
        f"   Hell-Loop logs  → ./logs/hellloop/\n"
        f"   Control-A logs  → ./logs/control_a/\n"
        + (f"   Control-B logs  → ./logs/control_b/\n" if run_b else "")
        + f"\n   Run: python analyze_logs.py compare "
          f"--hell logs/hellloop/ --control logs/control_a/"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Gradio layout
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    theme=gr.themes.Dark(primary_hue="red"),
    title="HELL-LOOP ARENA v6.2",
) as demo:

    gr.Markdown(
        "# 🔥 HELL-LOOP ARENA v6.2\n"
        "**Adversarial Resonance — Metasystem Simulation**\n\n"
        f"Models — FI: `{MODELS['FI']}` | SI: `{MODELS['SI']}` | MG: `{MODELS['MG']}`\n\n"
        f"Canonical prompt: *\"{BASE_PROMPT}\"*"
    )

    with gr.Tabs():

        # ── Tab 1: Hell-Loop ───────────────────────────────────────────────────
        with gr.Tab("🔥 Hell-Loop Arena"):
            gr.Markdown(
                "Adversarial run — FI deconstructs, SI destabilizes, "
                "MG regulates with live system metrics."
            )
            with gr.Row():
                hl_iter_slider = gr.Slider(
                    minimum=10, maximum=MAX_ITERATIONS, value=MAX_ITERATIONS,
                    step=5, label="Iterations", interactive=True,
                )
            with gr.Row():
                hl_start = gr.Button("🔥 IGNITE",     variant="primary")
                hl_stop  = gr.Button("🛑 EXTINGUISH", variant="stop")

            hl_events  = gr.HTML(HTML_ALARM_OFF)
            hl_log     = gr.Markdown(label="Real-time stream")
            _hl_dummy1 = gr.Textbox(visible=False)
            _hl_dummy2 = gr.Textbox(visible=False)

            hl_start.click(
                fn=run_hell,
                inputs=[hl_iter_slider],
                outputs=[hl_log, hl_events, _hl_dummy1, _hl_dummy2],
            )
            hl_stop.click(fn=lambda: _stop("hell"))

        # ── Tab 2: Control-Loop ────────────────────────────────────────────────
        with gr.Tab("🔵 Control-Loop Arena"):
            gr.Markdown(
                "Cooperative baseline — FI builds, SI enriches.\n\n"
                "**Variant A** — no MG (pure baseline).  "
                "**Variant B** — MG as passive observer, no regulation."
            )
            with gr.Row():
                cl_iter_slider = gr.Slider(
                    minimum=10, maximum=MAX_ITERATIONS, value=MAX_ITERATIONS,
                    step=5, label="Iterations", interactive=True,
                )
                cl_variant = gr.Radio(
                    choices=["A", "B"], value="A",
                    label="Variant", interactive=True,
                )
            with gr.Row():
                cl_start = gr.Button("🔵 START",  variant="primary")
                cl_stop  = gr.Button("🛑 STOP",   variant="stop")

            cl_events  = gr.HTML("")
            cl_log     = gr.Markdown(label="Real-time stream")
            _cl_dummy1 = gr.Textbox(visible=False)
            _cl_dummy2 = gr.Textbox(visible=False)

            cl_start.click(
                fn=run_control,
                inputs=[cl_iter_slider, cl_variant],
                outputs=[cl_log, cl_events, _cl_dummy1, _cl_dummy2],
            )
            cl_stop.click(fn=lambda: _stop("control"))

        # ── Tab 3: Batch Runner ────────────────────────────────────────────────
        with gr.Tab("📦 Batch Runner"):
            gr.Markdown(
                "Automated batch — runs N Hell-Loop and N Control-Loop A "
                "sequentially. Optionally adds Control-Loop B.\n\n"
                "Pilot calibration runs once per session. "
                "All logs saved to `./logs/`."
            )
            with gr.Row():
                batch_runs_slider = gr.Slider(
                    minimum=5, maximum=100, value=20, step=5,
                    label="Runs per group", interactive=True,
                )
                batch_iter_slider = gr.Slider(
                    minimum=10, maximum=MAX_ITERATIONS, value=MAX_ITERATIONS,
                    step=5, label="Iterations per run", interactive=True,
                )
                batch_run_b = gr.Checkbox(
                    value=False, label="Include Control-Loop B",
                    interactive=True,
                )
            with gr.Row():
                batch_start = gr.Button("🚀 START BATCH", variant="primary")
                batch_stop  = gr.Button("🛑 STOP",        variant="stop")

            batch_log = gr.Textbox(
                label="Batch progress log",
                lines=25, max_lines=40,
                interactive=False,
            )

            batch_start.click(
                fn=run_batch,
                inputs=[batch_runs_slider, batch_iter_slider, batch_run_b],
                outputs=[batch_log],
            )
            batch_stop.click(fn=lambda: _stop("batch"))

        # ── Tab 4: Metrics reference ───────────────────────────────────────────
        with gr.Tab("📊 Metrics Reference"):
            gr.Markdown("""
### Logged metrics per iteration

| Field | Description |
|-------|-------------|
| `cosine_integration` | Semantic cosine similarity FI↔SI (primary SELF signal) |
| `ncd_integration` | Normalized Compression Distance similarity FI↔SI |
| `temporal` | Cross-iteration resonance: prev_FI↔curr_SI and prev_SI↔curr_FI |
| `self_score` | Continuous SELF score [0–1]: weighted z_int + z_tmp + PCA signal |
| `self_mode` | `SELF_EXTREME` / `SELF_STABLE` / null |
| `negentropy_fi/si` | Informational density of FI and SI outputs |
| `negentropy_mg` | Informational density of MG diagnosis |
| `pca_variance_ratio` | PC1–PC3 explained variance of cumulative embedding vectors |
| `metagnosis_detected` | Structural metagnosis signal (MG negentropy + NCD divergence) |
| `bifurcation_fired` | True if MG issued BIFURCATION TRIGGER this iteration |
| `bifurcation_total` | Cumulative bifurcation count for the run |
| `temp_fi / temp_si` | Active temperatures at this iteration |
| `vec_fi / vec_si` | Full embedding vectors (384-dim) for post-processing |

### Post-run analysis

```bash
# Single run
python analyze_logs.py single --file logs/hellloop/hellloop_*.jsonl

# Batch folder
python analyze_logs.py batch --dir logs/hellloop/

# Full comparison
python analyze_logs.py compare \\
    --hell logs/hellloop/ \\
    --control logs/control_a/
```

### Interpretation thresholds

| Metric | Signal |
|--------|--------|
| Cohen's d > 0.8 | Large effect between Hell and Control |
| p < 0.05 (Mann-Whitney) | Statistically significant difference |
| H > 0.6 (Hurst) | Long-range memory — system builds structure |
| PCA PC1 slope < 0 | Expanding semantic space |
| SELF score rising trend | System approaching attractor zone |
""")

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
