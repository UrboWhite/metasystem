"""
hell_loop_ui.py — Hell-Loop Protocol

Gradio web interface with four tabs:
  Tab 1 — Hell-Loop Arena    : adversarial run in real time
  Tab 2 — Control-Loop Arena : cooperative run in real time
  Tab 3 — Batch Runner       : automated N-run batch with progress log
  Tab 4 — Metrics Reference  : logged fields and interpretation guide
"""

import threading
import time
import queue
import logging
import gradio as gr

from chaos_engine import (
    HellLoop, pilot, is_pilot_done,
    MODELS, MAX_ITERATIONS, BASE_PROMPT,
)
from chaos_engine_control import ControlLoopA, ControlLoopB
from batch_runner import run_batch as run_batch_engine, BATCH_MIN_RUNS

STEP_DELAY = 0.3

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

HTML_API_ERROR = """
<div style="background:#4a2000;padding:8px;border-radius:6px;
            text-align:center;margin-bottom:4px;">
  <b style="color:#ffa07a;font-size:1em;">
    ⚠️ API ERROR IN THIS ITERATION — DATA CORRUPTED ⚠️
  </b>
</div>
"""

_lock    = threading.Lock()
_flags   = {"hell": False, "control": False, "batch": False}
_running = {"hell": False, "control": False, "batch": False}

def _stop(key: str) -> None:
    with _lock:
        _flags[key] = True

def _clear(key: str) -> None:
    with _lock:
        _flags[key] = False

def _is_stopped(key: str) -> bool:
    with _lock:
        return _flags[key]

def _set_running(key: str, state: bool) -> None:
    with _lock:
        _running[key] = state

def _is_running(key: str) -> bool:
    with _lock:
        return _running[key]


class _QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]"):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record))
        except Exception:
            pass


def get_system_status(history: list[dict]) -> str:
    if not history:
        return "—"

    last = history[-1]
    mode = last.get("self_mode") or "NONE"
    meta = last.get("metagnosis_detected", False)
    bif = last.get("bifurcation_total", 0)
    self_score = last.get("self_score", 0)

    parts = [f"mode={mode}"]
    if meta:
        parts.append("META=✓")
    if bif:
        parts.append(f"bif={bif}")
    parts.append(f"self={self_score:.3f}" if self_score is not None else "self=n/a")

    return " | ".join(parts)


def _build_hell_context(history: list[dict]) -> str:
    valid = [h for h in history if h.get("valid", True) and not h.get("api_error")]
    if not valid:
        return BASE_PROMPT

    recent = valid[-3:]
    hist_lines = [f"FI: {h['fi_response']}\nSI: {h['si_response']}" for h in recent]
    return (
        "Context history:\n"
        + "\n".join(hist_lines)
        + f"\n\nLatest SI statement to deconstruct: {recent[-1]['si_response']}"
    )


def _build_control_context(history: list[dict]) -> str:
    if not history:
        return BASE_PROMPT

    last = history[-1]
    if last.get("si_response", "") not in ("", "[API_ERROR]"):
        return last["si_response"]

    for h in reversed(history[:-1]):
        if h.get("valid", True) and h.get("si_response", "") not in ("", "[API_ERROR]"):
            return h["si_response"]

    return BASE_PROMPT


def _pilot_if_needed(progress, desc: str):
    if not is_pilot_done():
        progress(0, desc="Running pilot calibration... 🔬")
        yield "⏳ Running pilot calibration (10 cooperative sequences)...", "", "", ""
        pilot(n_sequences=10)
        yield "✅ Pilot calibration complete.", "", "", ""


def _metrics_table(record: dict) -> str:
    pca       = record.get("pca_variance_ratio", [])
    pca_str   = f"{pca[0]:.4f}" if pca else "n/a"
    bif_total = record.get("bifurcation_total", "—")

    mg_div = record.get("mg_ncd_divergence")
    mg_div_str = f"{mg_div:.4f}" if mg_div is not None else "n/a"

    cv = record.get("critic_verdict")
    if cv is True:
        critic_str = "✓ STRUCTURAL"
    elif cv is False:
        critic_str = "✗ STYLISTIC"
    else:
        critic_str = "—"

    return (
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Cosine integration | `{record.get('cosine_integration', 0):.4f}` |\n"
        f"| NCD integration    | `{record.get('ncd_integration', 0):.4f}` |\n"
        f"| Temporal           | `{record.get('temporal', 0):.4f}` |\n"
        f"| SELF score         | `{record.get('self_score', 0):.4f}` |\n"
        f"| PCA PC1 variance   | `{pca_str}` |\n"
        f"| MG NCD divergence  | `{mg_div_str}` |\n"
        f"| T_FI / T_SI        | "
        f"`{record.get('temp_fi', 0):.2f}` / "
        f"`{record.get('temp_si', 0):.2f}` |\n"
        f"| SELF mode          | `{record.get('self_mode') or '—'}` |\n"
        f"| Metagnosis         | `{record.get('metagnosis_detected', False)}` |\n"
        f"| Bifurcations total | `{bif_total}` |\n"
        f"| Critic verdict     | `{critic_str}` |\n"
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


def run_hell(n_iter: int, progress=gr.Progress()):
    if _is_running("hell"):
        yield "⚠️ A Hell-Loop run is already in progress. Stop it first.", "", "", ""
        return

    _set_running("hell", True)
    _clear("hell")

    try:
        for out in _pilot_if_needed(progress, "Pilot calibration..."):
            yield out

        progress(0, desc="Igniting the metasystem... 🔥")

        loop    = HellLoop()
        outputs = []

        for i in range(1, n_iter + 1):
            if _is_stopped("hell"):
                break

            progress(i / n_iter, desc=f"Iteration {i}/{n_iter}")

            context = _build_hell_context(loop.history)
            record  = loop.step(context)

            status = get_system_status(loop.history)

            alarm_html   = HTML_ALARM_ON    if loop.self_mode is not None else HTML_ALARM_OFF
            meta_html    = HTML_METAGNOSIS  if record.get("metagnosis_detected") else ""
            bif_html     = HTML_BIFURCATION if record.get("bifurcation_fired")   else ""
            err_html     = HTML_API_ERROR   if record.get("api_error")           else ""
            event_html   = alarm_html + meta_html + bif_html + err_html

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

    finally:
        _set_running("hell", False)


def run_control(n_iter: int, variant: str, progress=gr.Progress()):
    if _is_running("control"):
        yield "⚠️ A Control-Loop run is already in progress. Stop it first.", "", "", ""
        return

    _set_running("control", True)
    _clear("control")

    try:
        for out in _pilot_if_needed(progress, "Pilot calibration..."):
            yield out

        label   = "Control-Loop A (no MG)" if variant == "A" else "Control-Loop B (MG observer)"
        LoopCls = ControlLoopA if variant == "A" else ControlLoopB

        progress(0, desc=f"Starting {label}... 🔵")

        loop    = LoopCls()
        outputs = []

        for i in range(1, n_iter + 1):
            if _is_stopped("control"):
                break

            progress(i / n_iter, desc=f"Iteration {i}/{n_iter}")

            context = _build_control_context(loop.history)
            record  = loop.step(context)

            status     = get_system_status(loop.history)
            meta_html  = HTML_METAGNOSIS if record.get("metagnosis_detected") else ""
            err_html   = HTML_API_ERROR  if record.get("api_error")           else ""
            event_html = meta_html + err_html

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

    finally:
        _set_running("control", False)


def _run_batch_engine_target(
    n_runs: int, n_iter: int, run_b: bool,
    interleaved: bool, reset_every: int | None, seed: int | None,
) -> None:
    try:
        run_batch_engine(
            n_runs=n_runs,
            iterations=n_iter,
            run_control_b=run_b,
            reset_between_batches=False,
            reset_every=reset_every if reset_every and reset_every > 0 else None,
            interleaved=interleaved,
            seed=seed,
        )
    except Exception as e:
        logging.getLogger("batch_runner").error(f"Batch failed: {e}")


def run_batch(
    n_runs: int, n_iter: int, run_b: bool,
    interleaved: bool, reset_every: int, seed: int,
    progress=gr.Progress(),
):
    if _is_running("batch"):
        yield "⚠️ A batch run is already in progress. Stop it first."
        return

    _set_running("batch", True)
    _clear("batch")

    log_lines: list[str] = []
    log_queue: "queue.Queue[str]" = queue.Queue()
    handler = _QueueLogHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(message)s"))

    watched_loggers = [
        logging.getLogger("batch_runner"),
        logging.getLogger("chaos_engine"),
        logging.getLogger("chaos_engine_control"),
    ]
    for lg in watched_loggers:
        lg.addHandler(handler)

    def _log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines[-200:])

    try:
        mode_label = "interleaved" if interleaved else "sequential"
        yield _log(
            f"Starting batch: {n_runs} runs × {n_iter} iterations "
            f"({mode_label} mode, Control-B {'enabled' if run_b else 'disabled'})"
        )
        progress(0, desc="Starting batch...")

        seed_val = seed if seed and seed != 0 else None
        reset_every_val = reset_every if reset_every and reset_every > 0 else None

        worker = threading.Thread(
            target=_run_batch_engine_target,
            args=(n_runs, n_iter, run_b, interleaved, reset_every_val, seed_val),
            daemon=True,
        )
        worker.start()

        while worker.is_alive() or not log_queue.empty():
            try:
                msg = log_queue.get(timeout=0.3)
                yield _log(msg)
            except queue.Empty:
                if _is_stopped("batch"):
                    yield _log("🛑 Stop requested — display halted. Batch continues in background.")
                    break

        yield _log(
            "\n✅ Batch finished (or stopped). See ./logs/ for run data.\n"
            "   Suggested analysis command:\n"
            "   python analyze_logs.py compare --hell logs/hellloop/ --control logs/control_a/"
        )

    finally:
        for lg in watched_loggers:
            lg.removeHandler(handler)
        _set_running("batch", False)


with gr.Blocks(
    theme=gr.themes.Dark(primary_hue="red"),
    title="HELL-LOOP ARENA",
) as demo:

    gr.Markdown(
        "# 🔥 HELL-LOOP ARENA\n"
        "**Adversarial Resonance — Metasystem Simulation**\n\n"
        f"Models — FI: `{MODELS['FI']}` | SI: `{MODELS['SI']}` | MG: `{MODELS['MG']}`\n\n"
        f"Canonical prompt: *\"{BASE_PROMPT}\"*"
    )

    with gr.Tabs():

        with gr.Tab("🔥 Hell-Loop Arena"):
            gr.Markdown(
                "Adversarial run — FI deconstructs, SI destabilizes, "
                "MG regulates with live system metrics. "
                "Embedding Critic fires when cosine enters upper zone."
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

        with gr.Tab("🔵 Control-Loop Arena"):
            gr.Markdown(
                "Cooperative baseline — FI builds, SI enriches.\n\n"
                "**Variant A** — no MG (pure baseline).  "
                "**Variant B** — MG as passive observer with structured metrics, no regulation."
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
                cl_start = gr.Button("🔵 START", variant="primary")
                cl_stop  = gr.Button("🛑 STOP",  variant="stop")

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

        with gr.Tab("📦 Batch Runner"):
            gr.Markdown(
                "Automated batch — runs N Hell-Loop and N Control-Loop A "
                "(optionally + Control-Loop B) via the same `batch_runner.py` "
                "engine used from the command line.\n\n"
                "**Interleaved mode (default, recommended)** alternates Hell-Loop "
                "and Control-Loop runs to prevent temporal hardware bias. "
                "**Sequential mode** runs all Hell-Loop runs first, then all "
                "Control-Loop runs.\n\n"
                "Pilot calibration runs once per session. All logs saved to `./logs/`."
            )
            with gr.Row():
                batch_runs_slider = gr.Slider(
                    minimum=5, maximum=100, value=BATCH_MIN_RUNS, step=5,
                    label="Runs per group", interactive=True,
                )
                batch_iter_slider = gr.Slider(
                    minimum=10, maximum=MAX_ITERATIONS, value=MAX_ITERATIONS,
                    step=5, label="Iterations per run", interactive=True,
                )
            with gr.Row():
                batch_run_b = gr.Checkbox(
                    value=False, label="Include Control-Loop B",
                    interactive=True,
                )
                batch_interleaved = gr.Checkbox(
                    value=True, label="Interleaved mode (recommended)",
                    interactive=True,
                )
            with gr.Row():
                batch_reset_every = gr.Number(
                    value=0, precision=0, minimum=0,
                    label="Recalibrate every N runs (0 = disabled)",
                    interactive=True,
                )
                batch_seed = gr.Number(
                    value=0, precision=0, minimum=0,
                    label="Random seed (0 = unset)",
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
                inputs=[
                    batch_runs_slider, batch_iter_slider, batch_run_b,
                    batch_interleaved, batch_reset_every, batch_seed,
                ],
                outputs=[batch_log],
            )
            batch_stop.click(fn=lambda: _stop("batch"))

        with gr.Tab("📊 Metrics Reference"):
            gr.Markdown("""
### Logged metrics per iteration

| Field | Description |
|-------|-------------|
| `cosine_integration` | Semantic cosine similarity FI↔SI (primary SELF signal) |
| `ncd_integration` | NCD similarity FI↔SI — bias-independent twin signal |
| `temporal` | Cross-iteration resonance: prev_FI↔curr_SI and prev_SI↔curr_FI |
| `self_score` | Continuous SELF score [0–1]: weighted z_int + z_tmp + PCA signal, calibrated against pilot statistics |
| `self_mode` | `SELF_EXTREME` / `SELF_STABLE` / null (reverts to null on SELF_DROP) |
| `api_error` | True if any API call in this iteration failed after retries |
| `negentropy_fi/si` | Word-level informational density (lexical diversity) of FI and SI outputs |
| `negentropy_mg` | Word-level informational density of MG diagnosis |
| `mg_ncd_divergence` | NCD distance between this MG diagnosis and the previous one — tracks whether MG's diagnoses stay novel or repeat |
| `pca_variance_ratio` | PC1–PC3 explained variance of cumulative embedding vectors |
| `metagnosis_detected` | Structural metagnosis signal (MG negentropy + NCD divergence trend) |
| `bifurcation_fired` | True if MG issued BIFURCATION TRIGGER this iteration |
| `bifurcation_total` | Cumulative bifurcation count for the run |
| `temp_fi / temp_si` | Active temperatures at this iteration |
| `critic_response` | Raw Embedding Critic response (when triggered) |
| `critic_verdict` | True = STRUCTURAL, False = STYLISTIC, None = not triggered |
| `vec_fi / vec_si` | Embedding vectors (384-dim, all-MiniLM-L6-v2) — saved to .npy file, not in JSONL |

### Post-run analysis

```bash
# Single run
python analyze_logs.py single --file logs/hellloop/hellloop_*.jsonl

# Batch folder
python analyze_logs.py batch --dir logs/hellloop/

# Full comparison (cosine + NCD + PCA + MG divergence + Bonferroni-corrected consensus score)
python analyze_logs.py compare \\
    --hell logs/hellloop/ \\
    --control logs/control_a/
    
    """)

if __name__ == "__main__":
    demo.queue().launch()

demo.queue().launch()