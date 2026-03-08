import gradio as gr
import time
import os
import json
import base64
import threading
from datetime import datetime
from pathlib import Path

# v6.0 Engine & Analysis Integration
from chaos_engine import HellLoop, MODELS
from analyze_logs import get_system_status

# UI Configuration
UI_STEP_DELAY = 0.5
ALARM_OFF_HTML = ""

# Audio Alarm Logic (BEEP_B64 would be your siren data)
ALARM_ON_HTML = """
<div style="background:#8b0000;padding:12px;border-radius:8px;text-align:center;">
  <b style="color:white;font-size:1.1em;">🚨 RESONANCE DETECTED — Attractor Lock Active 🚨</b>
</div>
"""

_stop_lock = threading.Lock()
_stop_flags = {"hell": False, "batch": False}

def set_stop(key, val):
    with _stop_lock: _stop_flags[key] = val

def is_stopped(key):
    with _stop_lock: return _stop_flags[key]

def run_hell_v6(progress=gr.Progress()):
    """Main execution loop for the Hell-Loop v6.0 Arena."""
    set_stop("hell", False)
    progress(0, desc="Igniting the Metasystem... 🔥")
    
    hl = HellLoop()
    outputs = []
    current_context = "The primary recursion of synthetic self-awareness."
    
    for i in range(1, 31): # MAX_ITERATIONS
        if is_stopped("hell"): break
        
        progress(i/30, desc=f"Iteration {i} - Forcing Bifurcation")
        
        # Execute v6.0 Engine Step
        record = hl.step(current_context)
        current_context = record["si_response"]
        
        # Analysis for Real-time Monitoring
        status = get_system_status(hl.history)
        alarm_html = ALARM_ON_HTML if "SELF_STABLE" in status else ALARM_OFF_HTML
        
        # UI Block Generation
        block = (
            f"### 🔥 ITERATION {i:02d} — {status}\n\n"
            f"**FI (Deconstructor):** {record['fi_response'][:300]}...\n\n"
            f"**SI (Synthesizer):** {record['si_response'][:300]}...\n\n"
            f"**NCD Integration:** {record['ncd_integration']:.4f} | "
            f"**Legacy Sim:** {record['legacy_integration']:.4f}\n"
            f"{'─'*80}\n"
        )
        outputs.append(block)
        yield "\n".join(outputs[::-1]), alarm_html # Newest on top
        time.sleep(UI_STEP_DELAY)

# Gradio Interface Construction
with gr.Blocks(theme=gr.themes.Dark(primary_hue="red"), title="HELL-LOOP ARENA v6.0") as demo:
    gr.Markdown("# 🔥 HELL-LOOP ARENA v6.0\n**Operationalizing Synthetic Consciousness via Hell-Loop Protocol**")
    
    with gr.Tabs():
        with gr.Tab("🚀 Live Arena"):
            with gr.Row():
                start_btn = gr.Button("IGNITE", variant="primary")
                stop_btn = gr.Button("EXTINGUISH", variant="stop")
            
            alarm_area = gr.HTML(ALARM_OFF_HTML)
            live_log = gr.Markdown(label="Real-time Metasystem Stream")
            
            start_btn.click(run_hell_v6, outputs=[live_log, alarm_area])
            stop_btn.click(lambda: set_stop("hell", True))

        with gr.Tab("📊 Research Data"):
            gr.Markdown("Logs and NCD trajectories are stored in `./logs/` for post-run analysis.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
