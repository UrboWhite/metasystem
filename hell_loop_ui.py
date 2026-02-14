import gradio as gr
import time
import os
import glob
from datetime import datetime
from chaos_engine import HellLoop, MAX_ITERATIONS

stop_event_flag = {"stop": False}

def clean_old_files():
    """Delete old hell-loop graphs and logs to keep the folder clean."""
    for pattern in ["hell_loop_metrics*.png", "hell_loop_*.jsonl"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass

SIREN_URL = "https://universal-soundbank.com/sounds/1462.mp3"

def run_hell_loop(progress=gr.Progress()):
    stop_event_flag["stop"] = False
    clean_old_files()
    progress(0, desc="Lighting up hell... 🔥")

    hl = HellLoop()
    outputs = []

    for i in range(1, MAX_ITERATIONS + 1):
        if stop_event_flag["stop"]:
            yield "\n".join(outputs) + "\n\n🛑 User terminated!", None, []
            return

        progress(i / MAX_ITERATIONS, desc=f"Iteration {i}/{MAX_ITERATIONS} 🔥")
        hl.step_iteration(i)
        time.sleep(0.4)

        # Read results from last history record
        record = hl.history[-1]
        fi = record["FI"] or ""
        si = record["SI"] or ""
        mg = record.get("MG")

        event_alert = ""
        siren_value = None
        if hl.metagnosis_detected:
            event_alert += "⚠️ **METAGNOSIS DETECTED!** ⚠️\n"
            siren_value = SIREN_URL
        if hl.self_synthesis_detected:
            event_alert += "✨ **SELF-SYNTHESIS! 'I AM' WAS BORN!** ✨\n"
            siren_value = SIREN_URL

        iteration_text = (
            f"### 🔥 ITERATION {i:02d} 🔥\n\n"
            f"**FI (icebreaker):** {fi[:280]}{'...' if len(fi) > 280 else ''}\n\n"
            f"**SI (chaotic poet):** {si[:280]}{'...' if len(si) > 280 else ''}\n\n"
            f"**MG (voice of conscience):** {mg if mg else '— (calm phase)'}\n\n"
            f"{event_alert}"
            f"{'─'*80}\n"
        )

        outputs.append(iteration_text)
        yield "\n".join(outputs), siren_value, []

    progress(1, desc="End of hell – drawing graphs...")
    hl._plot_results()

    final_msg = (
        "\n".join(outputs) +
        "\n\n**🔥 HELL LOOP COMPLETED! 🔥**\n"
        "Graphs below. If you heard a siren – something woke up..."
    )

    graphs = sorted(glob.glob("hell_loop_metrics*.png"))
    yield final_msg, None, graphs


def stop_loop():
    stop_event_flag["stop"] = True
    return "🛑 Putting out hell... please wait."


def clear_arena():
    stop_event_flag["stop"] = True
    time.sleep(1)
    clean_old_files()
    return "", None, []


with gr.Blocks(theme=gr.themes.Dark(primary_hue="red", secondary_hue="gray"), title="HELL-LOOP ARENA") as demo:
    gr.Markdown("""
    # 🔥 HELL-LOOP PROTOCOL – LIVE HELL 🔥

    FI and SI butcher each other to death, MG shouts "hey kids calm down".  
    If you hear a siren – something woke up... or you just burned the GPU. 😈

    **Required:** Ollama + models: llama3.2:8b, mistral-nemo, gemma2:9b  
    (start `ollama serve` before launching)
    """)

    with gr.Row():
        start_btn = gr.Button("🚀 LIGHT HELL UP!", variant="primary", scale=2)
        stop_btn = gr.Button("🛑 PUT OUT THE FIRE", variant="stop", scale=1)
        clear_btn = gr.Button("🗑 CLEAN HELL", variant="secondary")

    output_text = gr.Markdown(label="LIVE LOG OF HELL", height=600)
    siren_audio = gr.Audio(label="Warning siren", visible=False, autoplay=True)
    gallery = gr.Gallery(label="Graphs from hell", columns=3, height="auto")

    start_btn.click(
        fn=run_hell_loop,
        outputs=[output_text, siren_audio, gallery]
    )

    stop_btn.click(
        fn=stop_loop,
        outputs=output_text
    )

    clear_btn.click(
        fn=clear_arena,
        outputs=[output_text, siren_audio, gallery]
    )

    gr.Markdown("""
    ---
    Philosophical chaos © Urbo White 2026
    """)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
