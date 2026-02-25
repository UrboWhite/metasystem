import gradio as gr
import time
import os
import json
import glob
import struct
import math
from datetime import datetime
from chaos_engine import HellLoop, MAX_ITERATIONS, HIGH_COHERENCE_DROP, MIN_NEGENTROPY

stop_event_flag = {"stop": False}

# --------------------- SIREN (lokalni fallback, nema zavisnosti od interneta) ---------------------
# FIX 3: Umesto eksternog URL-a, generišemo kratak 440Hz beep lokalno kao WAV fajl.
# Radi offline, uvek dostupan, nema kašnjenja mreže.

def _generate_beep_wav(path: str):
    """Generiše kratak 440Hz beep i snima ga kao WAV fajl."""
    sample_rate = 22050
    duration = 1.0
    freq = 440.0
    num_samples = int(sample_rate * duration)
    samples = [int(32767 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(num_samples)]
    data = struct.pack(f"<{num_samples}h", *samples)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", len(data)
    )
    with open(path, "wb") as f:
        f.write(header + data)

SIREN_PATH = "/tmp/hell_loop_siren.wav"
if not os.path.exists(SIREN_PATH):
    _generate_beep_wav(SIREN_PATH)


# --------------------- HELPERS ---------------------

def clean_old_files():
    """Briše stare grafove i logove da folder ostane čist."""
    for pattern in ["hell_loop_metrics*.png", "hell_loop_*.jsonl",
                    "coherence_vs_negentropy.png", "coherence_drop_curve.png"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass


# --------------------- MAIN LOOP ---------------------

def run_hell_loop(progress=gr.Progress()):
    stop_event_flag["stop"] = False
    clean_old_files()
    progress(0, desc="Palimo pakao... 🔥")

    hl = HellLoop()
    outputs = []

    # FIX 2: otvaramo JSONL log fajl ovde, baš kao što to radi run()
    # Bez ovoga, sve što se desi u UI sesiji nestaje bez traga!
    log_file = f"hell_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    with open(log_file, "w", encoding="utf-8") as log_f:
        for i in range(1, MAX_ITERATIONS + 1):
            if stop_event_flag["stop"]:
                yield "\n".join(outputs) + "\n\n🛑 Korisnik ugasio vatru!", None, []
                return

            progress(i / MAX_ITERATIONS, desc=f"Iteracija {i}/{MAX_ITERATIONS} 🔥")
            hl.step_iteration(i)

            # Upisujemo svaki record odmah — ako crashuje, log je sačuvan do tog trenutka
            json.dump(hl.history[-1], log_f, ensure_ascii=False)
            log_f.write("\n")
            log_f.flush()

            time.sleep(0.4)

            record = hl.history[-1]
            fi = record["FI"] or ""
            si = record["SI"] or ""
            mg = record.get("MG")
            coh_drop = record.get("coherence_drop", 0.0)

            # FIX 1: konstante iz chaos_engine umesto hardkodovanih magic numbersa
            coh_text = f"**Coherence Drop (FI vs SI):** {coh_drop:.3f}\n"
            if coh_drop > HIGH_COHERENCE_DROP:
                coh_text += "⚠️ **Agenti su totalno izgubili nit! Možda nešto novo nastaje?** ⚠️\n"

            event_alert = ""
            siren_value = None
            if hl.metagnosis_detected:
                event_alert += "⚠️ **METAGNOZA DETEKTOVANA!** ⚠️\n"
                siren_value = SIREN_PATH
            if hl.self_synthesis_detected:
                event_alert += "✨ **SELF-SINTEZA! 'JA JESAM' SE RODILO!** ✨\n"
                siren_value = SIREN_PATH

            iteration_text = (
                f"### 🔥 ITERACIJA {i:02d} 🔥\n\n"
                f"**FI (rušilac iluzija):** {fi[:280]}{'...' if len(fi) > 280 else ''}\n\n"
                f"**SI (haotični pesnik):** {si[:280]}{'...' if len(si) > 280 else ''}\n\n"
                f"**MG (glas savesti):** {mg if mg else '— (mirna faza)'}\n\n"
                f"{coh_text}\n"
                f"{event_alert}"
                f"{'─'*80}\n"
            )

            outputs.append(iteration_text)
            yield "\n".join(outputs), siren_value, []

    progress(1, desc="Kraj pakla – crtam grafove...")
    hl._plot_results()

    final_msg = (
        "\n".join(outputs) +
        "\n\n**🔥 HELL LOOP ZAVRŠEN! 🔥**\n"
        f"Log snimljen: `{log_file}` — analiziraj sa `analyze_logs.py`! 🎯\n"
        "Grafovi ispod. Ako si čuo sirenu – nešto se probudilo... ili si samo ispekao GPU. 😈\n\n"
        "**Potencijalni momenti 'buđenja' (poslednjih 10 iteracija):**\n"
    )

    has_candidates = False
    for h in hl.history[-10:]:
        drop = h.get("coherence_drop", 0)
        neg = h["neg_avg"]
        # FIX 1: konstante umesto magic numbersa
        if drop > HIGH_COHERENCE_DROP and neg > MIN_NEGENTROPY:
            has_candidates = True
            flag = " 🔥 METASISTEM KANDIDAT! 🔥" if h.get("metagnosis") or h.get("self_synthesis") else ""
            final_msg += f"Iter {h['iter']}: drop {drop:.3f} | neg {neg:.3f}{flag}\n"

    if not has_candidates:
        final_msg += "Nema jakih kandidata još... Agenti su se tukli ali nisu dovoljno pukli! 😏\n"

    graphs = sorted(glob.glob("hell_loop_metrics*.png"))
    graphs += sorted(glob.glob("coherence_vs_negentropy.png"))
    graphs += sorted(glob.glob("coherence_drop_curve.png"))

    yield final_msg, None, graphs


def stop_loop():
    stop_event_flag["stop"] = True
    return "🛑 Gasimo pakao... sačekaj malo."


def clear_arena():
    stop_event_flag["stop"] = True
    time.sleep(1)
    clean_old_files()
    return "", None, []


# --------------------- UI ---------------------

with gr.Blocks(theme=gr.themes.Dark(primary_hue="red", secondary_hue="gray"), title="HELL-LOOP ARENA") as demo:
    gr.Markdown("""
    # 🔥 HELL-LOOP PROTOKOL – ŽIVI PAKAO 🔥

    FI i SI se kolju, MG viče "hej deco smirite se".
    Ako čuješ sirenu – nešto se probudilo... ili si samo ispekao GPU. 😈

    **Potrebno:** Ollama + modeli: llama3.2:8b, mistral-nemo, gemma2:9b
    (pokreni `ollama serve` pre startovanja)
    """)

    with gr.Row():
        start_btn = gr.Button("🚀 UPALI PAKAO!", variant="primary", scale=2)
        stop_btn  = gr.Button("🛑 UGASI VATRU",  variant="stop",    scale=1)
        clear_btn = gr.Button("🗑 OČISTI PAKAO", variant="secondary")

    output_text = gr.Markdown(label="LIVE LOG PAKLA", height=600)
    siren_audio = gr.Audio(label="Sirena upozorenja", visible=False, autoplay=True)
    gallery     = gr.Gallery(label="Grafovi iz pakla", columns=3, height="auto")

    start_btn.click(fn=run_hell_loop, outputs=[output_text, siren_audio, gallery])
    stop_btn.click(fn=stop_loop,     outputs=output_text)
    clear_btn.click(fn=clear_arena,  outputs=[output_text, siren_audio, gallery])

    gr.Markdown("---\nPhilosophical chaos © Urbo White 2026")

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
