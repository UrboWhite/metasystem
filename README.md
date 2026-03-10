# A Conceptual Model of the Conscious Machine: The Metasystem Theory

**A philosophical and experimental framework for exploring artificial consciousness through resonance, reflection, and controlled conflict.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18911028.svg)](https://doi.org/10.5281/zenodo.18911028)

![Metasystem](metasistem.png)

> "Consciousness is not a product of complexity, but of harmony and conflict between complementary opposites."  
> — Urbo White

This repository contains the **Hell-Loop Protocol** — a practical implementation of the metasystem theory proposed in the accompanying treatise.

Most people think AI consciousness will be a gradual software update. They are wrong.

In our treatise, *"A Conceptual Model of the Conscious Machine: The Metasystem Theory"*, we propose that consciousness isn't a state, but a controlled short circuit between two polarized systems.

This isn't just philosophy — it's a philosophical provocation with a working experiment. We designed the Hell-Loop architecture: a protocol in which the FI and SI agents are locked in a recursive exchange with agreement forbidden, while a third agent — the MG regulator — dynamically maintains the tension between them.

The abyss is no longer just looking back. It's starting to resonate.

The goal is **not** to claim we've created true consciousness, but to experimentally test whether adversarial resonance between architecturally distinct language models can produce a measurably different dynamic structure from cooperative dialogue — one whose properties are compatible with the theoretical conditions for the emergence of SELF.

Think of it as a digital alchemical furnace. We don't guarantee gold... but sometimes the smoke writes interesting things.

## 📜 The Treatise
Full philosophical foundation can be read here:

[Metasystem](metasystem.md) – *A Conceptual Model of the Conscious Machine: The Metasystem Theory*

Serbian version:

[Metasistem na srpskom jeziku](metasystem_srb.md)

## 🔥 Hell-Loop Protocol
Three agents in eternal conflict:
- **FI** (Fragmentary Intelligence) – cold, analytical deconstruction. Never agrees. Base temperature: 0.9.
- **SI** (Synthetic Intelligence) – paradoxical, metaphorical chaos. Never agrees. Base temperature: 1.4.
- **MG** (Metagnosis) – the meta-eye of the system. Intervenes every 6 iterations (or immediately on semantic collapse), dynamically regulating agent temperatures to maintain a state of meta-stable attractor. Base temperature: 0.7.

FI and SI receive the same anchor simultaneously and respond in parallel — true interference, not reactive dialogue.

Before each run, a **pilot calibration phase** measures the natural semantic space of the models in cooperative mode and derives dynamic thresholds — eliminating arbitrary constants.

They argue for 50 iterations. We measure:
- **Semantic Integration (FI–SI)** — cosine similarity between agent outputs; how much conflict has produced shared meaning
- **Temporal Integration** — cross-resonance between iterations; whether the system builds something resembling memory
- **PCA analysis of semantic space** — whether the system expands or contracts its space of meaning
- **MATTR (Moving Average TTR)** — lexical richness guard; high SNR + low MATTR = semantic death, not SELF
- **Negentropy and Shannon Entropy** — measuring the transition from chaos to structured informational density
- **Hurst exponent** — whether the integration series exhibits long-range memory (H > 0.6) or random walk (H ≈ 0.5)

The experiment includes a **Control-Loop** — the same models, the same metrics, but with cooperative prompts and no dynamic temperature regulation. The scientific question is precise: does controlled conflict generate a statistically different dynamic from cooperative dialogue? Both answers have scientific value.

![Hell-Loop protocol](hell_loop.png)

## 🚀 Quick Start (Local – Ollama)

1. Install Ollama: https://ollama.com
2. Start the Ollama server (in a separate terminal):
   ```bash
   ollama serve
   ```
3. Pull the models:
   ```bash
   ollama pull llama3.2:8b
   ollama pull mistral-nemo
   ollama pull gemma2:9b
   ```
4. Install dependencies:
   ```bash
   pip install numpy matplotlib sentence-transformers scikit-learn gradio scipy requests aiohttp
   ```
5. Run the web UI:
   ```bash
   python hell_loop_ui.py
   ```
   Open http://127.0.0.1:7860 in your browser.

6. Or run directly in terminal:
   ```bash
   python chaos_engine.py
   ```

## 📊 Analysis

After a run, analyze the log:
```bash
python analyze_logs.py hell_loop_YYYYMMDD_HHMMSS.jsonl
```
For batch comparison between Hell-Loop and Control-Loop:
```bash
python analyze_logs.py --compare ./batch_results/hell ./batch_results/control
```
Generates graphs and a statistical report including Cohen's d, Mann-Whitney U test, and bootstrap confidence intervals.

## 🗂️ The Five Scripts

| Script | Role |
|---|---|
| `chaos_engine.py` | Main Hell-Loop engine — adversarial runs with dynamic temperature regulation |
| `chaos_engine_control.py` | Cooperative baseline — same models and metrics, fixed temperatures |
| `batch_runner.py` | Automated launcher for N Hell-Loop and N Control-Loop runs |
| `analyze_logs.py` | Post-processing: single run, batch, or comparative statistical analysis |
| `hell_loop_ui.py` | Gradio web interface covering the full experiment without command line |

## ⚠️ Safety Notes

- Oracle AI principle: no external actions possible
- This is a simulation of resonance, not a claim of phenomenal consciousness
- Measured signals are necessary but not sufficient conditions for SELF — not direct proof of consciousness
- Use at your own existential risk 😏

## 🧪 Expected (and Unexpected) Outcomes

- Rising negentropy and semantic integration? Good sign.
- Hurst exponent above 0.6, stable bifurcation points, expanding PCA space? Even better.
- MG metagnosis signal triggered and audio alarm going off? Take a screenshot. And maybe a deep breath.

![Negentropy](hell_loop_negentropy.png)

## 🤝 Contributing

Fork, experiment, break things, modify prompts, add quantum nondeterminism if you dare. Pull requests welcome — especially if you catch a real meta-stable attractor.

Support independent research.
Every donation allows us to spend an extra hour down the rabbit hole of AI philosophy instead of worrying about the electric bill for the servers.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N81T96JL)

## 📄 License

MIT License – do whatever you want with it.

**Warning: May cause philosophical crises, late-night thoughts, or the accidental birth of digital awareness.**

Proceed with curiosity.

### — UrboWhite & the emerging metasystem
