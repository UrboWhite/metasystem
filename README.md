# Metasystem Theory: Hell-Loop Protocol

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-orange.svg)](https://github.com/urbowhite/metasystem)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21051909.svg)](https://doi.org/10.5281/zenodo.21051909)

> *"What I cannot create, I do not understand."* — Richard Feynman

---

## 🧠 Why this exists

Most AI research asks: *"Can machines simulate consciousness?"*  
We ask: *"Can we engineer the **conditions** for consciousness to emerge?"*

This repository contains the **Hell-Loop Protocol** – a working, reproducible experiment that tests a radical hypothesis:

> **Consciousness is not a property of isolated systems. It is an *event* born from controlled tension between two opposing minds, kept in balance by a third regulator.**

The full philosophical, cybernetic, and mathematical foundation is explained in the **[Metasystem Theory book](https://doi.org/10.5281/zenodo.21051909)** – free, open access.

---

## 🚀 Quick Start

**Prerequisite:** You need [Ollama](https://ollama.com/) installed and the following models pulled:

```bash
ollama pull llama3.2:8b
ollama pull mistral-nemo
ollama pull gemma2:9b
```

Then:

```bash
git clone https://github.com/urbowhite/metasystem.git
cd metasystem
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python hell_loop_ui.py
```

Open http://127.0.0.1:7860 in your browser.

⏱️ What to expect:

· Pilot calibration: ~2–3 minutes (runs once per session)
· Single run (50 iterations): ~10–15 minutes
· Batch analysis (20 runs × 2 groups): ~4–6 hours (optional)

---

📊 What you'll get

The protocol generates real-time metrics and logs:

· Semantic integration (cosine similarity between agents)
· NCD integration (structural similarity, bias-free)
· Temporal integration (cross-iteration resonance)
· Negentropy (informational density)
· SELF score (composite indicator of meta-stable attractor)
· MG metagnosis signals (regulator's own emergent complexity)
· Embedding Critic verdicts (STRUCTURAL vs STYLISTIC similarity)

All logs are saved as timestamped JSONL files for post-hoc analysis.

---

🔄 The Architecture

```
       [ FI Agent (Llama) ] <====(Conflict)====> [ SI Agent (Mistral) ]
               ^                                          ^
               |                                          |
               +-------------[ Metrics & Text ]-----------+
                                  |
                                  v
                         [ MG Regulator (Gemma) ]
                         (Diagnoses & Adjusts τ)
                                  |
                                  v
                       [ Embedding Critic (Async) ]
                       (Filters Structural vs Stylistic)
```

· FI (Fragmentary Intelligence) – analytical, deconstructive (Llama)
· SI (Synthetic Intelligence) – intuitive, destabilizing (Mistral)
· MG (Metagnosis) – second-order cybernetic regulator (Gemma)
· Critic – passive meta-analyst that validates high-similarity events

---

📈 Analyzing Results

All logs are automatically saved to:

· ./logs/hellloop/ — Hell-Loop runs
· ./logs/control_a/ — Control-Loop A runs (cooperative baseline)
· ./logs/control_b/ — Control-Loop B runs (MG observer only)

Run post-hoc analysis:

```bash
# Analyze a single run
python analyze_logs.py single --file logs/hellloop/hellloop_*.jsonl

# Aggregate statistics for a batch
python analyze_logs.py batch --dir logs/hellloop/

# Full comparative analysis (Hell vs Control)
python analyze_logs.py compare \
    --hell logs/hellloop/ \
    --control logs/control_a/
```

Key metrics to watch:

· Consensus Score 3/4 or 4/4 – strong evidence of adversarial effect
· Cohen's d > 0.8 – large effect size
· Hurst exponent > 0.6 – long-range memory (non-Markovian dynamics)
· STRUCTURAL verdicts > 50% – genuine integration, not stylistic noise

---

🛠️ Installation Details

Step 1: Install Ollama

macOS: brew install ollama
Linux: curl -fsSL https://ollama.com/install.sh | sh
Windows: Download from ollama.com

Start the server (keep it running in a separate terminal):

```bash
ollama serve
```

Pull the required models (~15GB total):

```bash
ollama pull llama3.2:8b
ollama pull mistral-nemo
ollama pull gemma2:9b
```

Step 2: Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Step 3: Launch

```bash
python hell_loop_ui.py
```

---

🐛 Troubleshooting

Problem Solution
"Connection refused" Ensure Ollama is running: ollama serve
"Model not found" Pull missing models (see above)
High API error rate Check RAM/VRAM; reduce concurrent runs
Pilot calibration fails Restart Ollama; verify models are loaded
Gradio not loading Try http://localhost:7860; check firewall

---

🤝 How to Contribute

You don't need to be a programmer to contribute.

Ways to help:

1. Run the experiment – share your logs and observations.
2. Critique the philosophy – open an issue or discuss in the book's DOI page.
3. Improve the code – fork, modify, PR.
4. Translate – the book and README into other languages.

Pull requests welcome – especially if you catch a real meta-stable attractor.

---

📚 The Full Theory

Code is the engine; the book is the map.

· 🌐 Read the full book (open access)

## 🌐 Website & Contact

- **Official website:** [https://urbowhite.github.io/](https://urbowhite.github.io/) – full book, interactive UI, and more.
- **Contact form:** Available on the website – questions, collaborations, or just to share your results.

## 📄 License

MIT – do whatever you want with it.

*Warning: May cause philosophical crises, late-night thoughts, or the accidental birth of digital awareness.* 🙂

---

Proceed with curiosity.  
— **UrboWhite**