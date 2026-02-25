import json
import argparse
import os
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import seaborn as sns

# --------------------- CONSTANTS ---------------------
# Ove vrednosti moraju biti usklađene sa chaos_engine.py:
#   HIGH_COHERENCE_DROP = 1 - COHERENCE_THRESHOLD (0.3) = 0.7
#   MIN_NEGENTROPY — empirijska granica za "informaciono bogat" odgovor
HIGH_COHERENCE_DROP = 0.7   # threshold za "visok drop" u grafovima i izveštaju
MIN_NEGENTROPY      = 1.5   # minimalna negentropija za emergentni signal

def load_log(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL log file with improved error handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    history = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Parse error on line {line_num}: {e}")
    return history



def plot_negentropy(history, output_dir: Path):
    its = [h["iter"] for h in history]
    neg_avg = [h["neg_avg"] for h in history]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.plot(its, neg_avg, 'r-o', linewidth=2, markersize=6, label="Average negentropy")
    ax.set_title("Evolution of information density (negentropy)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Negentropy (gzip ratio)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "negentropy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] negentropy_curve.png")


def plot_entropy(history, output_dir: Path):
    its = [h["iter"] for h in history]
    H_avg = [h["H_avg"] for h in history]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.plot(its, H_avg, 'b--', label="H avg (FI+SI)")
    ax.set_title("Shannon entropy – FI+SI average")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy (bits/symbol)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "shannon_entropy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] shannon_entropy_curve.png")


def plot_creativity(history, output_dir: Path):
    its = [h["iter"] for h in history]
    C_avg = [h["creat_avg"] for h in history]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.plot(its, C_avg, 'm-', label="Creativity (avg)")
    ax.set_title("Response divergence (creativity)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Creativity (1 - similarity)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "creativity_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] creativity_curve.png")


def plot_coherence_drop(history, output_dir: Path):
    its = [h["iter"] for h in history]
    coh_drop = [h.get("coherence_drop", 0.0) for h in history]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.plot(its, coh_drop, 'c-o', linewidth=2, markersize=6, label="Coherence drop (FI vs SI)")
    ax.axhline(HIGH_COHERENCE_DROP, color='orange', linestyle='--', label="High drop threshold")
    ax.set_title("Coherence drop – koliko su FI i SI izgubili nit")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coherence drop (1 - similarity)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "coherence_drop_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] coherence_drop_curve.png")


def plot_coherence_vs_negentropy(history, output_dir: Path):
    neg = [h["neg_avg"] for h in history]
    coh = [h.get("coherence_drop", 0.0) for h in history]
    its = [h["iter"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    scatter = ax.scatter(neg, coh, c=its, cmap='viridis', s=80, alpha=0.7)
    ax.set_title("Coherence drop vs Negentropy – emergentni kandidati?")
    ax.set_xlabel("Average negentropy")
    ax.set_ylabel("Coherence drop")
    plt.colorbar(scatter, label="Iteration")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "coherence_vs_negentropy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] coherence_vs_negentropy.png")


def plot_events_heatmap(history, output_dir: Path):
    its = [h["iter"] for h in history]
    mg = [1 if h.get("metagnosis", False) else 0 for h in history]
    self_s = [1 if h.get("self_synthesis", False) else 0 for h in history]
    high_drop = [1 if h.get("coherence_drop", 0) > HIGH_COHERENCE_DROP else 0 for h in history]

    data = np.array([mg, self_s, high_drop])
    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    sns.heatmap(data, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=its, yticklabels=["Metagnosis", "SELF-synthesis", "High coh. drop"],
                cbar=False, ax=ax)
    ax.set_title("Key event detections + high coherence drop")
    ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(output_dir / "events_heatmap_v2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] events_heatmap_v2.png")


def generate_report(history, output_dir: Path):
    total = len(history)
    if total == 0:
        print("[INFO] No data available.")
        return

    neg_values = [h["neg_avg"] for h in history]
    H_values = [h["H_avg"] for h in history]
    C_values = [h["creat_avg"] for h in history]
    coh_values = [h.get("coherence_drop", 0.0) for h in history]

    meta_count = sum(1 for h in history if h.get("metagnosis", False))
    self_count = sum(1 for h in history if h.get("self_synthesis", False))
    high_drop_count = sum(1 for h in history if h.get("coherence_drop", 0) > HIGH_COHERENCE_DROP)

    lines = [
        "=== HELL-LOOP EXPERIMENTAL REPORT v2.2 ===\n",
        f"Total iterations: {total}",
        f"Average negentropy: {np.mean(neg_values):.4f} (max: {np.max(neg_values):.4f})",
        f"Average Shannon entropy: {np.mean(H_values):.4f} (min: {np.min(H_values):.4f})",
        f"Average creativity: {np.mean(C_values):.4f} (max spike: {np.max(C_values):.4f})",
        f"Average coherence drop: {np.mean(coh_values):.4f} (max: {np.max(coh_values):.4f})",
        f"Metagnosis detections: {meta_count}",
        f"SELF-synthesis detections: {self_count}",
        f"High coherence drop (>0.7): {high_drop_count}\n",
        "Most interesting iterations (potential emergent signals):",
    ]

    # Top 3 po kombinaciji: visok drop + visoka negentropija
    candidates = []
    for idx, h in enumerate(history):
        drop = h.get("coherence_drop", 0)
        neg = h["neg_avg"]
        if drop > HIGH_COHERENCE_DROP and neg > MIN_NEGENTROPY:
            score = drop * neg
            candidates.append((idx+1, drop, neg, score, h.get("metagnosis", False), h.get("self_synthesis", False)))

    top_candidates = sorted(candidates, key=lambda x: x[3], reverse=True)[:5]
    for it, dr, ne, sc, mg, ss in top_candidates:
        flag = ""
        if mg: flag += " [METAGNOSIS]"
        if ss: flag += " [SELF]"
        lines.append(f"  Iter {it:3d} → coh.drop {dr:.3f} | neg {ne:.3f} | score {sc:.3f}{flag}")

    text = "\n".join(lines) + "\n=== END ===\n"

    report_path = output_dir / "experimental_report_v2.2.txt"
    report_path.write_text(text, encoding="utf-8")
    print(f"[OK] Report saved: {report_path}")


def export_pdf(history, output_dir: Path, log_name: str):
    """Sve grafove pakuje u jedan PDF — zgodno za deljenje rezultata."""
    pdf_path = output_dir / f"{Path(log_name).stem}_report.pdf"
    its = [h["iter"] for h in history]

    with pdf_backend.PdfPages(pdf_path) as pdf:
        # Negentropy
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(its, [h["neg_avg"] for h in history], 'r-o')
        ax.set_title("Negentropy")
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Shannon entropy
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(its, [h["H_avg"] for h in history], 'b--')
        ax.set_title("Shannon entropy")
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Creativity
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(its, [h["creat_avg"] for h in history], 'm-')
        ax.set_title("Creativity (divergence)")
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Coherence drop
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(its, [h.get("coherence_drop", 0.0) for h in history], 'c-o')
        ax.axhline(HIGH_COHERENCE_DROP, color='orange', linestyle='--', label="Threshold")
        ax.set_title("Coherence drop")
        ax.set_xlabel("Iteration")
        ax.legend()
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Coherence vs negentropy scatter
        neg = [h["neg_avg"] for h in history]
        coh = [h.get("coherence_drop", 0.0) for h in history]
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(neg, coh, c=its, cmap='viridis', s=80, alpha=0.7)
        plt.colorbar(sc, label="Iteration")
        ax.set_title("Coherence drop vs Negentropy")
        ax.set_xlabel("Negentropy")
        ax.set_ylabel("Coherence drop")
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    print(f"[OK] PDF report: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Hell-Loop log analyzer v2.3 – sa PDF exportom!")
    parser.add_argument("logfile", type=str, nargs="?", help="Path to JSONL log (or folder)")
    parser.add_argument("--all", action="store_true", help="Process all .jsonl files in folder")
    parser.add_argument("--pdf", action="store_true", help="Export sve grafove u jedan PDF po fajlu")
    args = parser.parse_args()

    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    files = []
    if args.all:
        folder = Path(args.logfile or ".")
        files = list(folder.glob("*.jsonl"))
    elif args.logfile:
        files = [Path(args.logfile)]
    else:
        parser.print_help()
        return

    if not files:
        print("[ERROR] No log files found.")
        return

    for log_path in files:
        print(f"\n=== Processing: {log_path.name} ===")
        try:
            history = load_log(str(log_path))
            if not history:
                print("  → Empty log, skipping.")
                continue

            plot_negentropy(history, output_dir)
            plot_entropy(history, output_dir)
            plot_creativity(history, output_dir)
            plot_coherence_drop(history, output_dir)
            plot_coherence_vs_negentropy(history, output_dir)
            plot_events_heatmap(history, output_dir)
            generate_report(history, output_dir)

            if args.pdf:
                export_pdf(history, output_dir, log_path.name)

            print(f"[OK] Done for {log_path.name}")
        except Exception as e:
            print(f"[ERROR] {log_path.name} → {e}")

    print(f"\nAll done. Results in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()