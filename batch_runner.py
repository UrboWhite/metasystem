"""
batch_runner.py — Hell-Loop Protocol Automated Batch Runner
Executes N Hell-Loop runs and N Control-Loop runs, saves JSONL logs,
and prints a comparative statistical summary.

Usage:
    python batch_runner.py [--runs 20] [--iterations 50] [--pilot 10]
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from chaos_engine import HellLoop, MAX_ITERATIONS, pilot, is_pilot_done
from chaos_engine_control import ControlLoop
from analyze_logs import save_jsonl, batch_summary, compare, print_comparison

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _make_log_dir(base: str = "logs") -> Path:
    """Create a timestamped log directory for this batch session."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_batch(n_runs: int = 20, n_iterations: int = MAX_ITERATIONS, n_pilot: int = 10):
    """
    Execute n_runs of Hell-Loop and n_runs of Control-Loop.
    - Pilot calibration runs once before all batch runs.
    - Each run is saved as a separate JSONL file.
    - Comparative statistics are printed at the end.
    """
    log_dir = _make_log_dir()
    logger.info(f"\n{'═' * 60}")
    logger.info(f"  HELL-LOOP BATCH RUNNER")
    logger.info(f"  runs={n_runs}  iterations={n_iterations}  pilot={n_pilot}")
    logger.info(f"  Logs → {log_dir}")
    logger.info(f"{'═' * 60}\n")

    # ── Pilot calibration (once per batch session) ────────────────────────────
    if not is_pilot_done():
        pilot(n_sequences=n_pilot)
    else:
        logger.info("[batch] Pilot already calibrated, skipping.")

    hell_histories = []
    ctrl_histories = []

    # ── Hell-Loop runs ────────────────────────────────────────────────────────
    logger.info(f"\n{'─' * 40}")
    logger.info(f"  HELL-LOOP — {n_runs} adversarial runs")
    logger.info(f"{'─' * 40}")

    for i in range(n_runs):
        logger.info(f"\n[hell run {i:03d}/{n_runs}]")
        loop    = HellLoop(run_id=i)
        context = "The origin of synthetic recursion."

        for _ in range(n_iterations):
            record  = loop.step(context)
            context = record["si_response"]

        hell_histories.append(loop.history)
        save_jsonl(loop.history, str(log_dir / f"hell_run_{i:03d}.jsonl"))
        logger.info(f"  → saved hell_run_{i:03d}.jsonl | SELF={loop.self_mode or '-'} MG={loop.metagnosis_detected}")

    # ── Control-Loop runs ─────────────────────────────────────────────────────
    logger.info(f"\n{'─' * 40}")
    logger.info(f"  CONTROL-LOOP — {n_runs} cooperative runs")
    logger.info(f"{'─' * 40}")

    for i in range(n_runs):
        logger.info(f"\n[ctrl run {i:03d}/{n_runs}]")
        loop = ControlLoop()

        for _ in range(n_iterations):
            loop.step()

        ctrl_histories.append(loop.history)
        save_jsonl(loop.history, str(log_dir / f"ctrl_run_{i:03d}.jsonl"))
        logger.info(f"  → saved ctrl_run_{i:03d}.jsonl | SELF={loop.self_mode or '-'} MG={loop.metagnosis_detected}")

    # ── Batch summaries ───────────────────────────────────────────────────────
    hell_summary = batch_summary(hell_histories)
    ctrl_summary = batch_summary(ctrl_histories)

    logger.info(f"\n{'═' * 60}")
    logger.info("  HELL-LOOP BATCH SUMMARY")
    for k, v in hell_summary.items():
        logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    logger.info(f"\n  CONTROL-LOOP BATCH SUMMARY")
    for k, v in ctrl_summary.items():
        logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # ── Comparative analysis ──────────────────────────────────────────────────
    if len(hell_histories) >= 2 and len(ctrl_histories) >= 2:
        results = compare(hell_histories, ctrl_histories)
        print_comparison(results)
        save_jsonl([results], str(log_dir / "comparison.jsonl"))
        logger.info(f"[batch] Comparison saved → {log_dir / 'comparison.jsonl'}")
    else:
        logger.warning("[batch] Not enough runs for statistical comparison (need ≥ 2 per group).")

    logger.info(f"\n[batch] Done. All logs in: {log_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hell-Loop Batch Runner")
    parser.add_argument("--runs",       type=int, default=20,            help="Number of runs per group (default: 20)")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS, help=f"Iterations per run (default: {MAX_ITERATIONS})")
    parser.add_argument("--pilot",      type=int, default=10,            help="Pilot calibration sequences (default: 10)")
    args = parser.parse_args()

    run_batch(n_runs=args.runs, n_iterations=args.iterations, n_pilot=args.pilot)
