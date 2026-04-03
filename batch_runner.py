"""
batch_runner.py — Hell-Loop Protocol
Automated batch runner for statistical validation.

Executes N runs of HellLoop and N runs of ControlLoopA sequentially.
ControlLoopB is optional (run_control_b flag).
Pilot calibration runs once before the entire batch.
Each run is saved as a separate JSONL file in its respective log directory.
"""

import logging
import argparse

from chaos_engine import (
    HellLoop, pilot, is_pilot_done, reset_globals, MAX_ITERATIONS,
)
from chaos_engine_control import ControlLoopA, ControlLoopB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

BATCH_MIN_RUNS = 20   # minimum scientifically relevant runs per group


def _summarize(loop) -> str:
    scores = loop.self_score_series
    mean_s = sum(scores) / len(scores) if scores else 0.0
    return (
        f"mode={loop.self_mode or 'NONE':20s} | "
        f"metagnosis={str(loop.metagnosis_detected):5s} | "
        f"self_score_mean={mean_s:.4f}"
    )


def _recalibrate(label: str) -> None:
    """Reset globals and re-run pilot. Used between batch groups when requested."""
    logger.info(f"\n── Resetting and recalibrating before {label} batch ──────")
    reset_globals()
    pilot(n_sequences=10)
    logger.info("── Recalibration complete ─────────────────────────────────\n")


def run_batch(
    n_runs: int = BATCH_MIN_RUNS,
    iterations: int = MAX_ITERATIONS,
    run_control_b: bool = False,
    reset_between_batches: bool = False,
) -> None:
    """
    Run the full batch experiment.

    Parameters
    ----------
    n_runs : int
        Number of runs per group (HellLoop, ControlLoopA, optionally ControlLoopB).
    iterations : int
        Number of iterations per run.
    run_control_b : bool
        Whether to also run ControlLoopB (isolated MG effect).
    reset_between_batches : bool
        If True, resets embedding model and pilot stats between each batch group
        and re-runs pilot calibration. Ensures each group starts from a clean,
        freshly-calibrated state. Useful for long batches or multi-day runs.
        Note: adds one pilot run (10 sequences) between each group.
    """
    if n_runs < BATCH_MIN_RUNS:
        logger.warning(
            f"n_runs={n_runs} is below the recommended minimum of "
            f"{BATCH_MIN_RUNS}. Statistical conclusions may be unreliable."
        )

    # ── Pilot calibration — once before the first batch group ─────────────────
    if not is_pilot_done():
        logger.info("═" * 60)
        logger.info("PILOT CALIBRATION")
        logger.info("═" * 60)
        pilot(n_sequences=10)
    else:
        logger.info("Pilot already calibrated — skipping.")

    # ── Hell-Loop batch ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 60)
    logger.info(f"HELL-LOOP BATCH  ({n_runs} runs × {iterations} iterations)")
    logger.info("═" * 60)

    for i in range(n_runs):
        logger.info(f"\n── HellLoop run {i+1}/{n_runs} ──────────────────────")
        loop = HellLoop(run_id=f"hl_{i:03d}")
        loop.run(iterations=iterations)
        logger.info(f"  → {_summarize(loop)}")

    # ── Optional reset + recalibration before Control-A ───────────────────────
    if reset_between_batches:
        _recalibrate("Control-Loop A")

    # ── Control-Loop A batch ───────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 60)
    logger.info(f"CONTROL-LOOP A BATCH  ({n_runs} runs × {iterations} iterations)")
    logger.info("═" * 60)

    for i in range(n_runs):
        logger.info(f"\n── ControlLoopA run {i+1}/{n_runs} ─────────────────")
        loop = ControlLoopA(run_id=f"ca_{i:03d}")
        loop.run(iterations=iterations)
        logger.info(f"  → {_summarize(loop)}")

    # ── Control-Loop B batch (optional) ───────────────────────────────────────
    if run_control_b:
        # Optional reset + recalibration before Control-B
        if reset_between_batches:
            _recalibrate("Control-Loop B")

        logger.info("")
        logger.info("═" * 60)
        logger.info(
            f"CONTROL-LOOP B BATCH  ({n_runs} runs × {iterations} iterations)"
        )
        logger.info("═" * 60)

        for i in range(n_runs):
            logger.info(f"\n── ControlLoopB run {i+1}/{n_runs} ──────────────")
            loop = ControlLoopB(run_id=f"cb_{i:03d}")
            loop.run(iterations=iterations)
            logger.info(f"  → {_summarize(loop)}")

    logger.info("")
    logger.info("═" * 60)
    logger.info("BATCH COMPLETE")
    logger.info(f"  Hell-Loop logs    → ./logs/hellloop/")
    logger.info(f"  Control-A logs    → ./logs/control_a/")
    if run_control_b:
        logger.info(f"  Control-B logs    → ./logs/control_b/")
    logger.info(f"  Run analyze_logs.py for statistical analysis.")
    logger.info("═" * 60)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hell-Loop Protocol v6.2 — Batch Runner"
    )
    parser.add_argument(
        "--runs", type=int, default=BATCH_MIN_RUNS,
        help=f"Number of runs per group (default: {BATCH_MIN_RUNS})",
    )
    parser.add_argument(
        "--iterations", type=int, default=MAX_ITERATIONS,
        help=f"Iterations per run (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--control-b", action="store_true",
        help="Also run Control-Loop B (isolated MG effect)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help=(
            "Reset embedding model and re-run pilot between batch groups. "
            "Ensures fresh calibration for each group."
        ),
    )
    args = parser.parse_args()

    run_batch(
        n_runs=args.runs,
        iterations=args.iterations,
        run_control_b=args.control_b,
        reset_between_batches=args.reset,
    )
