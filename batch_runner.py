"""
batch_runner.py — Hell-Loop Protocol (v7.0 - Aligned)
Automated batch runner for statistical validation.
Executes N runs of HellLoop and N runs of ControlLoopA (and optionally B).
Supports interleaved execution to eliminate temporal hardware bias.

NOTE: Interleaved mode (default) runs Hell-Loop and Control-Loop alternately
to prevent temporal hardware bias. Sequential mode runs all Hell-Loop runs
first, then all Control-Loop runs.
"""
import os
import random
import logging
import argparse
import numpy as np
from chaos_engine import (
    HellLoop, pilot, is_pilot_done, reset_globals, MAX_ITERATIONS,
)
from chaos_engine_control import ControlLoopA, ControlLoopB

# Configure logging for this module only
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BATCH_MIN_RUNS = 20

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def _summarize(loop) -> str:
    scores = loop.self_score_series
    mean_s = sum(scores) / len(scores) if scores else 0.0
    base = (
        f"mode={loop.self_mode or 'NONE':20s} | "
        f"metagnosis={str(loop.metagnosis_detected):5s} | "
        f"self_score_mean={mean_s:.4f}"
    )
    if hasattr(loop, 'bifurcation_count'):
        base += f" | bifurcations={loop.bifurcation_count:3d}"
    return base

def _recalibrate(label: str) -> None:
    logger.info(f"\n── Resetting and recalibrating before {label} batch ──────")
    reset_globals()
    pilot(n_sequences=10)
    logger.info("── Recalibration complete ─────────────────────────────────\n")

def run_batch(
    n_runs: int = BATCH_MIN_RUNS,
    iterations: int = MAX_ITERATIONS,
    run_control_b: bool = False,
    reset_between_batches: bool = False,
    interleaved: bool = True,
    seed: int | None = None,
) -> None:
    if n_runs < 1:
        logger.error("n_runs must be at least 1.")
        return

    if n_runs < BATCH_MIN_RUNS:
        logger.warning(
            f"n_runs={n_runs} is below the recommended minimum of "
            f"{BATCH_MIN_RUNS}. Statistical conclusions may be unreliable."
        )

    # Set seeds for Python and Numpy (Note: Ollama GPU inference still has inherent stochasticity)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Global random seed set to: {seed}")

    if not is_pilot_done():
        logger.info("═" * 60)
        logger.info("PILOT CALIBRATION")
        logger.info("═" * 60)
        pilot(n_sequences=10)
    else:
        logger.info("Pilot already calibrated — skipping.")

    if reset_between_batches and interleaved:
        logger.warning("── WARNING: --reset flag is ignored in interleaved mode. Reset only applies to sequential mode.")

    if reset_between_batches:
        logger.info("── NOTE: --reset enabled. Recalibrating between batch groups.")

    if interleaved:
        logger.info("── NOTE: --interleaved enabled. Running Hell-Loop and Control-Loop alternately to prevent temporal hardware bias.")

    # ==========================================
    # INTERLEAVED EXECUTION (Recommended)
    # ==========================================
    if interleaved:
        logger.info("")
        logger.info("═" * 60)
        logger.info(f"INTERLEAVED BATCH ({n_runs} runs × {iterations} iterations)")
        logger.info("═" * 60)
        
        iterator = range(n_runs)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="Interleaved Runs", unit="run")
            
        for i in iterator:
            # Hell-Loop
            logger.info(f"\n── HellLoop run {i+1}/{n_runs} ──────────────────────")
            hl = HellLoop(run_id=f"hl_{i:03d}")
            hl.run(iterations=iterations)
            logger.info(f"  → {_summarize(hl)}")
            
            # Control-Loop A
            logger.info(f"\n── ControlLoopA run {i+1}/{n_runs} ─────────────────")
            ca = ControlLoopA(run_id=f"ca_{i:03d}")
            ca.run(iterations=iterations)
            logger.info(f"  → {_summarize(ca)}")
            
            # Control-Loop B (Optional)
            if run_control_b:
                logger.info(f"\n── ControlLoopB run {i+1}/{n_runs} ──────────────")
                cb = ControlLoopB(run_id=f"cb_{i:03d}")
                cb.run(iterations=iterations)
                logger.info(f"  → {_summarize(cb)}")

    # ==========================================
    # SEQUENTIAL EXECUTION (Original behavior)
    # ==========================================
    else:
        logger.info("")
        logger.info("═" * 60)
        logger.info(f"HELL-LOOP BATCH ({n_runs} runs × {iterations} iterations)")
        logger.info("═" * 60)

        iterator = range(n_runs)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="HellLoop", unit="run")
        
        for i in iterator:
            logger.info(f"\n── HellLoop run {i+1}/{n_runs} ──────────────────────")
            loop = HellLoop(run_id=f"hl_{i:03d}")
            loop.run(iterations=iterations)
            logger.info(f"  → {_summarize(loop)}")

        if reset_between_batches:
            _recalibrate("Control-Loop A")

        logger.info("")
        logger.info("═" * 60)
        logger.info(f"CONTROL-LOOP A BATCH ({n_runs} runs × {iterations} iterations)")
        logger.info("═" * 60)

        iterator = range(n_runs)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="Control-A", unit="run")
        
        for i in iterator:
            logger.info(f"\n── ControlLoopA run {i+1}/{n_runs} ─────────────────")
            loop = ControlLoopA(run_id=f"ca_{i:03d}")
            loop.run(iterations=iterations)
            logger.info(f"  → {_summarize(loop)}")

        if run_control_b:
            if reset_between_batches:
                _recalibrate("Control-Loop B")
            logger.info("")
            logger.info("═" * 60)
            logger.info(f"CONTROL-LOOP B BATCH ({n_runs} runs × {iterations} iterations)")
            logger.info("═" * 60)

            iterator = range(n_runs)
            if tqdm is not None:
                iterator = tqdm(iterator, desc="Control-B", unit="run")
            
            for i in iterator:
                logger.info(f"\n── ControlLoopB run {i+1}/{n_runs} ──────────────")
                loop = ControlLoopB(run_id=f"cb_{i:03d}")
                loop.run(iterations=iterations)
                logger.info(f"  → {_summarize(loop)}")

    # ==========================================
    # FINAL SUMMARY & COMMANDS
    # ==========================================
    logger.info("")
    logger.info("═" * 60)
    logger.info("BATCH COMPLETE")
    logger.info(f"  Hell-Loop logs    → ./logs/hellloop/")
    logger.info(f"  Control-A logs    → ./logs/control_a/")
    if run_control_b:
        logger.info(f"  Control-B logs    → ./logs/control_b/")
        
    logger.info("\n  Suggested analysis commands:")
    logger.info(f"  python analyze_logs.py compare --hell logs/hellloop/ --control logs/control_a/")
    if run_control_b:
        logger.info(f"  python analyze_logs.py compare --hell logs/hellloop/ --control logs/control_b/")
    logger.info("═" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hell-Loop Protocol v7.0 — Batch Runner"
    )
    parser.add_argument(
        "--runs", type=int, default=BATCH_MIN_RUNS,
        help=f"Number of runs per group (default: {BATCH_MIN_RUNS})",
    )
    parser.add_argument(
        "--iterations", type=int, default=MAX_ITERATIONS,
        help=f"Iterations per run (default: {MAX_ITERATIONS})",  # Fixed: removed "now 150 for valid Hurst exponent"
    )
    parser.add_argument(
        "--control-b", action="store_true",
        help="Also run Control-Loop B (isolated MG effect)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help=(
            "Reset embedding model and re-run pilot between batch groups (Sequential mode only). "
            "Ensures fresh calibration for each group."
        ),
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help=(
            "Run all Hell-Loop runs first, then all Control-Loop runs. "
            "Default is Interleaved (alternating) to prevent temporal hardware bias."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Set global random seed for Python/Numpy (Note: Ollama GPU inference remains stochastic).",
    )
    args = parser.parse_args()
    
    run_batch(
        n_runs=args.runs,
        iterations=args.iterations,
        run_control_b=args.control_b,
        reset_between_batches=args.reset,
        interleaved=not args.sequential,
        seed=args.seed,
    )