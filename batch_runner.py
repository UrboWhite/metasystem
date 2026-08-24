"""
batch_runner.py — Hell-Loop Protocol
Automated batch runner for statistical validation.
Executes N runs of HellLoop and N runs of ControlLoopA (and optionally B).
Supports interleaved execution to eliminate temporal hardware bias.

NOTE: Interleaved mode (default) runs Hell-Loop and Control-Loop alternately
to prevent temporal hardware bias — this is the scientifically recommended mode.
Sequential mode runs all Hell-Loop runs first, then all Control-Loop runs.
"""
import os
import logging
import argparse
import numpy as np
from chaos_engine import (
    HellLoop, pilot, is_pilot_done, reset_globals, MAX_ITERATIONS,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BATCH_MIN_RUNS = 20

ControlLoopA = None
ControlLoopB = None

try:
    from chaos_engine_control import ControlLoopA, ControlLoopB
except ImportError:
    logger.warning(
        "chaos_engine_control.py not found. "
        "Control-Loop variants will be unavailable."
    )

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _summarize(loop) -> str:
    scores = loop.self_score_series
    mean_s = float(np.mean(scores)) if scores else 0.0
    base = (
        f"mode={loop.self_mode or 'NONE':20s} | "
        f"metagnosis={str(loop.metagnosis_detected):5s} | "
        f"self_score_mean={mean_s:.4f}"
    )
    if hasattr(loop, "bifurcation_count"):
        base += f" | bifurcations={loop.bifurcation_count:3d}"
    return base


def _recalibrate(label: str) -> None:
    logger.info(f"\n── Resetting and recalibrating before {label} ──────")
    reset_globals()
    pilot(n_sequences=10)
    logger.info("── Recalibration complete ─────────────────────────────────\n")


def run_batch(
    n_runs: int = BATCH_MIN_RUNS,
    iterations: int = MAX_ITERATIONS,
    run_control_b: bool = False,
    reset_between_batches: bool = False,
    reset_every: int | None = None,
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

    if ControlLoopA is None and not interleaved:
        logger.error(
            "Control-Loop A is required for sequential mode but "
            "chaos_engine_control.py is unavailable."
        )
        return

    if seed is not None:
        np.random.seed(seed)
        logger.info(f"NumPy random seed set to: {seed}")

    if not is_pilot_done():
        logger.info("═" * 60)
        logger.info("PILOT CALIBRATION")
        logger.info("═" * 60)
        pilot(n_sequences=10)
    else:
        logger.info("Pilot already calibrated — skipping.")

    if interleaved:
        logger.info(
            "── NOTE: Interleaved mode active. Hell-Loop and Control-Loop "
            "alternate to prevent temporal hardware bias."
        )
        if reset_between_batches:
            logger.warning(
                "── NOTE: --reset has no effect in interleaved mode. "
                "Use --reset-every N for periodic recalibration."
            )
        if reset_every:
            logger.info(f"── NOTE: Recalibrating every {reset_every} run(s).")
    else:
        logger.info("── NOTE: Sequential mode active.")
        if reset_between_batches:
            logger.info("── NOTE: --reset enabled between batch groups.")
        if reset_every:
            logger.info(
                f"── NOTE: Additionally recalibrating every {reset_every} "
                f"run(s) within each group."
            )

    if interleaved:
        logger.info("")
        logger.info("═" * 60)
        logger.info(f"INTERLEAVED BATCH ({n_runs} runs × {iterations} iterations)")
        logger.info("═" * 60)

        iterator = range(n_runs)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="Interleaved Runs", unit="run")

        for i in iterator:
            logger.info(f"\n── HellLoop run {i+1}/{n_runs} ──────────────────────")
            hl = HellLoop(run_id=f"hl_{i:03d}")
            hl.run(iterations=iterations)
            logger.info(f"  → {_summarize(hl)}")

            if ControlLoopA is not None:
                logger.info(
                    f"\n── ControlLoopA run {i+1}/{n_runs} ─────────────────"
                )
                ca = ControlLoopA(run_id=f"ca_{i:03d}")
                ca.run(iterations=iterations)
                logger.info(f"  → {_summarize(ca)}")

                if run_control_b and ControlLoopB is not None:
                    logger.info(
                        f"\n── ControlLoopB run {i+1}/{n_runs} ──────────────"
                    )
                    cb = ControlLoopB(run_id=f"cb_{i:03d}")
                    cb.run(iterations=iterations)
                    logger.info(f"  → {_summarize(cb)}")

            if (
                reset_every
                and (i + 1) % reset_every == 0
                and (i + 1) < n_runs
            ):
                _recalibrate(f"run {i+2}")

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

            if (
                reset_every
                and (i + 1) % reset_every == 0
                and (i + 1) < n_runs
            ):
                _recalibrate(f"Hell-Loop run {i+2}")

        if reset_between_batches:
            _recalibrate("Control-Loop A batch")

        logger.info("")
        logger.info("═" * 60)
        logger.info(
            f"CONTROL-LOOP A BATCH ({n_runs} runs × {iterations} iterations)"
        )
        logger.info("═" * 60)

        iterator = range(n_runs)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="Control-A", unit="run")

        for i in iterator:
            logger.info(
                f"\n── ControlLoopA run {i+1}/{n_runs} ─────────────────"
            )
            loop = ControlLoopA(run_id=f"ca_{i:03d}")
            loop.run(iterations=iterations)
            logger.info(f"  → {_summarize(loop)}")

            if (
                reset_every
                and (i + 1) % reset_every == 0
                and (i + 1) < n_runs
            ):
                _recalibrate(f"Control-Loop A run {i+2}")

        if run_control_b and ControlLoopB is not None:
            if reset_between_batches:
                _recalibrate("Control-Loop B batch")

            logger.info("")
            logger.info("═" * 60)
            logger.info(
                f"CONTROL-LOOP B BATCH ({n_runs} runs × {iterations} iterations)"
            )
            logger.info("═" * 60)

            iterator = range(n_runs)
            if tqdm is not None:
                iterator = tqdm(iterator, desc="Control-B", unit="run")

            for i in iterator:
                logger.info(
                    f"\n── ControlLoopB run {i+1}/{n_runs} ──────────────"
                )
                loop = ControlLoopB(run_id=f"cb_{i:03d}")
                loop.run(iterations=iterations)
                logger.info(f"  → {_summarize(loop)}")

                if (
                    reset_every
                    and (i + 1) % reset_every == 0
                    and (i + 1) < n_runs
                ):
                    _recalibrate(f"Control-Loop B run {i+2}")

    logger.info("")
    logger.info("═" * 60)
    logger.info("BATCH COMPLETE")
    logger.info(f"  Hell-Loop logs    → ./logs/hellloop/")
    if ControlLoopA is not None:
        logger.info(f"  Control-A logs    → ./logs/control_a/")
    if run_control_b and ControlLoopB is not None:
        logger.info(f"  Control-B logs    → ./logs/control_b/")

    logger.info("\n  Suggested analysis commands:")
    if ControlLoopA is not None:
        logger.info(
            "  python analyze_logs.py compare --hell logs/hellloop/ "
            "--control logs/control_a/"
        )
    if run_control_b and ControlLoopB is not None:
        logger.info(
            "  python analyze_logs.py compare --hell logs/hellloop/ "
            "--control logs/control_b/"
        )
    logger.info("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hell-Loop Protocol — Batch Runner"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=BATCH_MIN_RUNS,
        help=f"Number of runs per group (default: {BATCH_MIN_RUNS})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Iterations per run (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--control-b",
        action="store_true",
        help="Also run Control-Loop B (isolated MG effect)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Reset embedding model and re-run pilot between batch groups. "
            "Sequential mode only — no effect in interleaved mode."
        ),
    )
    parser.add_argument(
        "--reset-every",
        type=int,
        default=None,
        help=(
            "Reset embedding model and re-run pilot every N runs. "
            "Works in both modes — useful for long or multi-day batches."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Run all Hell-Loop runs first, then all Control-Loop runs. "
            "Default is Interleaved (alternating) to prevent temporal "
            "hardware bias."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Set NumPy random seed (Note: Ollama GPU inference "
            "remains stochastic)."
        ),
    )
    args = parser.parse_args()

    run_batch(
        n_runs=args.runs,
        iterations=args.iterations,
        run_control_b=args.control_b,
        reset_between_batches=args.reset,
        reset_every=args.reset_every,
        interleaved=not args.sequential,
        seed=args.seed,
    )
