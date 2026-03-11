import logging
from chaos_engine import HellLoop, pilot, is_pilot_done
from analyze_logs import get_system_status

logger = logging.getLogger(__name__)

def run_batch_v6(runs=20, iterations=30):
    """
    Executes automated stress-testing of the Metasystem.
    FIX: pilot() is now called once before the batch so all HellLoop
    instances use calibrated thresholds instead of fallback constants.
    """
    # ── Pilot calibration (once per batch) ────────────────────────────────────
    if not is_pilot_done():
        logger.info("Running pilot calibration before batch...")
        pilot(n_sequences=10)
    else:
        logger.info("Pilot already calibrated — skipping.")

    # ── Batch runs ─────────────────────────────────────────────────────────────
    for i in range(runs):
        loop = HellLoop(run_id=i)
        current_context = "The origin of synthetic recursion."

        for _ in range(iterations):
            record = loop.step(current_context)
            current_context = record["si_response"]  # Feed SI back to next FI

        status = get_system_status(loop.history)
        print(f"Run {i:03d} | SELF={loop.self_mode or '-':30s} | Status: {status}")
