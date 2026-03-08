import logging
from chaos_engine import HellLoop
from analyze_logs import get_system_status

def run_batch_v6(runs=20, iterations=30):
    """Executes automated stress-testing of the Metasystem."""
    for i in range(runs):
        loop = HellLoop(run_id=i)
        current_context = "The origin of synthetic recursion."
        
        for _ in range(iterations):
            record = loop.step(current_context)
            current_context = record["si_response"] # Feed SI back to next FI

        status = get_system_status(loop.history)
        print(f"Run {i:03d} | Status: {status}")
