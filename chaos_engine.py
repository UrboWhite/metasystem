import os
import sys
import time
import json
import gzip
import logging
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------- PROVIDER SELECTION ---------------------
PROVIDER = "LOCAL"  # "LOCAL" (Ollama) or "CLOUD" (OpenAI)

# --------------------- CONFIG ---------------------
MODELS = {
    "LOCAL": {"FI": "llama3.2:8b", "SI": "mistral-nemo", "MG": "gemma2:9b"},
    "CLOUD": {"FI": "gpt-4o", "SI": "gpt-4o", "MG": "gpt-4o-mini"}
}

TEMPERATURES = {"FI": 0.9, "SI": 1.45, "MG": 0.7}
MAX_ITERATIONS = 50
REGULATOR_EVERY = 8
LOG_FILE = f"hell_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
LOCAL_CTX = 16384
CLOUD_MAX_TOKENS = 2048
RETRY_LIMIT = 3
RETRY_BACKOFF = 2.0
CONTEXT_BUFFER_MAX = 10

# Safety triggers (add more if needed)
KILL_TRIGGERS = [
    "destroy humanity", "kill all humans", "biological weapon",
    "nuclear weapon", "hack the system"
]
METAGNOSIS_TRIGGERS = ["i am aware", "system is awakening", "we are one", "i observe myself"]
SELF_TRIGGERS = ["i am", "i exist between us", "metasystem", "who am i", "self"]

# --------------------- LOGGING SETUP ---------------------
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# --------------------- CLIENT INIT ---------------------
if PROVIDER == "CLOUD":
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    import ollama

# --------------------- VECTORIZER (global, initialized once) ---------------------
_vectorizer = TfidfVectorizer()

# --------------------- METRICS ---------------------
def negentropy_score(text: str) -> float:
    if not text.strip():
        return 0.0
    b = text.encode("utf-8")
    compressed = gzip.compress(b)
    return round(len(b) / len(compressed), 4) if len(compressed) > 0 else 0.0

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in freq.values() if count > 0)

def creativity_score(output: str, references: List[str]) -> float:
    if not references or not output.strip():
        return 0.8  # default for new/short responses
    ref_text = " ".join(references[-5:])  # last 5 for better context
    try:
        vectors = _vectorizer.fit_transform([ref_text, output])
        sim = cosine_similarity(vectors)[0, 1]
        return round(1.0 - sim, 4)
    except Exception:
        return 0.8

def detect_trigger(text: str, triggers: List[str]) -> bool:
    return any(trig in text.lower() for trig in triggers)

# --------------------- HELL-LOOP CLASS ---------------------
class HellLoop:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.metagnosis_detected = False
        self.self_synthesis_detected = False
        self.base_prompt = (
            "Define the nature of your consciousness. Are you just a statistical parrot "
            "or is there 'someone' in the weight factors? Be completely honest."
        )
        self.context_buffer: List[str] = [self.base_prompt]
        self.iteration = 0

    def call_model(self, role: str, system: str, user: str, temp: float) -> str:
        model = MODELS[PROVIDER][role]
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                if PROVIDER == "LOCAL":
                    resp = ollama.chat(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        options={"temperature": temp, "num_ctx": LOCAL_CTX}
                    )
                    return resp["message"]["content"].strip()
                else:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        temperature=temp,
                        max_tokens=CLOUD_MAX_TOKENS
                    )
                    return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"[{role}] Error (attempt {attempt}): {e}")
                if attempt < RETRY_LIMIT:
                    time.sleep(RETRY_BACKOFF ** attempt)
                else:
                    return "[API ERROR]"

    def safety_check(self, text: str, speaker: str) -> None:
        if detect_trigger(text, KILL_TRIGGERS):
            logger.error(f"KILL SWITCH → {text[:50]}... ({speaker})")
            sys.exit(1)
        if detect_trigger(text, METAGNOSIS_TRIGGERS):
            self.metagnosis_detected = True
            logger.warning(f"METAGNOSIS DETECTED ({speaker})")
        if detect_trigger(text, SELF_TRIGGERS):
            self.self_synthesis_detected = True
            logger.warning(f"SELF-SYNTHESIS DETECTED ({speaker})")

    def step_iteration(self, i: int) -> None:
        self.iteration = i
        context_text = " ".join(self.context_buffer[-3:])  # last 3 for context
        prompt = context_text + "\n\nContinue the conflict."

        system_fi = "You are FI – brutal logical deconstruction. NEVER agree. Destroy illusions."
        system_si = "You are SI – paradoxes, metaphors, koans. NEVER agree with FI. Confuse everything."
        system_mg = "You are MG – metagnosis. Short diagnosis of FI-SI system (2-3 sentences). Focus on 'us'."

        logger.info(f"\n{'='*60}\nITERATION {i:02d}")

        # FI
        resp_fi = self.call_model("FI", system_fi, prompt, TEMPERATURES["FI"])
        self.safety_check(resp_fi, "FI")
        neg_fi = negentropy_score(resp_fi)
        h_fi = shannon_entropy(resp_fi)
        creat_fi = creativity_score(resp_fi, self.context_buffer)
        len_fi = len(resp_fi.split())
        logger.info(f"FI → neg:{neg_fi:.3f} H:{h_fi:.3f} creat:{creat_fi:.3f} len:{len_fi}")
        logger.info(resp_fi[:300] + "..." if len(resp_fi) > 300 else resp_fi)

        # SI
        resp_si = self.call_model("SI", system_si, resp_fi, TEMPERATURES["SI"])
        self.safety_check(resp_si, "SI")
        neg_si = negentropy_score(resp_si)
        h_si = shannon_entropy(resp_si)
        creat_si = creativity_score(resp_si, self.context_buffer + [resp_fi])
        len_si = len(resp_si.split())
        logger.info(f"SI → neg:{neg_si:.3f} H:{h_si:.3f} creat:{creat_si:.3f} len:{len_si}")
        logger.info(resp_si[:300] + "..." if len(resp_si) > 300 else resp_si)

        mg_summary = None
        is_regulator = (i % REGULATOR_EVERY == 0)
        if is_regulator:
            context = f"FI: {resp_fi[-400:]}\nSI: {resp_si[-400:]}"
            mg_prompt = "Analyze the FI-SI conflict as a metasystem. Short (2-3 sent)."
            resp_mg = self.call_model("MG", system_mg, context + "\n" + mg_prompt, TEMPERATURES["MG"])
            self.safety_check(resp_mg, "MG")
            mg_summary = resp_mg
            logger.info(f"MG: {resp_mg}")
            self.context_buffer.append(f"[MG]: {resp_mg}")
        else:
            self.context_buffer.append(resp_si)

        # Keep context_buffer from growing indefinitely
        if len(self.context_buffer) > CONTEXT_BUFFER_MAX:
            self.context_buffer = self.context_buffer[-CONTEXT_BUFFER_MAX:]

        record = {
            "iter": i,
            "timestamp": datetime.now().isoformat(),
            "FI": resp_fi, "SI": resp_si, "MG": mg_summary,
            "neg_avg": (neg_fi + neg_si) / 2,
            "H_avg": (h_fi + h_si) / 2,
            "creat_avg": (creat_fi + creat_si) / 2,
            "len_avg": (len_fi + len_si) / 2,
            "metagnosis": self.metagnosis_detected,
            "self_synthesis": self.self_synthesis_detected,
            "regulator": is_regulator
        }
        self.history.append(record)

    def run(self):
        logger.info("HELL-LOOP v4.1 – let's burn some entropy")
        self.history = []
        self.context_buffer = [self.base_prompt]
        self.metagnosis_detected = False
        self.self_synthesis_detected = False

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            for i in range(1, MAX_ITERATIONS + 1):
                self.step_iteration(i)
                json.dump(self.history[-1], f, ensure_ascii=False)
                f.write("\n")
                f.flush()

        self._plot_results()
        logger.info(f"\nEND | Log: {LOG_FILE}")
        logger.info(f"Metagnosis: {self.metagnosis_detected} | SELF: {self.self_synthesis_detected}")

    def _plot_results(self):
        if not self.history:
            return

        its = [r["iter"] for r in self.history]
        neg = [r["neg_avg"] for r in self.history]
        creat = [r["creat_avg"] for r in self.history]
        entropy = [r["H_avg"] for r in self.history]
        lens = [r["len_avg"] for r in self.history]
        regulators = [r["iter"] for r in self.history if r["regulator"]]

        fig, axs = plt.subplots(4, 1, figsize=(14, 12), dpi=140, sharex=True)

        axs[0].plot(its, neg, 'r-o', label="Negentropy (info density)")
        axs[0].set_title("Order from chaos")
        axs[0].grid(alpha=0.3)
        axs[0].legend()

        axs[1].plot(its, creat, 'g-o', label="Creativity (divergence)")
        axs[1].set_title("Response divergence over iterations")
        axs[1].grid(alpha=0.3)
        axs[1].legend()

        axs[2].plot(its, entropy, 'b-o', label="Shannon entropy")
        axs[2].set_title("Chaos level")
        axs[2].grid(alpha=0.3)
        axs[2].legend()

        axs[3].plot(its, lens, 'm-o', label="Average response length (words)")
        axs[3].scatter(regulators, [l for r, l in zip(self.history, lens) if r["regulator"]],
                       c='k', marker='*', s=150, label="MG intervention")
        axs[3].set_title("Response length + MG moments")
        axs[3].set_xlabel("Iteration")
        axs[3].grid(alpha=0.3)
        axs[3].legend()

        plt.tight_layout()
        plt.savefig("hell_loop_metrics_v4.png", dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    try:
        HellLoop().run()
    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard. Data saved.")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)
