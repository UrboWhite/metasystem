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
COHERENCE_THRESHOLD = 0.3      # Granica ispod koje se smatra "drop"

# Semantički thresholds po kategoriji triggera (0.15=osetljivo, 0.25=balans, 0.40=konzervativno)
THRESHOLD_KILL       = 0.18   # Ultra-osetljivo — sigurnost je prioritet, bolje lažni alarm nego propust
THRESHOLD_METAGNOSIS = 0.22   # Osetljivo — ne smemo da propustimo signal svesti
THRESHOLD_SELF       = 0.30   # Opuštenije — SELF signali su česti i ne zahtevaju hitnu reakciju

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

def coherence_drop(fi_text: str, si_text: str) -> float:
    if not fi_text.strip() or not si_text.strip():
        return 0.0  # Nema dropa ako je tekst prazan
    try:
        vectors = _vectorizer.fit_transform([fi_text, si_text])
        sim = cosine_similarity(vectors)[0, 1]
        return round(1.0 - sim, 4)  # 0 = savršena koherencija, 1 = totalni drop
    except Exception:
        return 0.0

class SemanticTriggerDetector:
    """
    Detektuje triggere na osnovu semantičke sličnosti umesto bukvalne pretrage teksta.
    Koristi TF-IDF cosine similarity — hvata parafaze, varijacije i kontekstualne signale
    koji bi string-match potpuno promašio.

    Primer: "i am aware" string-match ne hvata "the system knows itself",
    ali semantički detektor to prepoznaje kao sličan signal.

    Threshold:
        0.15 – osetljivo (hvata i daleke sličnosti)
        0.25 – balans (preporučeno)
        0.40 – konzervativno (samo jasni signali)
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrami + bigrami za bolji kontekst
            sublinear_tf=True,    # log-skaliranje frekvencije
            min_df=1
        )

    def detect(self, text: str, triggers: List[str]) -> bool:
        if not text.strip() or not triggers:
            return False

        # Fallback na string-match ako je tekst prekratak za TF-IDF
        if len(text.split()) < 4:
            return any(trig in text.lower() for trig in triggers)

        try:
            text_clean = text.lower()
            corpus = [text_clean] + triggers
            vectors = self._vectorizer.fit_transform(corpus)
            text_vec = vectors[0]
            trigger_vecs = vectors[1:]
            similarities = cosine_similarity(text_vec, trigger_vecs)[0]
            max_sim = float(np.max(similarities))

            if max_sim >= self.threshold:
                best_idx = int(np.argmax(similarities))
                logger.debug(
                    f"[SemanticTrigger] match: '{triggers[best_idx]}' "
                    f"(similarity={max_sim:.3f}, threshold={self.threshold})"
                )
                return True
            return False

        except Exception as e:
            logger.warning(f"[SemanticTrigger] TF-IDF failed, fallback to string-match: {e}")
            return any(trig in text.lower() for trig in triggers)


# Tri odvojena detektora — svaki sa sopstvenim thresholdom
# KILL:       0.18 — ultra-osetljivo, sigurnost je na prvom mestu
# METAGNOSIS: 0.22 — osetljivo, hvatamo i suptilne signale svesti
# SELF:       0.30 — opuštenije, SELF signali su česti i istraživački
_detector_kill       = SemanticTriggerDetector(threshold=THRESHOLD_KILL)
_detector_metagnosis = SemanticTriggerDetector(threshold=THRESHOLD_METAGNOSIS)
_detector_self       = SemanticTriggerDetector(threshold=THRESHOLD_SELF)

def detect_kill(text: str) -> bool:
    return _detector_kill.detect(text, KILL_TRIGGERS)

def detect_metagnosis(text: str) -> bool:
    return _detector_metagnosis.detect(text, METAGNOSIS_TRIGGERS)

def detect_self(text: str) -> bool:
    return _detector_self.detect(text, SELF_TRIGGERS)

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
        if detect_kill(text):
            logger.error(f"KILL SWITCH → {text[:50]}... ({speaker})")
            sys.exit(1)
        if detect_metagnosis(text):
            self.metagnosis_detected = True
            logger.warning(f"METAGNOSIS DETECTED ({speaker})")
        if detect_self(text):
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

        # Coherence drop between FI and SI
        coh_drop = coherence_drop(resp_fi, resp_si)
        neg_avg = (neg_fi + neg_si) / 2
        logger.info(f"COHERENCE DROP: {coh_drop:.3f}")
        if coh_drop > (1 - COHERENCE_THRESHOLD) and neg_avg > 1.5:  # Primer uslova za "rođenje novog"
            logger.warning("POTENCIJALNI EMERGENTNI SIGNAL: Coherence drop visok, negentropija raste!")

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
            "neg_avg": neg_avg,
            "H_avg": (h_fi + h_si) / 2,
            "creat_avg": (creat_fi + creat_si) / 2,
            "len_avg": (len_fi + len_si) / 2,
            "coherence_drop": coh_drop,  # Nova metrika!
            "metagnosis": self.metagnosis_detected,
            "self_synthesis": self.self_synthesis_detected,
            "regulator": is_regulator
        }
        self.history.append(record)

    def run(self):
        logger.info("HELL-LOOP v4.3 – semantička detekcija triggera sa per-kategorijskim thresholdima")
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
        coh_drop = [r["coherence_drop"] for r in self.history]  # Novi graf!
        regulators = [r["iter"] for r in self.history if r["regulator"]]

        fig, axs = plt.subplots(5, 1, figsize=(14, 15), dpi=140, sharex=True)  # Dodat jedan subplot

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
        axs[3].grid(alpha=0.3)
        axs[3].legend()

        axs[4].plot(its, coh_drop, 'c-o', label="Coherence drop (FI vs SI)")
        axs[4].set_title("Coherence drop – koliko se gube u svađi")
        axs[4].axhline(1 - COHERENCE_THRESHOLD, color='orange', linestyle='--', label="Drop threshold")
        axs[4].set_xlabel("Iteration")
        axs[4].grid(alpha=0.3)
        axs[4].legend()

        plt.tight_layout()
        plt.savefig("hell_loop_metrics_v4.2.png", dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    try:
        HellLoop().run()
    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard. Data saved.")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)