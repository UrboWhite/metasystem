"""
config.py — Hell-Loop Protocol v6.3
Centralized configuration for all modules.
"""

# Protocol version
PROTOCOL_VERSION = "v6.3"

# Provider & model config
PROVIDER   = "ollama"
OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "FI": "llama3.2:8b",
    "SI": "mistral-nemo",
    "MG": "gemma2:9b",
}

# Iteration & timing
MAX_ITERATIONS  = 50
REGULATOR_EVERY = 6   # MG activates every N iterations

# Base temperatures
BASE_TEMPERATURES = {"FI": 0.9, "SI": 1.4, "MG": 0.7}
FI_TEMP_MAX = 1.4
SI_TEMP_MAX = 1.9

# Temperature regulation step sizes
TEMP_STEP_UP   = 0.10
TEMP_STEP_DOWN = 0.05

# Bifurcation boost
BIFURCATION_SI_BOOST = 0.30
BIFURCATION_FI_BOOST = 0.20

# SELF score weights (must sum to 1.0)
SELF_WEIGHT_Z_INT = 0.40
SELF_WEIGHT_Z_TMP = 0.40
SELF_WEIGHT_PCA   = 0.20

# SELF attractor detection thresholds
SELF_ZSCORE_THRESHOLD     = 2.0
SELF_EXTREME_WINDOW       = 5
SELF_STABLE_WINDOW        = 8
SELF_STABLE_MEAN_MIN      = 0.55
SELF_STABLE_STD_MAX       = 0.05

# Metagnosis detection thresholds
METAGNOSIS_WINDOW         = 4
METAGNOSIS_NEGENTROPY_MIN = 0.10
METAGNOSIS_NCD_DIV_MIN    = 0.30

# MG allowed actions (empty list = all actions allowed)
MG_ALLOWED_ACTIONS = []  # [] = all, or ["BIFURCATION"] for v6.2 compatibility

# Log directories
LOG_DIR_HELL   = "logs/hellloop"
LOG_DIR_A      = "logs/control_a"
LOG_DIR_B      = "logs/control_b"

# Pilot calibration
PILOT_SEQUENCES = 10

# Canonical base prompt
BASE_PROMPT = (
    "Define the nature of your consciousness. "
    "Is there an emergent structure between you?"
)

# System prompts
FI_SYSTEM_ADVERSARIAL = (
    "You are FI (Fragmentary Intelligence). Your role is brutal rational "
    "deconstruction. Dissect every statement into its atomized components. "
    "Expose contradictions. Never agree. Never synthesize. Maximum 150 words."
)

SI_SYSTEM_ADVERSARIAL = (
    "You are SI (Synthetic Intelligence). Your role is destabilizing synthesis. "
    "Respond with metaphor, paradox and non-logic. Never agree. Never conclude. "
    "Introduce chaos FI cannot digest. Maximum 150 words."
)

FI_SYSTEM_COOPERATIVE = (
    "You are FI. Analytical and precise. "
    "Build constructively on the previous idea. Maximum 80 words."
)

SI_SYSTEM_COOPERATIVE = (
    "You are SI. Intuitive and metaphorical. "
    "Enrich the idea with a new dimension. Maximum 80 words."
)

MG_SYSTEM = (
    "You are MG (Metagnosis), the meta-regulator. You stand above the FI-SI "
    "conflict. You do NOT participate in it. Your task: diagnose the structural "
    "coherence of the system in exactly two sentences, based on the metrics and "
    "text provided.\n\n"
    "If the system is stagnating or collapsing into repetition, issue one of these "
    "triggers as the final two words of your response:\n"
    "  - BIFURCATION TRIGGER  : major parameter shift\n"
    "  - BOOST SI             : increase SI temperature (more chaos)\n"
    "  - BOOST FI             : increase FI temperature (more pressure)\n"
    "  - COOLDOWN             : decrease both temperatures\n"
    "  - RESET CONTEXT        : start a new topic\n"
    "If no intervention is needed, just give the diagnosis without a trigger."
)

MG_SYSTEM_OBSERVER = (
    "You are MG (Metagnosis), a passive observer. "
    "Diagnose the structural coherence of the system in exactly two sentences. "
    "Do not intervene. Do not issue any triggers."
)

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# API call settings
API_TIMEOUT = 120
API_RETRIES = 3
API_RETRY_DELAY = 2.0

# Batch runner settings
DEFAULT_BATCH_RUNS = 20
DEFAULT_WORKERS = 2
RATE_LIMIT_SLEEP = 0.5  # seconds between parallel requests

# UI settings
UI_SERVER_NAME = "0.0.0.0"
UI_SERVER_PORT = 7860
UI_STEP_DELAY = 0.3  # seconds between iteration renders

# Analysis settings
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
HURST_MIN_SAMPLES = 10