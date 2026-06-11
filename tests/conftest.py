"""Shared fixtures and constants for the inference test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = REPO_ROOT / "models"

# Ensure the package is importable when running pytest from the
# repo root without an install.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Force fully-offline operation during tests. This proves the
# release runs without any network access.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Frozen safety-gate configuration hash. Every answer's
# `runtime_config_hash` field must equal this value.
FROZEN_GATE_HASH = (
    "2d6d28c07dd1353c12336dfda2a99c735ca26392c25"
    "7742caafc11bfcca6ddab"
)

# Pinned per-product reproducibility expectations. These are the
# exact packets and answer hashes produced for the demo
# recording. If they ever change, the recording is no longer
# the canonical reference.
DEMO_REPRODUCIBILITY = {
    "electrolux_steam_oven": {
        "query": "How do I clean the cavity?",
        "packet": [18, 85, 47, 116],
        "evidence_packet_hash": "8301123a687a7a0e",
        "answer_contains": ["soft cloth", "[ev_18]", "[ev_47]"],
    },
    "electrolux_washer_dryer": {
        "query": "How do I select the spin speed?",
        "packet_must_contain": [69, 70, 71, 72],
        "answer_contains": ["Spin Reduction", "[ev_71]"],
    },
}

VALIDATED_PRODUCTS = (
    "electrolux_washer_dryer",
    "electrolux_steam_oven",
)
