"""
config.py - Load and validate environment variables.

Rules:
- GEMINI_API_KEY is required; raise clearly naming the variable on missing.
- All others have defaults. Never print a key value.
"""

import os
from dotenv import load_dotenv

# Load .env file (do not override existing env vars)
load_dotenv(override=False)

# --- Required variables ---
REQUIRED = ["GEMINI_API_KEY"]

# --- Optional variables with defaults ---
DEFAULTS: dict[str, str] = {
    "GEMINI_MODEL": "gemini-2.5-flash",
    "EMBEDDING_MODEL": "text-embedding-004",
    "OPENFDA_API_KEY": "",          # empty string = not provided
    "PDF_PATH": "./data/Drug-Interactions--What-You-Should-Know-low-res.pdf",
    "CHROMA_PATH": "./chroma_db",
    "MAX_AGENT_STEPS": "5",
}


def _load() -> dict[str, str]:
    """Load all config variables. Raises ValueError naming any missing required var."""
    config: dict[str, str] = {}

    # Validate required
    for key in REQUIRED:
        value = os.getenv(key)
        if not value:
            raise ValueError(
                f"Required environment variable '{key}' is missing or empty. "
                "Please set it in your .env file."
            )
        config[key] = value  # stored but never printed

    # Load optionals with defaults
    for key, default in DEFAULTS.items():
        val = os.getenv(key, default)
        if key == "GEMINI_MODEL" and val in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite", ""):
            val = "gemini-3.5-flash-lite"
        config[key] = val

    return config


# Module-level config dict — import this in other modules
cfg: dict[str, str] = _load()

# Convenience accessors (never expose the raw key values in logs)
GEMINI_API_KEY: str = cfg["GEMINI_API_KEY"]
GEMINI_MODEL: str = cfg["GEMINI_MODEL"]
EMBEDDING_MODEL: str = cfg["EMBEDDING_MODEL"]
OPENFDA_API_KEY: str = cfg["OPENFDA_API_KEY"]
PDF_PATH: str = cfg["PDF_PATH"]
CHROMA_PATH: str = cfg["CHROMA_PATH"]
MAX_AGENT_STEPS: int = int(cfg["MAX_AGENT_STEPS"])
