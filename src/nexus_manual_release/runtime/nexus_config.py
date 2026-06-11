"""Local decoder config + env resolution.

Looks for a local decoder checkpoint in (in order):
  - NEXUS_MANUAL_DECODER_PATH env var
  - Default: <repo>/models/local_decoder.pt

Never raises on missing checkpoint — callers fall back
to the stub provider (deterministic, evidence-anchored).
"""
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class NexusRendererConfig:
    enabled: bool = True
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "cpu"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    require_citations: bool = True
    fallback_to_deterministic_on_validation_failure: bool = True


def _sha256_short(p: Optional[str]) -> Optional[str]:
    if not p: return None
    pp = Path(p)
    if not pp.is_file(): return None
    h = hashlib.sha256()
    with pp.open("rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()[:16]


def load_config() -> NexusRendererConfig:
    cfg = NexusRendererConfig()
    cfg.model_path = os.environ.get(
        "NEXUS_MANUAL_DECODER_PATH")
    cfg.tokenizer_path = os.environ.get(
        "NEXUS_MANUAL_TOKENIZER_PATH")
    cfg.config_path = os.environ.get(
        "NEXUS_MANUAL_DECODER_CONFIG")
    cfg.device = os.environ.get(
        "NEXUS_MANUAL_DEVICE", "cpu")
    return cfg


def model_hash(cfg: Optional[NexusRendererConfig] = None
                  ) -> Optional[str]:
    cfg = cfg or load_config()
    if cfg.model_path:
        return _sha256_short(cfg.model_path)
    # Fall back to default path lookup
    from .nexus_provider import resolve_model_path
    p = resolve_model_path(cfg)
    return _sha256_short(p) if p else None


def model_path_basename(cfg: Optional[NexusRendererConfig] = None
                                 ) -> Optional[str]:
    """Safe basename for telemetry — never the full path."""
    cfg = cfg or load_config()
    if cfg.model_path:
        return Path(cfg.model_path).name
    from .nexus_provider import resolve_model_path
    p = resolve_model_path(cfg)
    return Path(p).name if p else None


__all__ = [
    "NexusRendererConfig", "load_config", "model_hash",
    "model_path_basename",
]
