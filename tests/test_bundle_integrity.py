"""Bundle integrity — every asset the runtime needs is committed
to the repository at a discoverable path."""
from __future__ import annotations

import hashlib
import json

import pytest

from tests.conftest import (
    ARTIFACTS_DIR, FROZEN_GATE_HASH, MODELS_DIR, REPO_ROOT,
    VALIDATED_PRODUCTS,
)


# ---- Source code ---------------------------------------------------- #

def test_package_directory_exists():
    pkg = REPO_ROOT / "src" / "nexus_manual_release"
    assert pkg.is_dir(), pkg

def test_runtime_module_files_present():
    rt = REPO_ROOT / "src" / "nexus_manual_release" / "runtime"
    for f in ("contract.py", "llm_verbalizer.py",
                "nexus_config.py", "nexus_provider.py",
                "nexus_verbalizer.py", "packet_enrichment.py",
                "semantic_retrieval.py"):
        assert (rt / f).is_file(), f

def test_modeling_module_files_present():
    md = REPO_ROOT / "src" / "nexus_manual_release" / "modeling"
    for f in ("nexus.py", "nexus_config.py"):
        assert (md / f).is_file(), f


# ---- Models --------------------------------------------------------- #

def test_local_decoder_checkpoint_bundled():
    p = MODELS_DIR / "local_decoder.pt"
    assert p.is_file(), p
    assert p.stat().st_size > 10_000_000, "decoder seems too small"

def test_decoder_tokenizer_bundled():
    tok = MODELS_DIR / "tokenizer"
    assert tok.is_dir(), tok
    for f in ("tokenizer.json", "vocab.json", "merges.txt"):
        assert (tok / f).is_file(), f

def test_static_embedding_encoder_bundled():
    enc = MODELS_DIR / "encoder"
    assert enc.is_dir(), enc
    for f in ("config.json", "model.safetensors", "modules.json",
                "tokenizer.json", "tokenizer_config.json"):
        assert (enc / f).is_file(), f
    # Encoder weights should be ~30 MB
    weights = enc / "model.safetensors"
    assert weights.stat().st_size > 20_000_000, \
        "encoder weights look truncated"


# ---- Per-product artifacts ----------------------------------------- #

@pytest.mark.parametrize("product", VALIDATED_PRODUCTS)
def test_product_graph_files_present(product):
    g = ARTIFACTS_DIR / "products" / product / "graph"
    for f in ("nodes.jsonl", "edges.jsonl", "entities.jsonl"):
        assert (g / f).is_file(), f
    # Files must have at least one record
    for f in ("nodes.jsonl", "edges.jsonl", "entities.jsonl"):
        lines = [l for l in (g / f).read_text().splitlines()
                    if l.strip()]
        assert len(lines) > 0, f

@pytest.mark.parametrize("product", VALIDATED_PRODUCTS)
def test_semantic_index_bundled(product):
    idx = ARTIFACTS_DIR / "products" / product / "graph" \
            / "semantic_index.npz"
    assert idx.is_file(), idx
    import numpy as np
    data = np.load(idx, allow_pickle=False)
    assert "node_ids" in data.files
    assert "vectors" in data.files
    assert data["vectors"].ndim == 2
    assert data["vectors"].shape[0] == len(data["node_ids"])


# ---- Safety + gate -------------------------------------------------- #

def test_safety_weights_bundled():
    for f in ("safety_veto.json", "wrong_entity_veto.json"):
        p = ARTIFACTS_DIR / "safety" / f
        assert p.is_file(), p
        loaded = json.loads(p.read_text())
        assert "weights" in loaded, f
        assert isinstance(loaded["weights"], list)

def test_frozen_safety_gate_hash_matches():
    p = REPO_ROOT / "configs" / "safety_gate.yaml"
    assert p.is_file(), p
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == FROZEN_GATE_HASH, (
        f"safety gate config has been modified.\n"
        f"  expected: {FROZEN_GATE_HASH}\n"
        f"  got:      {h}")


# ---- Documentation -------------------------------------------------- #

def test_readme_present():
    p = REPO_ROOT / "README.md"
    assert p.is_file()
    text = p.read_text()
    assert "Manual Graph-RAG" in text

def test_docs_present():
    for f in ("architecture.md", "pipeline.md", "offline.md",
                "demo_script.md"):
        assert (REPO_ROOT / "docs" / f).is_file(), f

def test_license_present():
    p = REPO_ROOT / "LICENSE"
    assert p.is_file()
    assert "Apache License" in p.read_text()
