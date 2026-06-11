"""Phase 32L — semantic retrieval over the product graph.

Uses Model2Vec static embeddings (Apache-2.0):
  - Encoder: minishlab/potion-base-8M by default.
    8M params, 256-dim vectors, ~30MB FP32, no transformer
    forward pass — just vocab lookup + token-vector averaging.
    Same encoder runs on desktop AND on the embedded target,
    so retrieval quality is identical across environments.
  - Per-product index: float32 [num_nodes × 256]. For the WD
    graph (190 nodes) that's ~195KB. Cached to disk so loading
    is instant on subsequent processes.

Architecture: build-time we embed every graph node's text once
and persist alongside the graph. At query time we embed the
query once (~0.5ms on CPU) and do a cosine-similarity scan
(<1ms for 190 nodes).

The encoder is loaded lazily on first semantic query so the
deterministic / lexical paths pay nothing.

Phase 32L.1: silence the HuggingFace Hub progress bars + network
calls. Once the encoder weights are in the local cache (~/.cache/
huggingface/hub) we never make another network call. Setting the
env vars BEFORE any model2vec / huggingface_hub import is the only
way they take effect, so we set them at module load.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Mute hub progress bars + transformers chatter unconditionally
# (these are presentation-layer concerns, not safety).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# If the user has already set HF_HUB_OFFLINE=1 honor it. We DO
# NOT force offline here — the very first run on a fresh machine
# needs network to download the encoder weights. Subsequent runs
# read from cache and make no network call regardless.
# httpx + urllib3 are the actual transports underneath
# huggingface_hub; silence those too so the revision-check ping
# never prints in customer demos.
for _logger in ("huggingface_hub", "model2vec",
                      "transformers", "httpx", "httpcore",
                      "urllib3", "filelock"):
    try:
        logging.getLogger(_logger).setLevel(logging.ERROR)
    except Exception:
        pass

# Default Model2Vec encoder. Same model used for the offline
# node index and for the on-device query encode, so retrieval
# behavior matches between desktop and the embedded target.
DEFAULT_ENCODER = "minishlab/potion-base-8M"
INDEX_VERSION = 1


@dataclass
class SemanticIndex:
    encoder_name: str
    node_ids: list   # ordered list of node ids
    vectors: object  # np.ndarray [N, dim], L2-normalized
    dim: int


def _normalize(v):
    import numpy as np
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


def _index_cache_path(product_dir: Path,
                              encoder_name: str) -> Path:
    safe = encoder_name.replace("/", "_")
    return product_dir / f"semantic_index_{safe}.npz"


def _model_cache():
    """Module-level cache so repeated calls in the same process
    don't reload the encoder."""
    if not hasattr(_model_cache, "_m"):
        _model_cache._m = {}
    return _model_cache._m


def get_encoder(encoder_name: str = DEFAULT_ENCODER):
    cache = _model_cache()
    if encoder_name in cache:
        return cache[encoder_name]
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(encoder_name)
    cache[encoder_name] = m
    return m


def build_index_for_product(
        nodes: list,
        product_dir: Path,
        encoder_name: str = DEFAULT_ENCODER,
        force_rebuild: bool = False) -> SemanticIndex:
    """Build (or load from cache) the per-product semantic
    index. nodes is the full nodes.jsonl payload list."""
    import numpy as np
    cache_path = _index_cache_path(product_dir, encoder_name)
    if cache_path.is_file() and not force_rebuild:
        try:
            data = np.load(cache_path, allow_pickle=False)
            if int(data.get("version", -1)) == INDEX_VERSION:
                return SemanticIndex(
                    encoder_name=encoder_name,
                    node_ids=data["node_ids"].tolist(),
                    vectors=data["vectors"],
                    dim=int(data["vectors"].shape[-1]))
        except Exception:
            pass  # rebuild
    encoder = get_encoder(encoder_name)
    texts, ids = [], []
    for n in nodes:
        nid = n.get("node_id")
        text = (n.get("text") or "").strip()
        if nid is None or not text:
            continue
        ids.append(int(nid))
        texts.append(text[:1000])  # cap per-node text for stable
                                            # vectors
    vectors = encoder.encode(texts) if texts \
        else np.zeros((0, 256), dtype=np.float32)
    vectors = vectors.astype(np.float32, copy=False)
    vectors = _normalize(vectors)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path,
                 version=np.int32(INDEX_VERSION),
                 node_ids=np.array(ids, dtype=np.int64),
                 vectors=vectors)
    return SemanticIndex(
        encoder_name=encoder_name, node_ids=ids,
        vectors=vectors, dim=int(vectors.shape[-1]))


def retrieve_top(index: SemanticIndex,
                       query: str,
                       k: int = 1) -> list:
    """Encode query and return top-k (node_id, score) pairs."""
    import numpy as np
    if not query.strip():
        return []
    encoder = get_encoder(index.encoder_name)
    q = encoder.encode([query]).astype(np.float32)
    q = _normalize(q)[0]
    scores = index.vectors @ q  # cosine since both normalized
    if len(scores) == 0:
        return []
    k = min(k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(index.node_ids[i]), float(scores[i]))
                for i in top_idx]


__all__ = [
    "SemanticIndex", "DEFAULT_ENCODER",
    "build_index_for_product", "retrieve_top",
    "get_encoder",
]
