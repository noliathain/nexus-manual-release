"""Blessed configuration for the manual-RAG Nexus decoder.

Two flavours:

* :func:`nexus_manual_rag_8m_512_spec` — returns a pure-Python
  :class:`NexusModelSpec` (no PyTorch import). Used by ``mgr model-info`` and
  any non-training tooling.

* :func:`create_nexus_manual_rag_8m_512_config` — instantiates the real
  :class:`modeling.nexus.NexusConfig` (lazy import; needs the ``train``
  extra).

Keep this file in sync with ``configs/model/nexus_manual_rag_8m_512.yaml`` —
the unit test ``tests/python/unit/test_nexus_config_yaml_match.py`` (added in
the model phase) asserts the two never drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from manual_graph_rag.common.errors import MgrError
from manual_graph_rag.schemas.model import NexusModelSpec

if TYPE_CHECKING:  # pragma: no cover
    from manual_graph_rag.modeling.nexus import NexusConfig

NEXUS_8M_512_CHECKPOINT = (
    "/teamspace/lightning_storage/nexus-data/"
    "nexus_tinystories_checkpoints-4096-8m-2_2-fineweb-climbmix/best_model_512.pt"
)


def nexus_manual_rag_8m_512_spec() -> NexusModelSpec:
    """Return the blessed model spec as a Pydantic object.

    Importing this function does *not* require PyTorch.
    """
    # Phase 5 discovery: the blessed checkpoint actually ships with
    # intermediate_size=384 and num_hidden_layers=14 (the Phase 0 design
    # note said 512 / 10, which the weights then refused to load into).
    # The checkpoint is authoritative — we changed the spec to match.
    return NexusModelSpec(
        name="nexus_manual_rag_8m_512",
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=14,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        vocab_size=4096,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        dropout=0.0,
        hidden_act="silu",
        tie_word_embeddings=True,
        flash_attn=True,
        inference_rope_scaling=False,
        bos_token_id=1,
        eos_token_id=2,
        checkpoint_path=NEXUS_8M_512_CHECKPOINT,
    )


def create_nexus_manual_rag_8m_512_config() -> "NexusConfig":
    """Instantiate the canonical ``NexusConfig``.

    This is a lazy import: callers that don't need PyTorch never pay for it.
    Raises :class:`MgrError` with a clear message when the ``train`` extra
    isn't installed.
    """
    try:
        from manual_graph_rag.modeling.nexus import NexusConfig
    except ImportError as e:  # pragma: no cover - exercised in fresh envs
        raise MgrError(
            "The Nexus model requires PyTorch + transformers. Install them with:\n"
            "  uv sync --extra train\n"
            f"(underlying error: {e})"
        ) from e

    spec = nexus_manual_rag_8m_512_spec()
    return NexusConfig(
        dropout=spec.dropout,
        bos_token_id=spec.bos_token_id,
        eos_token_id=spec.eos_token_id,
        hidden_act=spec.hidden_act,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        max_position_embeddings=spec.max_position_embeddings,
        num_attention_heads=spec.num_attention_heads,
        num_hidden_layers=spec.num_hidden_layers,
        num_key_value_heads=spec.num_key_value_heads,
        vocab_size=spec.vocab_size,
        rms_norm_eps=spec.rms_norm_eps,
        rope_theta=spec.rope_theta,
        inference_rope_scaling=spec.inference_rope_scaling,
        flash_attn=spec.flash_attn,
        tie_word_embeddings=spec.tie_word_embeddings,
    )


def spec_summary(spec: NexusModelSpec | None = None) -> dict[str, Any]:
    """Compact summary used by ``mgr model-info``.

    Numbers are approximate; the canonical parameter count comes from
    instantiating the model.
    """
    s = spec or nexus_manual_rag_8m_512_spec()
    # Rough param estimate for sanity-printing.
    embed = s.vocab_size * s.hidden_size
    per_layer_attn = 4 * s.hidden_size * s.hidden_size
    per_layer_mlp = 3 * s.hidden_size * s.intermediate_size
    per_layer = per_layer_attn + per_layer_mlp
    approx_params = embed + s.num_hidden_layers * per_layer
    if not s.tie_word_embeddings:
        approx_params += embed
    return {
        "name": s.name,
        "approx_parameters": approx_params,
        "hidden_size": s.hidden_size,
        "intermediate_size": s.intermediate_size,
        "num_hidden_layers": s.num_hidden_layers,
        "num_attention_heads": s.num_attention_heads,
        "num_key_value_heads": s.num_key_value_heads,
        "max_position_embeddings": s.max_position_embeddings,
        "vocab_size": s.vocab_size,
        "rope_theta": s.rope_theta,
        "rms_norm_eps": s.rms_norm_eps,
        "tie_word_embeddings": s.tie_word_embeddings,
        "flash_attn": s.flash_attn,
        "checkpoint_path": s.checkpoint_path,
    }
