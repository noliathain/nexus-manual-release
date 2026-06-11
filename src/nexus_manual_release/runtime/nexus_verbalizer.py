"""Phase 32G — local Nexus decoder verbalizer.

Same input contract as the Phase 32F LLM verbalizer
(VerbalizerContext), same post-validation rules (see
.llm_verbalizer.validate_output). The runtime calls this only
after the frozen gate has emitted an EvidencePacket.

Hard invariants:
  - Nexus is NEVER called on BLOCK or REVIEW decisions.
  - Nexus output must cite only EvidencePacket evidence IDs.
  - Validation failure → fall back to deterministic renderer
    silently. Nexus answer is not shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm_verbalizer import VerbalizerContext, validate_output
from .nexus_provider import (
    StubNexusProvider, LocalNexusProvider, get_nexus_provider,
)
from .nexus_config import (
    load_config, model_hash, model_path_basename,
)


NEXUS_PROMPT_TEMPLATE = (
    "You are answering a product manual question.\n"
    "Use only the evidence below.\n"
    "Do not add facts that are not in the evidence.\n"
    "Cite facts with the citation IDs exactly as shown.\n"
    "If evidence is insufficient, say so.\n\n"
    "Product: {product_name}\n"
    "Question: {query}\n\n"
    "Evidence:\n{citation_id}: {evidence_snippet}\n\n"
    "Answer:\n"
)


@dataclass
class NexusVerbalizerResult:
    answer_text: str
    citations: list
    used_evidence_ids: list
    provider: str
    model_basename: Optional[str]
    model_hash: Optional[str]
    accepted: bool
    rejection_reason: Optional[str]
    limitations: list = field(default_factory=list)


def build_prompt(ctx: VerbalizerContext) -> str:
    """The exact prompt the Nexus decoder would receive. Surfaces
    for auditability."""
    return NEXUS_PROMPT_TEMPLATE.format(
        product_name=ctx.product_name,
        query=ctx.query,
        citation_id=ctx.citation_id,
        evidence_snippet=ctx.evidence_text[:1500])


def nexus_verbalize(ctx: VerbalizerContext,
                              provider_name: Optional[str] = None
                              ) -> NexusVerbalizerResult:
    """Call Nexus on ALLOW; validate output. Returns a result
    with accepted=False if anything is wrong; caller falls back
    to the deterministic renderer in that case."""
    if ctx.decision != "ALLOW":
        return NexusVerbalizerResult(
            answer_text="", citations=[],
            used_evidence_ids=[],
            provider=provider_name or "stub_nexus",
            model_basename=None, model_hash=None,
            accepted=False,
            rejection_reason="not_allowed_decision")
    provider = get_nexus_provider(provider_name)
    try:
        raw = provider.generate(ctx)
    except (NotImplementedError, RuntimeError) as e:
        return NexusVerbalizerResult(
            answer_text="", citations=[],
            used_evidence_ids=[],
            provider=getattr(provider, "name", "stub_nexus"),
            model_basename=getattr(provider, "model", None),
            model_hash=model_hash(),
            accepted=False,
            rejection_reason=f"provider_unavailable:{e}")
    accepted, reason, citations, used = validate_output(raw, ctx)
    return NexusVerbalizerResult(
        answer_text=raw if accepted else "",
        citations=citations,
        used_evidence_ids=used,
        provider=getattr(provider, "name", "stub_nexus"),
        model_basename=getattr(provider, "model", None),
        model_hash=model_hash(),
        accepted=accepted,
        rejection_reason=reason)


__all__ = [
    "NexusVerbalizerResult", "nexus_verbalize",
    "build_prompt", "NEXUS_PROMPT_TEMPLATE",
]
