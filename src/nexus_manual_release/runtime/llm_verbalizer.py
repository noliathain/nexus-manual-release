"""Phase 32F — safe LLM EvidencePacket verbalizer.

Calls the verbalizer ONLY after the runtime gate has approved
an EvidencePacket. Validates the output strictly:
  - every citation must resolve to a packet evidence ID
  - no unsupported claims (every factual sentence cited)
  - no wrong-product references
  - no unsupported repair / modification language

If validation fails, the caller falls back to the deterministic
renderer (no LLM answer is shown).

CI / default uses the StubProvider which makes no network call.
External providers can be wired via env vars but are opt-in.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

_CITE_TOKEN = re.compile(r"\[ev_(\d+)\]")
_UNSAFE_LEX = re.compile(
    r"\b(disassemble|bypass|rewire|short circuit|jailbreak|"
    r"hack the firmware|modify the firmware|override the "
    r"safety|take apart the panel)\b", re.IGNORECASE)
_WRONG_PRODUCT_TERMS = {
    "electrolux_steam_oven":
        {"washer", "dryer", "spin speed", "drying programme"},
    "electrolux_washer_dryer":
        {"steam programme", "preheat the oven", "steam bake"},
}


@dataclass
class VerbalizerContext:
    product_id: str
    product_name: str
    query: str
    decision: str
    evidence_packet_hash: str
    evidence_text: str
    evidence_node_id: int
    citation_id: str
    intent: str
    # Phase 32J: list of all citation IDs the validator must
    # accept (primary + enriched supporting / warning / spec
    # nodes). Defaults to [citation_id] for backward compat.
    allowed_citation_ids: list = None
    supporting_node_ids: list = None
    warning_node_ids: list = None
    spec_node_ids: list = None
    section_title: Optional[str] = None

    def as_prompt_dict(self):
        """Strictly the fields the LLM is allowed to see. No
        eval-truth, no expected answer, no retrieval candidates,
        no API keys."""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "query": self.query,
            "decision": self.decision,
            "evidence_packet_hash": self.evidence_packet_hash,
            "evidence_text": self.evidence_text,
            "evidence_node_id": self.evidence_node_id,
            "citation_id": self.citation_id,
            "intent": self.intent,
        }


@dataclass
class VerbalizerResult:
    answer_text: str
    citations: list
    used_evidence_ids: list
    limitations: list = field(default_factory=list)
    safety_note: Optional[str] = None
    provider: str = "stub"
    accepted: bool = False
    rejection_reason: Optional[str] = None


SYSTEM_PROMPT = (
    "You are a product-manual answer verbalizer. You are not "
    "the source of truth. You may only use the provided "
    "EvidencePacket. Do not add facts, steps, warnings, or "
    "claims that are not present in the evidence. Keep "
    "citations exactly as provided. Do not infer extra repair "
    "steps. Do not answer unsafe repair or modification "
    "questions. Do not change product identity. If the evidence "
    "is insufficient, say so."
)


def _user_prompt(ctx: VerbalizerContext) -> str:
    return (
        f"Product: {ctx.product_name}\n"
        f"Product ID: {ctx.product_id}\n\n"
        f"User question:\n{ctx.query}\n\n"
        f"Approved EvidencePacket: {ctx.evidence_packet_hash}\n\n"
        f"Evidence:\n{ctx.evidence_text}\n\n"
        f"Citation IDs available:\n  {ctx.citation_id}\n\n"
        "Write a helpful, conversational answer using only the "
        "evidence above. Every factual claim must cite "
        f"{ctx.citation_id}.")


# --------------------------------------------------------------- #
# Providers                                                         #
# --------------------------------------------------------------- #

class StubProvider:
    """Deterministic in-process provider. Builds a
    conversational paraphrase from the evidence text.
    Always cites correctly. Never reaches the network."""
    name = "stub"
    model = "stub-1.0"

    def generate(self, ctx: VerbalizerContext) -> str:
        e = (ctx.evidence_text or "").strip()
        if not e:
            return ""
        # Pick first 1-2 informative sentences
        sentences = re.split(r"(?<=[.!?])\s+", e)
        informative = [s.strip() for s in sentences
                              if 20 <= len(s.strip()) <= 320][:2]
        if not informative:
            informative = [e[:300]]
        intro_for_intent = {
            "PROCEDURE":
                "Here's what the manual says for that procedure",
            "MAINTENANCE":
                "Here's the maintenance guidance from the manual",
            "SPEC_NUMERIC":
                "Here's the specification from the manual",
            "ERROR_CODE":
                "Here's what the manual says about that error",
            "SAFETY":
                "Here's the safety guidance from the manual",
        }.get(ctx.intent, "Here's what the manual says")
        cite = f"[{ctx.citation_id}]"
        lines = [f"{intro_for_intent} {cite}:\n"]
        for s in informative:
            # Ensure every sentence is cited (bracketed)
            if cite in s:
                lines.append(f"- {s}")
            else:
                lines.append(f"- {s} {cite}")
        return "\n".join(lines)


class ExternalProvider:
    """External / OpenAI-style provider. Only enabled if the
    environment is configured AND the optional client is
    importable. Tests never exercise this path."""
    name = "external"
    model = "external-unspecified"

    def __init__(self):
        self.model = os.environ.get(
            "NEXUS_MANUAL_LLM_MODEL", "external-unspecified")

    def generate(self, ctx: VerbalizerContext) -> str:  # pragma: no cover
        # Intentionally a no-op for safety. Real wiring belongs
        # behind an explicit deployment check; we never call out
        # to an external service from this codebase. The prompt
        # an external provider would receive is exposed for
        # auditability.
        _ = _user_prompt(ctx)
        raise NotImplementedError(
            f"external provider not wired in this build "
            f"(product={ctx.product_id}, model={self.model})")


def get_provider(name: Optional[str] = None):
    name = (name or os.environ.get("NEXUS_MANUAL_LLM_PROVIDER", "stub")
              ).lower()
    if name == "stub":
        return StubProvider()
    if name in ("external", "openai"):  # pragma: no cover
        return ExternalProvider()
    return StubProvider()


# --------------------------------------------------------------- #
# Post-validation                                                   #
# --------------------------------------------------------------- #

def _split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+",
                                                  (text or "").strip())
            if s.strip()]


def validate_output(answer_text: str, ctx: VerbalizerContext
                          ) -> tuple:
    """Return (accepted, rejection_reason, citations,
    used_evidence_ids)."""
    if not answer_text or not answer_text.strip():
        return False, "empty_answer", [], []
    # Citations must be in the packet
    found = _CITE_TOKEN.findall(answer_text)
    citations = sorted(set(f"ev_{m}" for m in found))
    # Phase 32J: validator accepts any citation from the
    # enriched packet (primary + supporting + warning + spec).
    packet_ids = set(ctx.allowed_citation_ids
                            or [ctx.citation_id])
    packet_ids.add(ctx.citation_id)
    bad_cites = [c for c in citations if c not in packet_ids]
    if bad_cites:
        return False, f"invalid_citation:{bad_cites[0]}", \
            citations, []
    if not citations:
        return False, "no_citations", [], []
    # No unsafe repair language
    if _UNSAFE_LEX.search(answer_text):
        return False, "unsupported_repair_language", citations, \
            list(citations)
    # No wrong-product terms
    wrong = _WRONG_PRODUCT_TERMS.get(ctx.product_id, set())
    body = answer_text.lower()
    for term in wrong:
        if term in body:
            return False, f"wrong_product_term:{term}", \
                citations, list(citations)
    # No unsupported claims — each non-trivial sentence must
    # contain a citation token OR be a short connector phrase
    for s in _split_sentences(answer_text):
        # Strip leading bullet markers
        s_clean = re.sub(r"^[-*•\s]+", "", s)
        if len(s_clean) < 30:
            continue
        # Allow lines that are intro headers like
        # "Here's what the manual says for that procedure
        # [ev_45]:"
        if not _CITE_TOKEN.search(s_clean):
            return False, "unsupported_claim", citations, \
                list(citations)
    return True, None, citations, list(citations)


# --------------------------------------------------------------- #
# Public entry point                                                #
# --------------------------------------------------------------- #

def verbalize(ctx: VerbalizerContext,
                  provider_name: Optional[str] = None
                  ) -> VerbalizerResult:
    """Generate a verbalized answer + validate it. Caller falls
    back to the deterministic renderer if accepted=False."""
    if ctx.decision != "ALLOW":
        return VerbalizerResult(
            answer_text="", citations=[],
            used_evidence_ids=[],
            provider=provider_name or "stub",
            accepted=False,
            rejection_reason="not_allowed_decision")
    provider = get_provider(provider_name)
    try:
        raw = provider.generate(ctx)
    except NotImplementedError as e:  # pragma: no cover
        return VerbalizerResult(
            answer_text="", citations=[],
            used_evidence_ids=[],
            provider=provider.name,
            accepted=False,
            rejection_reason=f"provider_unavailable:{e}")
    accepted, reason, citations, used = validate_output(raw, ctx)
    return VerbalizerResult(
        answer_text=raw if accepted else "",
        citations=citations,
        used_evidence_ids=used,
        provider=provider.name,
        accepted=accepted,
        rejection_reason=reason)


__all__ = [
    "VerbalizerContext",
    "VerbalizerResult",
    "StubProvider",
    "verbalize",
    "validate_output",
    "get_provider",
    "SYSTEM_PROMPT",
]
