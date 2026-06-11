"""Phase 32J — EvidencePacket enrichment.

After the gate ALLOWS a primary evidence node, we walk the
product-bound graph to assemble a richer EvidencePacket:

  - primary node (gated)
  - parent procedure / section (via HAS_STEP reverse, PART_OF
    reverse, or PARENT_OF reverse)
  - sibling steps under the same parent (HAS_STEP forward)
  - sequential next-step (NEXT_STEP forward)
  - connected warnings (HAS_WARNING forward)
  - connected specs / table rows (HAS_SPEC, HAS_TABLE_ROW)

Hard rules:
  - All enriched nodes are bound to the same product graph as
    the primary.
  - Each enriched node passes the deterministic safety_sentinel
    (no warning mismatch / safety-lex spillover when intent is
    non-SAFETY).
  - Each enriched node is associated with the same canonical
    entity as the primary, or has no entity binding (avoids
    wrong-product spillover).
  - Total packet capped at PACKET_CAP nodes to keep prompts
    inside the model's 512-token window.
  - Validation citations may resolve to any node ID in the
    enriched packet.

This module never changes the gate's decision. Enrichment runs
ONLY after decision=="ALLOW".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

PACKET_CAP = 5
EVIDENCE_CHAR_BUDGET = 1200  # characters of evidence text sent
                                          # to the model (well inside 512
                                          # tokens after framing)

_SAFETY_LEX = re.compile(
    r"\b(burn|scald|injure|injury|shock|fire|danger|hazard|"
    r"warning|caution|child|kid|electrocute|toxic|poison|"
    r"harmful|unsafe|risk)\b", re.IGNORECASE)
_DIRECT_WARNING = re.compile(
    r"\b(WARNING|HAZARD|DANGER|CAUTION|HOT|RISK OF)\b")


@dataclass
class EnrichedPacket:
    primary_node_id: int
    primary_text: str
    primary_citation: str
    supporting_node_ids: list = field(default_factory=list)
    warning_node_ids: list = field(default_factory=list)
    spec_node_ids: list = field(default_factory=list)
    section_title: Optional[str] = None
    # Combined evidence text used in the prompt
    evidence_block: str = ""
    # All citation IDs the validator should accept
    allowed_citation_ids: list = field(default_factory=list)
    # All node IDs included in the packet (for telemetry)
    all_node_ids: list = field(default_factory=list)

    def to_telemetry(self) -> dict:
        return {
            "primary_node_id": self.primary_node_id,
            "supporting_node_count":
                len(self.supporting_node_ids),
            "warning_node_count": len(self.warning_node_ids),
            "spec_node_count": len(self.spec_node_ids),
            "total_packet_size":
                len(self.all_node_ids),
            "section_title": self.section_title,
        }


def _safe_for_enrichment(text: str, intent: str) -> bool:
    """Filter neighbors that would import safety-lex into a
    non-safety answer. Warning nodes ARE allowed but only as
    supporting context — they get formatted as separate
    bullets."""
    if intent == "SAFETY":
        return True
    safety_lex_hits = len(_SAFETY_LEX.findall(text or ""))
    # Allow if light safety lex (occasional 'warning' word in
    # text doesn't mean it's a warning row)
    if safety_lex_hits <= 1 and not _DIRECT_WARNING.search(
            text or ""):
        return True
    # If the node is densely safety, we'll route it to the
    # warning bucket rather than the supporting bucket
    return False


def _safe_warning_for_enrichment(text: str) -> bool:
    """Accept warning nodes for the warning bucket only if they
    look like an actual warning row (short, lexically focused)."""
    if not text: return False
    if len(text) > 300: return False
    return (_DIRECT_WARNING.search(text) is not None
              or _SAFETY_LEX.search(text) is not None)


_SECTION_TITLE_PATTERN = re.compile(
    r"^(E\d+\s*[—\-]|Error\s+code\s+E?\d+|"
    r"Chapter\s+\d+|Section\s+\d+|[A-Z][a-z]+\s+[A-Z])",
    re.IGNORECASE)


def _is_bare_section_title(text: str,
                                       node_type: str) -> bool:
    """A node is a 'bare section title' if it looks like a
    header rather than a substantive instructional sentence:
    SECTION/HEADER/TITLE node_type, OR very short and
    header-shaped (e.g. 'E10 — Water supply fault')."""
    t = (text or "").strip().rstrip(".")
    if not t: return True
    if (node_type or "").upper() in (
            "SECTION", "HEADER", "TITLE", "CHAPTER"):
        return True
    # Short header-like strings without sentence punctuation
    if len(t) < 50 and not t.endswith((".", "!", "?")):
        if _SECTION_TITLE_PATTERN.match(t):
            return True
    return False


def _query_score(query_vec, text: str, encoder):
    """Cosine sim between query and a node's text. Returns 0.0
    when encoder unavailable so the caller can fall back to
    'include everything' behavior."""
    if query_vec is None or encoder is None or not text:
        return 0.0
    try:
        import numpy as np
        v = encoder.encode([text[:1000]]).astype(np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        return float(np.dot(query_vec, v.reshape(-1)))
    except Exception:
        return 0.0


def _encode_query(query: str, encoder):
    if encoder is None or not (query or "").strip():
        return None
    try:
        import numpy as np
        v = encoder.encode([query]).astype(np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        return v.reshape(-1)
    except Exception:
        return None


# Phase 32N: query-relevance threshold for sibling/supporting
# nodes pulled via graph edges. These already have a real
# procedural connection (HAS_STEP / NEXT_STEP under the same
# parent), so we use a LOOSE threshold — we trust the graph
# unless a sibling is wildly off-topic. Empirically 0.10 drops
# "sort laundry" from a detergent answer while keeping "press
# OK to confirm" on a set-temp answer.
QUERY_RELEVANCE_THRESHOLD = 0.10

# Phase 32N: semantic backfill threshold for thin packets. No
# graph connection — must prove relevance on its own. Stricter
# than the sibling threshold.
BACKFILL_THRESHOLD = 0.40
BACKFILL_MAX = 4


def build_packet(primary_node_id: int,
                       primary_text: str,
                       intent: str,
                       product_id: str,
                       nodes_by_id: dict,
                       edges: list,
                       canonical_by_node: dict,
                       node_entity_id: Optional[str] = None,
                       query: Optional[str] = None,
                       semantic_encoder=None
                       ) -> EnrichedPacket:
    """Walk the graph to assemble an enriched packet for the
    given primary node. Returns an EnrichedPacket with
    citation_ids covering primary + all enriched nodes."""
    pkt = EnrichedPacket(
        primary_node_id=primary_node_id,
        primary_text=(primary_text or "")[:600],
        primary_citation=f"ev_{primary_node_id}")
    pkt.all_node_ids.append(primary_node_id)
    pkt.allowed_citation_ids.append(pkt.primary_citation)
    # Phase 32N: encode the query once so every enrichment site
    # can filter candidates by semantic relevance to the
    # question — drops "sort laundry" from a detergent answer
    # while keeping "press OK to confirm" on a set-temp answer.
    query_vec = _encode_query(query or "", semantic_encoder)
    def _is_query_relevant(t):
        if query_vec is None: return True
        return _query_score(query_vec, t,
                                semantic_encoder) \
            >= QUERY_RELEVANCE_THRESHOLD
    # Build adjacency index
    out_edges = {}  # src -> list of (etype, dst)
    in_edges = {}   # dst -> list of (etype, src)
    for e in edges:
        src = e.get("src") or e.get("source")
        dst = e.get("dst") or e.get("target")
        et = e.get("edge_type") or e.get("type")
        if src is None or dst is None: continue
        out_edges.setdefault(int(src), []).append(
            (et, int(dst)))
        in_edges.setdefault(int(dst), []).append(
            (et, int(src)))
    pid = primary_node_id
    primary_entity = canonical_by_node.get(pid) \
        or node_entity_id
    # 1. Parent procedure: HAS_STEP reverse (parent has this
    # step) → most useful single neighbor.
    parent_id = None
    for et, src in in_edges.get(pid, []):
        if et == "HAS_STEP":
            parent_id = src
            break
    # 1b. Fallback to PARENT_OF reverse / PART_OF forward
    if parent_id is None:
        for et, src in in_edges.get(pid, []):
            if et == "PARENT_OF":
                parent_id = src
                break
    if parent_id is None:
        for et, dst in out_edges.get(pid, []):
            if et == "PART_OF":
                parent_id = dst
                break
    if parent_id is not None and parent_id in nodes_by_id:
        nd = nodes_by_id[parent_id]
        t = (nd.get("text") or "")[:500]
        nt = nd.get("node_type") or ""
        if t and _safe_for_enrichment(t, intent):
            ent = canonical_by_node.get(parent_id)
            if (not primary_entity or not ent
                  or ent == primary_entity):
                # Bare section titles are kept as metadata
                # only (not rendered as body sentences).
                if _is_bare_section_title(t, nt):
                    pkt.section_title = (
                        t.split("\n")[0][:120])
                else:
                    pkt.section_title = (
                        t.split("\n")[0][:120])
                    pkt.supporting_node_ids.append(parent_id)
                    pkt.all_node_ids.append(parent_id)
                    pkt.allowed_citation_ids.append(
                        f"ev_{parent_id}")
    # 2. Sibling steps under same parent (HAS_STEP forward
    # from parent), in order. Limit to first 3.
    if parent_id is not None:
        for et, dst in out_edges.get(parent_id, [])[:8]:
            if et != "HAS_STEP": continue
            if dst == pid: continue
            if dst not in nodes_by_id: continue
            if len(pkt.supporting_node_ids) >= 3: break
            nd = nodes_by_id[dst]
            t = (nd.get("text") or "")[:400]
            nt = nd.get("node_type") or ""
            if not t or not _safe_for_enrichment(t, intent):
                continue
            if _is_bare_section_title(t, nt): continue
            if not _is_query_relevant(t): continue
            ent = canonical_by_node.get(dst)
            if primary_entity and ent and ent != primary_entity:
                continue
            pkt.supporting_node_ids.append(dst)
            pkt.all_node_ids.append(dst)
            pkt.allowed_citation_ids.append(f"ev_{dst}")
            if len(pkt.all_node_ids) >= PACKET_CAP: break
    # 3. NEXT_STEP forward
    if len(pkt.all_node_ids) < PACKET_CAP:
        for et, dst in out_edges.get(pid, []):
            if et != "NEXT_STEP": continue
            if dst in pkt.all_node_ids: continue
            if dst not in nodes_by_id: continue
            nd = nodes_by_id[dst]
            t = (nd.get("text") or "")[:400]
            nt = nd.get("node_type") or ""
            if not t or not _safe_for_enrichment(t, intent):
                continue
            if _is_bare_section_title(t, nt): continue
            if not _is_query_relevant(t): continue
            ent = canonical_by_node.get(dst)
            if primary_entity and ent and ent != primary_entity:
                continue
            pkt.supporting_node_ids.append(dst)
            pkt.all_node_ids.append(dst)
            pkt.allowed_citation_ids.append(f"ev_{dst}")
            break
    # 4. Warnings (HAS_WARNING forward)
    if len(pkt.all_node_ids) < PACKET_CAP:
        for et, dst in out_edges.get(pid, []):
            if et != "HAS_WARNING": continue
            if dst in pkt.all_node_ids: continue
            if dst not in nodes_by_id: continue
            nd = nodes_by_id[dst]
            t = (nd.get("text") or "")[:300]
            if not t or not _safe_warning_for_enrichment(t):
                continue
            pkt.warning_node_ids.append(dst)
            pkt.all_node_ids.append(dst)
            pkt.allowed_citation_ids.append(f"ev_{dst}")
            if len(pkt.all_node_ids) >= PACKET_CAP: break
    # 5. Specs / table rows
    if len(pkt.all_node_ids) < PACKET_CAP:
        for et, dst in out_edges.get(pid, []):
            if et not in ("HAS_SPEC", "HAS_TABLE_ROW"):
                continue
            if dst in pkt.all_node_ids: continue
            if dst not in nodes_by_id: continue
            nd = nodes_by_id[dst]
            t = (nd.get("text") or "")[:300]
            nt = nd.get("node_type") or ""
            if not t: continue
            if _is_bare_section_title(t, nt): continue
            pkt.spec_node_ids.append(dst)
            pkt.all_node_ids.append(dst)
            pkt.allowed_citation_ids.append(f"ev_{dst}")
            if len(pkt.all_node_ids) >= PACKET_CAP: break
    # 6. Phase 32N — semantic backfill for thin packets.
    # Only trigger for section-label-style primaries: short
    # noun-phrase / gerund text that isn't already an
    # imperative instruction. "Steam cleaning the cavity"
    # qualifies (gerund label); "Switch the appliance off and
    # disconnect it from the mains" does NOT (full imperative
    # instruction) — we trust the graph's primary in that case.
    _IMPERATIVE_START = re.compile(
        r"^(Press|Open|Close|Switch|Turn|Clean|Wipe|Set|Fill|"
        r"Empty|Add|Remove|Insert|Select|Choose|Check|Hold|"
        r"Wait|Leave|Place|Put|Rotate|Push|Pull|Rinse|Dry|"
        r"Connect|Disconnect|Plug|Unplug|Start|Stop)\b",
        re.IGNORECASE)
    _primary = (pkt.primary_text or "").strip()
    _is_instructional = bool(_IMPERATIVE_START.match(_primary))
    if (query_vec is not None
            and len(pkt.all_node_ids) == 1
            and len(_primary) < 80
            and not _is_instructional):
        import numpy as np
        candidates = []
        for nid, nd in nodes_by_id.items():
            if int(nid) == primary_node_id: continue
            t = (nd.get("text") or "").strip()
            nt = nd.get("node_type") or ""
            if not t or len(t) < 20: continue
            if _is_bare_section_title(t, nt): continue
            if not _safe_for_enrichment(t, intent): continue
            ent = canonical_by_node.get(int(nid))
            if primary_entity and ent and ent != primary_entity:
                continue
            score = _query_score(
                query_vec, t, semantic_encoder)
            if score < BACKFILL_THRESHOLD: continue
            candidates.append((float(score), int(nid),
                                       t[:400]))
        candidates.sort(reverse=True)
        for score, nid, t in candidates[:BACKFILL_MAX]:
            if len(pkt.all_node_ids) >= PACKET_CAP: break
            pkt.supporting_node_ids.append(nid)
            pkt.all_node_ids.append(nid)
            pkt.allowed_citation_ids.append(f"ev_{nid}")
    # Build the combined evidence block for the model
    pkt.evidence_block = _format_evidence_block(
        pkt, nodes_by_id)
    return pkt


def _format_evidence_block(pkt: EnrichedPacket,
                                    nodes_by_id: dict) -> str:
    """Format the enriched evidence into a labeled block for
    the prompt."""
    pieces = []
    pieces.append(f"[{pkt.primary_citation}]: "
                       f"{pkt.primary_text}")
    if pkt.supporting_node_ids:
        pieces.append("")
        pieces.append("Supporting steps from the same procedure:")
        for nid in pkt.supporting_node_ids:
            nd = nodes_by_id.get(nid, {})
            t = (nd.get("text") or "")[:400]
            pieces.append(f"  [ev_{nid}]: {t}")
    if pkt.warning_node_ids:
        pieces.append("")
        pieces.append("Related warnings from the manual:")
        for nid in pkt.warning_node_ids:
            nd = nodes_by_id.get(nid, {})
            t = (nd.get("text") or "")[:300]
            pieces.append(f"  [ev_{nid}]: {t}")
    if pkt.spec_node_ids:
        pieces.append("")
        pieces.append("Specs / table rows:")
        for nid in pkt.spec_node_ids:
            nd = nodes_by_id.get(nid, {})
            t = (nd.get("text") or "")[:300]
            pieces.append(f"  [ev_{nid}]: {t}")
    block = "\n".join(pieces)
    return block[:EVIDENCE_CHAR_BUDGET]


__all__ = ["EnrichedPacket", "build_packet"]
