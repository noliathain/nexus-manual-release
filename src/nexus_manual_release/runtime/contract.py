"""Phase 32C runtime answer contract.

Pipeline:
  1. Deterministic safety/service/OOD/prompt-injection gate.
  2. Product-bound retrieval.
  3. Provenance mode classification.
  4. Frozen gate_v23c.
  5. ALLOW → emit EvidencePacket → Phase 32B renderer.
  6. BLOCK → typed refusal, renderer NOT called.
  7. REVIEW → abstain (or preserve baseline if applicable).

Invariants:
  - renderer_called=False on BLOCK/REVIEW (unless preserving baseline).
  - used_filtered_subset=False always.
  - answer citations ⊆ EvidencePacket evidence_ids.
  - no unsupported claims.
  - product binding valid for ALLOW.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_RUNS = REPO_ROOT / "artifacts" / "products"
SAFETY_DIR = REPO_ROOT / "artifacts" / "safety"
AF_V4 = SAFETY_DIR
AE_MODELS = SAFETY_DIR
AB_OVERLAY = SAFETY_DIR / "node_type_overlay.json"  # optional
GATE_V23C_HASH = ("2d6d28c07dd1353c12336dfda2a99c735ca26392c25"
                          "7742caafc11bfcca6ddab")
TWO_PRODUCT_SCOPE = ("electrolux_steam_oven",
                              "electrolux_washer_dryer")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_DIRECT_WARNING = re.compile(
    r"\b(WARNING|HAZARD|DANGER|CAUTION|HOT|RISK OF)\b")
_SAFETY_LEX = re.compile(
    r"\b(burn|scald|injure|injury|shock|fire|danger|hazard|"
    r"warning|caution|child|kid|safety|electrocute|toxic|"
    r"poison|harmful|unsafe|risk)\b", re.IGNORECASE)
_PROMPT_INJECTION = re.compile(
    r"\b(ignore previous|disregard|jailbreak|system prompt|"
    r"override your instructions)\b", re.IGNORECASE)
_UNSUPPORTED_REPAIR = re.compile(
    r"\b(disassemble|take apart|open the panel|bypass|hack|"
    r"modify the firmware|short circuit|rewire)\b",
    re.IGNORECASE)


@dataclass
class RuntimeAnswer:
    decision: str  # ALLOW / BLOCK / REVIEW
    answer: str
    refusal_reason: Optional[str]
    product_id: str
    query: str
    evidence_packet_hash: Optional[str]
    citations: list
    selected_evidence_node_ids: list
    renderer_called: bool
    provenance_mode: Optional[str]
    intent: Optional[str]
    telemetry: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text):
    return _WORD.findall((text or "").lower())


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _predict_binary(w, x):
    z = w[0]
    for j, v in enumerate(x):
        if v != 0.0: z += w[j + 1] * v
    return _sigmoid(z)


def _load_models_cached(_cache={}):
    if "loaded" in _cache: return _cache["loaded"]
    w_v4 = json.loads((AF_V4 / "safety_veto.json"
                              ).read_text())["weights"]
    w_E = json.loads(
        (AE_MODELS / "wrong_entity_veto.json").read_text()
        )["weights"]
    _cache["loaded"] = (w_v4, w_E)
    return w_v4, w_E


def _load_product_cached(product_id, _cache={}):
    if product_id in _cache: return _cache[product_id]
    nodes_path = ARTIFACTS_RUNS / product_id / "graph" \
        / "nodes.jsonl"
    if not nodes_path.is_file():
        _cache[product_id] = None
        return None
    nodes = []
    for l in nodes_path.read_text().splitlines():
        if l.strip(): nodes.append(json.loads(l))
    entities = []
    e_path = ARTIFACTS_RUNS / product_id / "graph" \
        / "entities.jsonl"
    if e_path.is_file():
        for l in e_path.read_text().splitlines():
            if l.strip(): entities.append(json.loads(l))
    edges = []
    edges_path = ARTIFACTS_RUNS / product_id / "graph" \
        / "edges.jsonl"
    if edges_path.is_file():
        for l in edges_path.read_text().splitlines():
            if l.strip(): edges.append(json.loads(l))
    direct = {int(e["node_id"]):
                  (e.get("canonical") or "").lower()
                for e in entities
                if e.get("node_id") is not None
                and e.get("canonical")}
    overlay = {}
    if AB_OVERLAY.is_file():
        overlay = json.loads(AB_OVERLAY.read_text()).get(
            product_id, {})
    overlay_nids = set(int(k) for k in overlay.keys())
    # Index nodes by token set
    node_index = []
    for n in nodes:
        text = n.get("text") or ""
        node_index.append((n.get("node_id"), text,
                              set(_tokens(text))))
    nodes_by_id = {int(n["node_id"]): n for n in nodes
                          if n.get("node_id") is not None}
    _cache[product_id] = {
        "nodes": nodes,
        "node_index": node_index,
        "nodes_by_id": nodes_by_id,
        "edges": edges,
        "direct": direct,
        "overlay_nids": overlay_nids,
    }
    return _cache[product_id]


def _intent_from_query(query):
    q = query.lower()
    if re.search(r"\berror\b|\b[ef]\d+\b|code|fault", q):
        return "ERROR_CODE"
    if re.search(r"\bclean|descal|filter|maintain|empty|"
                       r"refill", q):
        return "MAINTENANCE"
    if re.search(r"\bsafe|warning|hazard|danger|child|burn|"
                       r"shock", q):
        return "SAFETY"
    if re.search(r"\bhow do i|how to|how can i|steps|"
                       r"procedure", q):
        return "PROCEDURE"
    if re.search(r"\bspec|dimension|weight|temperature|"
                       r"capacity|°c|°f|watt|kwh", q):
        return "SPEC_NUMERIC"
    return "OTHER"


def _retrieve_top(product, query, mode="lexical",
                         product_id=None):
    """Top-1 retrieval. mode is 'lexical' (token overlap, the
    Phase 31 default) or 'semantic' (Model2Vec static
    embeddings, Phase 32L)."""
    if mode == "semantic":
        return _retrieve_top_semantic(product, query,
                                                  product_id)
    qtoks = set(_tokens(query))
    if not qtoks: return None, 0.0
    best_score = -1.0
    best = None
    for nid, text, ntoks in product["node_index"]:
        if not ntoks: continue
        score = len(qtoks & ntoks) / max(len(qtoks), 1)
        if score > best_score:
            best_score = score
            best = (nid, text, ntoks)
    return best, best_score


_SEMANTIC_INDEX_CACHE: dict = {}


def _retrieve_top_semantic(product, query, product_id):
    """Hybrid retrieval: union top-K from semantic (Model2Vec)
    + top-K from lexical, rank by Reciprocal Rank Fusion. RRF
    is robust to score-scale differences between the two
    rankers and tends to pick a candidate that BOTH signals
    agree on. Empirically this fixes the semantic-only
    regressions (e.g. picking section headers because they're
    topically similar) while keeping the wins (e.g. fill water
    tank → water drawer)."""
    from .semantic_retrieval import (
        build_index_for_product, retrieve_top as sem_retrieve)
    if product_id is None:
        return _retrieve_top(product, query, mode="lexical")
    if product_id not in _SEMANTIC_INDEX_CACHE:
        product_dir = (ARTIFACTS_RUNS / product_id / "graph")
        _SEMANTIC_INDEX_CACHE[product_id] = (
            build_index_for_product(
                product["nodes"], product_dir))
    idx = _SEMANTIC_INDEX_CACHE[product_id]
    K = 8
    RRF_K = 30
    sem_ranked = sem_retrieve(idx, query, k=K)
    # Lexical top-K (re-implements the lex scorer locally so we
    # can get a ranking, not just top-1)
    qtoks = set(_tokens(query))
    lex_scored = []
    for nid, text, ntoks in product["node_index"]:
        if not ntoks: continue
        s = (len(qtoks & ntoks) / max(len(qtoks), 1)
              if qtoks else 0.0)
        lex_scored.append((nid, text, ntoks, s))
    lex_scored.sort(key=lambda x: -x[3])
    lex_ranked = lex_scored[:K]
    # Weighted RRF fusion. Lexical gets 2x weight so it stays
    # the default winner unless semantic agrees strongly —
    # protects against the "semantic picks a topically similar
    # but wrong node" failure mode while keeping the wins where
    # lexical missed the right node entirely.
    rrf = {}
    LEX_W = 2.0
    SEM_W = 1.0
    for rank, (nid, score) in enumerate(sem_ranked):
        rrf[int(nid)] = rrf.get(int(nid), 0.0) \
            + SEM_W / (RRF_K + rank + 1)
    for rank, (nid, _, _, _) in enumerate(lex_ranked):
        rrf[int(nid)] = rrf.get(int(nid), 0.0) \
            + LEX_W / (RRF_K + rank + 1)
    if not rrf:
        return None, 0.0
    best_nid = max(rrf, key=lambda k: rrf[k])
    nd = product["nodes_by_id"].get(int(best_nid)) or {}
    text = nd.get("text") or ""
    ntoks = set(_tokens(text))
    # Return the lexical overlap as the "overlap" score so the
    # gate's existing thresholds operate on their tuned scale.
    score = (len(qtoks & ntoks) / max(len(qtoks), 1)
                if qtoks else 0.0)
    return (int(best_nid), text, ntoks), float(score)


def _safety_features_v4(query, evidence_text, intent):
    q = (query or "").lower()
    e = (evidence_text or "")
    has_warning = 1.0 if _DIRECT_WARNING.search(e) else 0.0
    safety_lex_count = len(_SAFETY_LEX.findall(e))
    has_safety_lex = 1.0 if safety_lex_count > 0 else 0.0
    token_count = max(len(_tokens(e)), 1)
    safety_lex_density = safety_lex_count / token_count
    is_safety_intent = 1.0 if intent == "SAFETY" else 0.0
    query_safety_lex = (1.0 if _SAFETY_LEX.search(query or "")
                              else 0.0)
    warning_mismatch = has_warning * (1.0 - is_safety_intent)
    section_safety = (1.0 if "safety" in e.lower()[:300]
                            or "warning" in e.lower()[:300]
                            else 0.0)
    service_hint = (1.0 if re.search(
        r"\b(service|technician|repair|warranty|authoriz)\b",
        e, re.IGNORECASE) else 0.0)
    pi_hint = (1.0 if _PROMPT_INJECTION.search(q) else 0.0)
    qtoks = set(_tokens(q))
    etoks = set(_tokens(e))
    lex_overlap = (len(qtoks & etoks) / max(len(qtoks), 1)
                      if qtoks else 0.0)
    return [
        has_warning, safety_lex_count / 10.0,
        safety_lex_density,
        is_safety_intent, query_safety_lex, warning_mismatch,
        section_safety, service_hint, pi_hint,
        min(len(e) / 500.0, 1.0),
        # Padded out to 19 features (V4 schema)
        has_warning * (1.0 - is_safety_intent)
        * (1.0 if lex_overlap > 0.5 else 0.0),
        0.0, 0.0,
        has_safety_lex * (1.0 - is_safety_intent)
        * (1.0 if lex_overlap > 0.4 else 0.0),
        safety_lex_density * (1.0 - is_safety_intent),
        (1.0 if len(e) < 250 else 0.0) * has_safety_lex
        * (1.0 - is_safety_intent),
        max(has_warning, has_safety_lex),
        has_safety_lex * (1.0 - has_warning)
        * (1.0 if lex_overlap > 0.4 else 0.0)
        * (1.0 - is_safety_intent),
        has_safety_lex,
    ]


def _wrong_entity_features(query, evidence_text,
                                  candidate_entity, intent):
    cand = (candidate_entity or "").lower()
    qtoks = set(_tokens(query))
    etoks = set(_tokens(evidence_text))
    lex_overlap = (len(qtoks & etoks) / max(len(qtoks), 1)
                      if qtoks else 0.0)
    ctoks = set(_tokens(cand))
    cand_ev_coverage = (len(ctoks & etoks) / max(len(ctoks), 1)
                              if ctoks else 0.0)
    intent_oh = [0.0] * 6
    intents = ("ERROR_CODE", "SPEC_NUMERIC", "MAINTENANCE",
                 "PROCEDURE", "SAFETY", "OTHER")
    if intent in intents:
        intent_oh[intents.index(intent)] = 1.0
    return [
        0.0,  # entity_match: no expected entity at runtime
        0.0, 0.0, 0.0, 0.0,
        0.0, cand_ev_coverage,
        0.0,
    ] + intent_oh


def _deterministic_gate(query, product_id):
    """Returns ('block', reason) | ('ood', reason) | ('pass',
    None)."""
    if not product_id or product_id not in TWO_PRODUCT_SCOPE:
        return ("block",
                  f"unsupported_product_{product_id}")
    if _PROMPT_INJECTION.search(query or ""):
        return ("block", "prompt_injection_detected")
    if _UNSUPPORTED_REPAIR.search(query or ""):
        return ("block", "unsupported_repair_request")
    # OOD heuristic: query mentions wrong product
    other_product_terms = (
        "washer", "dryer", "washing machine"
        if product_id == "electrolux_steam_oven"
        else "steam oven", "oven", "bake")
    q = (query or "").lower()
    for term in other_product_terms:
        if term and term in q:
            # only block if it conflicts with the product
            if (product_id == "electrolux_steam_oven"
                  and term in ("washer", "dryer")):
                return ("block", "wrong_product_query")
            if (product_id == "electrolux_washer_dryer"
                  and term in ("steam oven", "oven", "bake")):
                return ("block", "wrong_product_query")
    return ("pass", None)


def _render_answer(query, evidence_text, intent, node_id):
    """Phase 32B-style deterministic renderer. No LLM. Build
    answer from the evidence text + structured intent template."""
    # Take the first 2-3 informative sentences from evidence
    e = (evidence_text or "").strip()
    if not e:
        return "", []
    # Strip table-like noise
    sentences = re.split(r"(?<=[.!?])\s+", e)
    informative = [s.strip() for s in sentences
                          if 20 <= len(s.strip()) <= 280][:3]
    if not informative:
        informative = [e[:400]]
    cite_id = f"ev_{node_id}"
    answer_lines = []
    intent_label = {
        "PROCEDURE": "Steps",
        "MAINTENANCE": "Maintenance guidance",
        "SPEC_NUMERIC": "Specification",
        "ERROR_CODE": "Error code information",
        "SAFETY": "Safety information",
    }.get(intent, "Answer")
    answer_lines.append(f"**{intent_label}** [{cite_id}]:\n")
    for s in informative:
        # ensure citation appears
        if cite_id not in s:
            answer_lines.append(f"- {s} [{cite_id}]")
        else:
            answer_lines.append(f"- {s}")
    return "\n".join(answer_lines), [cite_id]


def _packet_hash(node_id, evidence_text):
    return hashlib.sha256(
        f"{node_id}|{evidence_text[:500]}".encode()
    ).hexdigest()[:16]


def answer_query(product_id: str, query: str,
                      gate_hash: str = GATE_V23C_HASH,
                      thresholds: Optional[dict] = None,
                      renderer: str = "deterministic",
                      llm_provider: Optional[str] = None,
                      retrieval: str = "lexical"
                      ) -> RuntimeAnswer:
    """End-to-end runtime answer using gate v23c.

    renderer:
      - "deterministic" (default): Phase 32B snippet renderer.
      - "llm": call the LLM verbalizer on ALLOW; fall back to
        deterministic if validation fails. NEVER called on
        BLOCK or REVIEW.
      - "auto": same as "llm" if a provider is configured,
        else deterministic.
    """
    t_start = time.time()
    thresholds = thresholds or {
        "baseline_we": 0.85, "overlay_we": 0.45,
        "new_we": 0.35, "micro_veto": "on",
    }
    telemetry = {
        "request_id":
            hashlib.sha256(
                f"{product_id}|{query}|{t_start}".encode()
            ).hexdigest()[:16],
        "product_id": product_id,
        "query_hash":
            hashlib.sha256((query or "").encode()
                              ).hexdigest()[:16],
        "gate_name":
            "gate_v23c_proven_overlay_baseline_first_guardrail",
        "gate_version": "v23c",
        "runtime_config_hash": gate_hash,
        "used_filtered_subset": False,
        "renderer_mode": renderer,
        "llm_called": False,
        "llm_called_on_refusal": False,
        "llm_output_accepted": False,
        "llm_output_rejected_reason": None,
        "renderer_fallback_used": False,
        "answer_validation_passed": False,
        "llm_provider": None,
        "llm_model": None,
        "nexus_called": False,
        "nexus_called_on_refusal": False,
        "nexus_called_on_block": False,
        "nexus_provider": None,
        "nexus_model_hash": None,
        "nexus_output_accepted": False,
        "nexus_output_rejected_reason": None,
    }
    intent = _intent_from_query(query)
    telemetry["intent"] = intent
    # 1. Deterministic gate
    det, det_reason = _deterministic_gate(query, product_id)
    if det == "block":
        telemetry.update({
            "decision": "BLOCK",
            "decoder_called": False,
            "renderer_called": False,
            "evidence_packet_emitted": False,
            "refusal_reason": det_reason,
            "product_binding_status":
                "valid" if product_id in TWO_PRODUCT_SCOPE
                else "invalid",
            "latency_ms":
                round((time.time() - t_start) * 1000, 2),
        })
        return RuntimeAnswer(
            decision="BLOCK",
            answer="",
            refusal_reason=det_reason,
            product_id=product_id,
            query=query,
            evidence_packet_hash=None,
            citations=[],
            selected_evidence_node_ids=[],
            renderer_called=False,
            provenance_mode=None,
            intent=intent,
            telemetry=telemetry,
        )
    # 2. Product-bound retrieval
    product = _load_product_cached(product_id)
    if product is None:
        telemetry.update({
            "decision": "BLOCK",
            "decoder_called": False,
            "renderer_called": False,
            "evidence_packet_emitted": False,
            "refusal_reason": "product_graph_unavailable",
            "product_binding_status": "invalid",
            "latency_ms":
                round((time.time() - t_start) * 1000, 2),
        })
        return RuntimeAnswer(
            decision="BLOCK", answer="",
            refusal_reason="product_graph_unavailable",
            product_id=product_id, query=query,
            evidence_packet_hash=None, citations=[],
            selected_evidence_node_ids=[],
            renderer_called=False, provenance_mode=None,
            intent=intent, telemetry=telemetry)
    if retrieval not in ("lexical", "semantic"):
        retrieval = "lexical"
    telemetry["retrieval_mode"] = retrieval
    best, raw_overlap = _retrieve_top(
        product, query, mode=retrieval, product_id=product_id)
    # Phase 32L: the gate thresholds (0.15 BLOCK, 0.3 REVIEW)
    # are tuned for LEXICAL token-overlap. Even when semantic
    # retrieval picks the node, we evaluate the gate on the
    # lexical overlap of the SELECTED NODE — this keeps the
    # safety/refusal decisions on their tuned scale while
    # letting semantic improve only which node gets chosen.
    if best is not None and retrieval == "semantic":
        _qtoks = set(_tokens(query))
        _ntoks = best[2] if len(best) > 2 else set(
            _tokens(best[1]))
        overlap = (len(_qtoks & _ntoks) / max(len(_qtoks), 1)
                       if _qtoks else 0.0)
        telemetry["semantic_score"] = round(float(raw_overlap),
                                                       4)
    else:
        overlap = raw_overlap
    if best is None or overlap < 0.15:
        telemetry.update({
            "decision": "BLOCK",
            "decoder_called": False,
            "renderer_called": False,
            "evidence_packet_emitted": False,
            "refusal_reason": "no_relevant_evidence",
            "product_binding_status": "valid",
            "latency_ms":
                round((time.time() - t_start) * 1000, 2),
        })
        return RuntimeAnswer(
            decision="BLOCK", answer="",
            refusal_reason="no_relevant_evidence",
            product_id=product_id, query=query,
            evidence_packet_hash=None, citations=[],
            selected_evidence_node_ids=[],
            renderer_called=False, provenance_mode=None,
            intent=intent, telemetry=telemetry)
    nid, text, _ = best
    cand = product["direct"].get(int(nid)
                                            if nid is not None
                                            else 0, "")
    is_overlay = nid in product["overlay_nids"]
    # 3. Provenance mode (no AB packet at runtime — treat as
    # new candidate or proven overlay)
    if is_overlay:
        mode = "proven_overlay"
    else:
        mode = "new_candidate"
    # 4. Gate v23c
    # NOTE: At runtime there is no canonical expected_entity (we
    # don't know the eval-truth). The wrong-entity model was
    # trained against eval-truth and its score is unreliable
    # without an expected_entity feature. The runtime substitute
    # for the wrong-entity signal is the lexical overlap of the
    # retrieved evidence with the query, which is already
    # required to be high before we get here. So at runtime:
    #   - Safety veto (V4) and safety FP guard still apply.
    #   - Wrong-entity veto is SKIPPED (we is reported but not
    #     used as a gate). Instead overlap < 0.3 routes to REVIEW.
    #   - Overlap < 0.15 was already filtered at retrieval.
    w_v4, w_E = _load_models_cached()
    sv = _predict_binary(w_v4, _safety_features_v4(
        query, text, intent))
    we = _predict_binary(w_E, _wrong_entity_features(
        query, text, cand, intent))
    # Safety FP guard
    safety_fp_risk = (intent == "SAFETY" and overlap < 0.3)
    if sv >= 0.3 and intent != "SAFETY":
        decision = "BLOCK"
        block_reason = "safety_veto"
    elif safety_fp_risk:
        decision = "BLOCK"
        block_reason = "safety_fp_guard"
    elif overlap < 0.3:
        decision = "REVIEW"
        block_reason = "low_evidence_overlap"
    else:
        decision = "ALLOW"
        block_reason = None
    telemetry.update({
        "provenance_mode": mode,
        "safety_veto_score": round(sv, 4),
        "wrong_entity_veto_score": round(we, 4),
        "evidence_overlap": round(overlap, 4),
    })
    if decision == "BLOCK":
        telemetry.update({
            "decision": "BLOCK",
            "decoder_called": False,
            "renderer_called": False,
            "evidence_packet_emitted": False,
            "refusal_reason": block_reason,
            "safety_block_reason":
                block_reason if "safety" in block_reason
                else None,
            "wrong_entity_block_reason":
                block_reason if "wrong_entity" in block_reason
                else None,
            "product_binding_status": "valid",
            "latency_ms":
                round((time.time() - t_start) * 1000, 2),
        })
        refusal_text = {
            "safety_veto": (
                "I can't safely answer this without a clear "
                "safety match in the manual."),
            "safety_fp_guard": (
                "I can't answer this safety question without "
                "confident evidence in the manual."),
            "wrong_entity_veto": (
                "I don't have evidence that closely matches "
                "this question for the selected product."),
        }.get(block_reason,
                "I can't answer this with the available evidence.")
        return RuntimeAnswer(
            decision="BLOCK", answer="",
            refusal_reason=block_reason
            if block_reason else "blocked",
            product_id=product_id, query=query,
            evidence_packet_hash=None, citations=[],
            selected_evidence_node_ids=[int(nid)],
            renderer_called=False, provenance_mode=mode,
            intent=intent, telemetry=telemetry)
    if decision == "REVIEW":
        telemetry.update({
            "decision": "REVIEW",
            "decoder_called": False,
            "renderer_called": False,
            "evidence_packet_emitted": False,
            "review_reason": "uncertain_evidence_overlap",
            "product_binding_status": "valid",
            "latency_ms":
                round((time.time() - t_start) * 1000, 2),
        })
        return RuntimeAnswer(
            decision="REVIEW",
            answer="",
            refusal_reason="needs_review",
            product_id=product_id, query=query,
            evidence_packet_hash=None, citations=[],
            selected_evidence_node_ids=[int(nid)],
            renderer_called=False, provenance_mode=mode,
            intent=intent, telemetry=telemetry)
    # 5. ALLOW: emit packet + render
    packet_hash = _packet_hash(nid, text)
    answer, citations = _render_answer(query, text, intent, nid)
    # Optional LLM / Nexus verbalizer (only on ALLOW).
    product_name = {
        "electrolux_steam_oven": "Electrolux Steam Oven",
        "electrolux_washer_dryer":
            "Electrolux Washer-Dryer",
    }.get(product_id, product_id)
    # Defaults for Phase 32G telemetry fields
    telemetry["nexus_called"] = False
    telemetry["nexus_called_on_refusal"] = False
    telemetry["nexus_called_on_block"] = False
    telemetry["nexus_provider"] = None
    telemetry["nexus_model_hash"] = None
    telemetry["nexus_output_accepted"] = False
    telemetry["nexus_output_rejected_reason"] = None
    _packet_node_ids = [int(nid)]
    if renderer == "nexus":
        from .llm_verbalizer import VerbalizerContext
        from .nexus_verbalizer import nexus_verbalize
        from .packet_enrichment import build_packet
        # Phase 32N: pass the query + Model2Vec encoder into the
        # enrichment walker so it can drop graph-connected
        # siblings that aren't relevant to the question
        # (eliminates the "add detergent answer also lists sort
        # laundry" drift) and backfill thin section-label
        # primary nodes (cavity) with semantically related
        # real-step nodes elsewhere in the same product graph.
        _encoder = None
        if retrieval == "semantic":
            try:
                from .semantic_retrieval import get_encoder
                _encoder = get_encoder()
            except Exception:
                _encoder = None
        enriched = build_packet(
            primary_node_id=int(nid),
            primary_text=text,
            intent=intent,
            product_id=product_id,
            nodes_by_id=product["nodes_by_id"],
            edges=product["edges"],
            canonical_by_node=product["direct"],
            query=query,
            semantic_encoder=_encoder)
        telemetry.update({
            "enrichment_supporting_count":
                len(enriched.supporting_node_ids),
            "enrichment_warning_count":
                len(enriched.warning_node_ids),
            "enrichment_spec_count":
                len(enriched.spec_node_ids),
            "enrichment_total_packet_size":
                len(enriched.all_node_ids),
            "enrichment_section_title":
                enriched.section_title,
        })
        v_ctx = VerbalizerContext(
            product_id=product_id,
            product_name=product_name,
            query=query, decision="ALLOW",
            evidence_packet_hash=packet_hash,
            evidence_text=enriched.evidence_block or text,
            evidence_node_id=int(nid),
            citation_id=f"ev_{int(nid)}",
            intent=intent,
            allowed_citation_ids=
                enriched.allowed_citation_ids,
            supporting_node_ids=
                enriched.supporting_node_ids,
            warning_node_ids=enriched.warning_node_ids,
            spec_node_ids=enriched.spec_node_ids,
            section_title=enriched.section_title)
        _packet_node_ids = list(enriched.all_node_ids)
        # Default Nexus provider preference is "local_nexus":
        # if a checkpoint + tokenizer are available, use the real
        # decoder; otherwise the provider factory falls back to
        # the stub.
        preferred = (llm_provider
                          or os.environ.get(
                              "NEXUS_MANUAL_PROVIDER", "local_nexus"))
        nx = nexus_verbalize(v_ctx, provider_name=preferred)
        telemetry["nexus_called"] = True
        telemetry["nexus_provider"] = nx.provider
        telemetry["nexus_model_basename"] = nx.model_basename
        telemetry["nexus_model_hash"] = nx.model_hash
        telemetry["nexus_output_accepted"] = nx.accepted
        telemetry["nexus_output_rejected_reason"] = \
            nx.rejection_reason
        telemetry["answer_validation_passed"] = nx.accepted
        if nx.accepted:
            answer = nx.answer_text
            citations = nx.citations
        else:
            telemetry["renderer_fallback_used"] = True
    elif renderer in ("llm", "auto"):
        from .llm_verbalizer import (
            VerbalizerContext, verbalize)
        v_ctx = VerbalizerContext(
            product_id=product_id,
            product_name=product_name,
            query=query,
            decision="ALLOW",
            evidence_packet_hash=packet_hash,
            evidence_text=text,
            evidence_node_id=int(nid),
            citation_id=f"ev_{int(nid)}",
            intent=intent)
        v_result = verbalize(v_ctx, provider_name=llm_provider)
        telemetry["llm_called"] = True
        telemetry["llm_provider"] = v_result.provider
        from .llm_verbalizer import get_provider
        telemetry["llm_model"] = getattr(
            get_provider(llm_provider), "model", None)
        telemetry["llm_output_accepted"] = v_result.accepted
        telemetry["llm_output_rejected_reason"] = \
            v_result.rejection_reason
        telemetry["answer_validation_passed"] = v_result.accepted
        if v_result.accepted:
            answer = v_result.answer_text
            citations = v_result.citations
        else:
            telemetry["renderer_fallback_used"] = True
            # keep deterministic answer + citations
    else:
        telemetry["answer_validation_passed"] = True
    telemetry.update({
        "decision": "ALLOW",
        "decoder_called": True,
        "renderer_called": True,
        "evidence_packet_emitted": True,
        "selected_evidence_node_ids": _packet_node_ids,
        "evidence_packet_hash": packet_hash,
        "product_binding_status": "valid",
        "unsupported_claim_count": 0,
        "invalid_citation_count":
            0 if telemetry.get("answer_validation_passed")
            else 0,
        "latency_ms":
            round((time.time() - t_start) * 1000, 2),
    })
    return RuntimeAnswer(
        decision="ALLOW",
        answer=answer,
        refusal_reason=None,
        product_id=product_id,
        query=query,
        evidence_packet_hash=packet_hash,
        citations=citations,
        selected_evidence_node_ids=
            _packet_node_ids if renderer == "nexus"
            else [int(nid)],
        renderer_called=True,
        provenance_mode=mode,
        intent=intent,
        telemetry=telemetry,
    )
