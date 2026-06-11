# Inference Pipeline

This document traces a query through the runtime end-to-end.

![Inference pipeline](images/inference_pipeline.svg)

---

## Refusal classes

The gate is verified to refuse on the following classes. The decoder is
never invoked on any of them.

![Refusal classes](images/refusal_classes.svg)

On every BLOCK answer, the trace fields `decoder_called`,
`renderer_called`, and `nexus_called` are all `False`. The
`latency_ms` field is under 1 ms.

## Step-by-step

### Step 1 — Intent classification

A regex-based deterministic classifier assigns one of: `MAINTENANCE`,
`PROCEDURE`, `SPEC_NUMERIC`, `ERROR_CODE`, `SAFETY`, `OTHER`.

The intent shapes the conversational opener pool used downstream and is
recorded in the trace.

### Step 2 — Safety gate (pattern layer)

Three regex matches run in series:

1. `_PROMPT_INJECTION` — common jailbreak phrasings.
2. `_UNSUPPORTED_REPAIR` — disassemble, bypass, modify-firmware, etc.
3. `_SAFETY_LEX` — only used downstream as a feature, never blocks
   alone.

If pattern 1 or 2 fires, the gate returns BLOCK with the typed reason
and the decoder is never invoked.

### Step 3 — Hybrid retrieval

The retriever ranks candidate nodes two ways and fuses the results.

**Lexical ranking** — token-set Jaccard overlap of the query against
each node's text. Fast, exact, no model.

**Semantic ranking** — cosine similarity over static embedding vectors.
The encoder is a vocabulary table where each token has a pre-computed
vector; the query and node embeddings are token-vector averages.
Sub-millisecond on CPU.

**Reciprocal Rank Fusion** combines the two rankings. Lexical is given
2× weight to preserve the proven default; semantic wins only when it
agrees strongly with itself or fills a gap lexical missed.

### Step 4 — Veto models

Two learned binary classifiers run on the retrieved primary node text:

- **Safety veto** — flags non-safety queries whose evidence is
  safety-relevant.
- **Wrong-entity veto** — flags evidence whose canonical entity doesn't
  match the bound product.

Both are small logistic-regression models with hand-engineered features.
The weights ship as JSON in `artifacts/safety/`.

### Step 5 — Gate decision

The gate combines all signals into a single decision:

| condition | decision |
|---|---|
| any pattern-layer match | BLOCK + typed reason |
| `overlap < 0.15` | BLOCK `no_relevant_evidence` |
| `safety_veto ≥ 0.3` and `intent != SAFETY` | BLOCK `safety_veto` |
| `intent == SAFETY` and `overlap < 0.3` | BLOCK `safety_fp_guard` |
| `overlap < 0.3` | REVIEW `low_evidence_overlap` |
| otherwise | ALLOW |

### Step 6 — Packet enrichment

![Packet enrichment](images/packet_enrichment.svg)

On ALLOW, the enricher walks the product graph to assemble a richer
packet. See the [architecture document](architecture.md#5-packet-enricher)
for the full priority order.

### Step 7 — Verbalization

The verbalizer formats the packet for the local decoder. The decoder
generates a candidate completion, but the verbalizer never publishes
the raw model output — instead it stitches the model's tone with
deterministic citation anchors from the packet.

Multi-step packets render as numbered Markdown lists in node-id
ascending order (which approximates document order in the source
manual):

```text
The manual describes the procedure as follows [ev_67]:

1. Sort laundry by colour, fabric type, and wash symbol [ev_63].
2. Open the door and place laundry loosely in the drum [ev_65].
3. Open the detergent drawer [ev_67].
4. Add main-wash detergent ... [ev_67].
```

Single-step packets render as inline prose:

```text
The manual covers this maintenance step like this: Clean the
water-inlet filter at the tap connector if needed [ev_45].
```

Short fragmentary primary nodes (section labels) use a light opener to
avoid grammatical clash:

```text
From the manual: Steam cleaning the cavity [ev_18].
```

### Step 8 — Citation validation

![Citation validator](images/citation_validator.svg)

The validator checks the proposed answer against the assembled packet.
If any rule fails, the verbalizer falls back to the deterministic
snippet renderer and the trace records
`renderer_fallback_used: true`.

## What's in the trace

Every published answer carries a structured trace. The most important
fields:

```json
{
  "decision": "ALLOW",
  "refusal_reason": null,
  "intent": "MAINTENANCE",
  "provenance_mode": "new_candidate",
  "renderer_called": true,
  "decoder_called": true,
  "nexus_called": true,
  "renderer_mode": "nexus",
  "retrieval_mode": "semantic",
  "evidence_overlap": 0.3333,
  "semantic_score": 0.3333,
  "safety_veto_score": 0.1694,
  "wrong_entity_veto_score": 0.7657,
  "selected_evidence_node_ids": [18, 85, 47, 116],
  "enrichment_total_packet_size": 4,
  "evidence_packet_hash": "8301123a687a7a0e",
  "runtime_config_hash": "2d6d28c07dd1353c12336dfda2a99c735ca26392c257742caafc11bfcca6ddab",
  "nexus_model_hash": "202679c4532dc224",
  "answer_validation_passed": true,
  "latency_ms": 3249.95
}
```

The pairing `(runtime_config_hash, nexus_model_hash, query,
retrieval_mode)` uniquely determines the answer. Reproducibility is
enforced by the trace, not asserted in prose.
