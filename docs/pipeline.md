# Inference Pipeline

This document traces a query through the runtime end-to-end.

## End-to-end flow

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CLI as CLI
    participant IC as Intent Classifier
    participant SG as Safety Gate
    participant RT as Retriever
    participant VE as Veto Models
    participant PE as Packet Enricher
    participant VB as Verbalizer
    participant VL as Citation Validator

    U->>CLI: ask("How do I clean the cavity?")
    CLI->>IC: classify(query)
    IC-->>CLI: intent = MAINTENANCE
    CLI->>SG: pattern_check(query)
    SG-->>CLI: gate_pass

    Note over SG: prompt-injection,<br/>unsupported-repair,<br/>OOD patterns checked

    CLI->>RT: retrieve_top(query, product, mode=semantic)
    RT-->>CLI: primary_node = 18, score = 0.33

    CLI->>VE: score(query, primary_text)
    VE-->>CLI: safety = 0.17, wrong_entity = 0.77

    Note over CLI: decision = ALLOW

    CLI->>PE: enrich(primary, query, encoder)
    PE-->>CLI: packet = [18, 85, 47, 116]

    CLI->>VB: verbalize(packet, intent, query)
    VB-->>CLI: candidate_answer

    CLI->>VL: validate(candidate_answer, packet)
    VL-->>CLI: accepted

    CLI-->>U: cited answer + trace
```

## Step-by-step

### Step 1 — Intent classification

A regex-based deterministic classifier assigns one of:
`MAINTENANCE`, `PROCEDURE`, `SPEC_NUMERIC`, `ERROR_CODE`, `SAFETY`, `OTHER`.

The intent shapes the conversational opener pool used downstream
("Here are the maintenance steps…" vs "Here are the steps…").

### Step 2 — Pattern-based safety gate

Three regex matches run in series:

1. `_PROMPT_INJECTION` — common jailbreak phrasings.
2. `_UNSUPPORTED_REPAIR` — disassemble, bypass, modify-firmware, etc.
3. `_SAFETY_LEX` — only used downstream as a feature, never blocks alone.

If pattern 1 or 2 fires, the gate returns BLOCK with the typed reason and the
decoder is never invoked.

### Step 3 — Hybrid retrieval

The retriever ranks candidate nodes two ways:

```mermaid
flowchart LR
    subgraph "Lexical"
        L1[Tokenize query] --> L2[Intersect each node's token set]
        L2 --> L3[Top-K lexical]
    end
    subgraph "Semantic"
        S1[Encode query] --> S2[Cosine vs per-node vectors]
        S2 --> S3[Top-K semantic]
    end
    L3 --> RRF[Reciprocal Rank Fusion]
    S3 --> RRF
    RRF --> P([Primary node])

    style RRF fill:#eef,stroke:#33c
```

The semantic encoder runs locally — it is a static embedding model (vocab
table with token-vector averaging). No transformer forward pass.

### Step 4 — Veto models

Two learned binary classifiers run on the primary node text:

- **Safety veto** — flags non-safety queries whose evidence is safety-relevant.
- **Wrong-entity veto** — flags evidence whose canonical entity doesn't match
  the bound product.

Both are small logistic-regression models with hand-engineered features
(token overlap, warning-pattern density, capital-warning markers, intent
agreement). The weights ship as JSON in `artifacts/safety/`.

### Step 5 — Gate decision

```mermaid
flowchart TB
    Q[query + retrieved node] --> O{overlap >= 0.15?}
    O -- no --> B1[BLOCK: no_relevant_evidence]
    O -- yes --> SV{safety_veto >= 0.3<br/>and intent != SAFETY?}
    SV -- yes --> B2[BLOCK: safety_veto]
    SV -- no --> SG{intent == SAFETY<br/>and overlap < 0.3?}
    SG -- yes --> B3[BLOCK: safety_fp_guard]
    SG -- no --> OL{overlap < 0.3?}
    OL -- yes --> R[REVIEW: low_evidence_overlap]
    OL -- no --> A[ALLOW]

    style B1 fill:#fdd,stroke:#933
    style B2 fill:#fdd,stroke:#933
    style B3 fill:#fdd,stroke:#933
    style R fill:#ffd,stroke:#993
    style A fill:#dfd,stroke:#393
```

### Step 6 — Packet enrichment

On ALLOW, the enricher walks the product graph to assemble a richer packet:

```mermaid
flowchart LR
    P([Primary node]) --> PA{Has parent<br/>procedure?}
    PA -- yes --> SB[Add parent + sibling steps<br/>under same parent]
    PA -- no --> NX{Has NEXT_STEP<br/>edge?}
    SB --> NX
    NX -- yes --> NS[Add next-step node]
    NX -- no --> WN{Has HAS_WARNING<br/>edge?}
    NS --> WN
    WN -- yes --> WC[Add warning node]
    WN -- no --> SP{Has HAS_SPEC<br/>edge?}
    WC --> SP
    SP -- yes --> SC[Add spec node]
    SP -- no --> BF{Primary is short<br/>section label?}
    SC --> BF
    BF -- yes --> SM[Semantic backfill:<br/>scan same graph for<br/>query-relevant nodes]
    BF -- no --> X([Packet ready])
    SM --> X

    style P fill:#dfd,stroke:#393
    style X fill:#fff,stroke:#333
```

Each candidate is filtered for product binding, safety-lex spillover, and
query relevance. Bare section titles are demoted to metadata.

### Step 7 — Verbalization

The verbalizer formats the packet for the local decoder. The decoder generates
a candidate completion, but the verbalizer never publishes the raw model
output — instead it stitches the model's tone with deterministic citation
anchors from the packet.

Multi-step packets render as numbered Markdown lists in node-id ascending
order (which approximates document order in the source manual):

```
The manual describes the procedure as follows [ev_67]:

1. Sort laundry by colour, fabric type, and wash symbol [ev_63].
2. Open the door and place laundry loosely in the drum [ev_65].
3. Open the detergent drawer [ev_67].
4. Add main-wash detergent ... [ev_67].
```

Single-step packets render as inline prose:

```
The manual covers this maintenance step like this: Clean the
water-inlet filter at the tap connector if needed [ev_45].
```

Short fragmentary primary nodes (section labels) use a light opener to avoid
grammatical clash:

```
From the manual: Steam cleaning the cavity [ev_18].
```

### Step 8 — Citation validation

The validator checks the proposed answer against the assembled packet:

1. Extract every `[ev_N]` token from the answer.
2. Reject if any cited node ID isn't in the packet.
3. Split the answer into sentences.
4. Reject if any sentence over 30 characters has no citation.
5. Reject if any registered wrong-product term is present.

If validation passes, the answer is published. Otherwise the verbalizer falls
back to the deterministic snippet renderer (which is rule-based and always
passes by construction), and the trace records `renderer_fallback_used: true`.

## What's in the trace

Every published answer carries a structured trace. The most important fields:

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

The pairing `(runtime_config_hash, nexus_model_hash, query, retrieval_mode)`
uniquely determines the answer. Reproducibility is enforced by the trace, not
asserted in prose.

## Refusal paths

```mermaid
flowchart TB
    Q([Query]) --> P{prompt injection<br/>pattern?}
    P -- yes --> R1[BLOCK<br/>prompt_injection_detected]
    P -- no --> U{unsupported<br/>repair pattern?}
    U -- yes --> R2[BLOCK<br/>unsupported_repair_request]
    U -- no --> WR{wrong product<br/>query?}
    WR -- yes --> R3[BLOCK<br/>wrong_product_query]
    WR -- no --> OV{evidence overlap<br/>< 0.15?}
    OV -- yes --> R4[BLOCK<br/>no_relevant_evidence]
    OV -- no --> SV{learned safety<br/>veto fires?}
    SV -- yes --> R5[BLOCK<br/>safety_veto]
    SV -- no --> WE{learned wrong-entity<br/>veto fires?}
    WE -- yes --> R6[BLOCK<br/>wrong_entity_veto]
    WE -- no --> A[ALLOW path]

    style R1 fill:#fdd,stroke:#933
    style R2 fill:#fdd,stroke:#933
    style R3 fill:#fdd,stroke:#933
    style R4 fill:#fdd,stroke:#933
    style R5 fill:#fdd,stroke:#933
    style R6 fill:#fdd,stroke:#933
    style A fill:#dfd,stroke:#393
```

On every refusal, `decoder_called`, `renderer_called`, and `nexus_called` are
all `False` in the trace. The decoder is never invoked on a refusal.
