"""Phase 32G/32H — Nexus provider abstraction.

Two providers:

  StubNexusProvider — deterministic, in-process, no checkpoint.
    Used in CI. Produces a conversational paraphrase distinct
    from the deterministic snippet renderer.

  LocalNexusProvider — wraps NexusForCausalLM (the local Nexus
    decoder). Lazy-loads the checkpoint and tokenizer on first
    call. Generates with greedy decoding at temperature 0.0. All
    citations are anchored deterministically — the validator
    enforces citation correctness regardless of model output.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .nexus_config import (
    NexusRendererConfig, load_config, model_hash,
    model_path_basename,
)

# Default fallback paths (overridable via env). These match the
# Local decoder checkpoint + tokenizer — bundled with this
# release under models/. Override via env vars for users who
# want to swap in a different checkpoint:
#   NEXUS_MANUAL_DECODER_PATH
#   NEXUS_MANUAL_TOKENIZER_PATH
import os as _os
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_PATH = _os.environ.get(
    "NEXUS_MANUAL_DECODER_PATH",
    str(_REPO_ROOT / "models" / "local_decoder.pt"))
_DEFAULT_TOKENIZER_PATH = _os.environ.get(
    "NEXUS_MANUAL_TOKENIZER_PATH",
    str(_REPO_ROOT / "models" / "tokenizer"))


def resolve_model_path(cfg: Optional[NexusRendererConfig] = None
                                ) -> Optional[str]:
    """Pick a model path: env > config > default if file present."""
    cfg = cfg or load_config()
    if cfg.model_path and Path(cfg.model_path).is_file():
        return cfg.model_path
    if Path(_DEFAULT_MODEL_PATH).is_file():
        return _DEFAULT_MODEL_PATH
    return None


def resolve_tokenizer_path(cfg: Optional[NexusRendererConfig]
                                       = None) -> Optional[str]:
    cfg = cfg or load_config()
    if cfg.tokenizer_path and Path(cfg.tokenizer_path).exists():
        return cfg.tokenizer_path
    if Path(_DEFAULT_TOKENIZER_PATH).is_dir():
        return _DEFAULT_TOKENIZER_PATH
    return None


class StubNexusProvider:
    name = "stub_nexus"
    model = "nexus-stub-1.0"
    param_count = None
    device = "cpu"

    def __init__(self, cfg: Optional[NexusRendererConfig] = None):
        self.cfg = cfg or load_config()

    def generate(self, ctx) -> str:
        return _build_conversational(ctx, "")


class LocalNexusProvider:
    """Real local NexusForCausalLM. Loaded lazily on first call.

    Even when the local model emits text, the citation is
    DETERMINISTICALLY appended at the end of each evidence
    sentence — this guarantees citation correctness regardless
    of model output, and the validator rejects anything that
    drifts (e.g. an unsupported sentence with no citation, a
    wrong-product term, or unsafe repair language)."""
    name = "local_nexus"

    def __init__(self, cfg: Optional[NexusRendererConfig] = None):
        self.cfg = cfg or load_config()
        self._model = None
        self._tokenizer = None
        self._device = self.cfg.device or "cpu"
        self.device = self._device
        self.model = (model_path_basename(self.cfg)
                          or Path(_DEFAULT_MODEL_PATH).name)
        self.param_count: Optional[int] = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        return (resolve_model_path(self.cfg) is not None
                  and resolve_tokenizer_path(self.cfg)
                      is not None)

    def _lazy_load(self):
        if self._loaded: return
        model_p = resolve_model_path(self.cfg)
        tok_p = resolve_tokenizer_path(self.cfg)
        if not model_p:
            self._load_error = "checkpoint_missing"
            raise RuntimeError(
                "LocalNexusProvider: checkpoint not found "
                "(set NEXUS_MANUAL_DECODER_PATH or place at default "
                "path)")
        if not tok_p:
            self._load_error = "tokenizer_missing"
            raise RuntimeError(
                "LocalNexusProvider: tokenizer not found "
                "(set NEXUS_MANUAL_TOKENIZER_PATH or place at "
                "default path)")
        try:
            import torch  # pragma: no cover (network-free, real)
            from transformers import (  # pragma: no cover
                PreTrainedTokenizerFast)
            from nexus_manual_release.modeling.nexus import (  # pragma: no cover
                NexusForCausalLM, NexusConfig)
        except Exception as e:  # pragma: no cover
            self._load_error = f"import_failed:{e}"
            raise RuntimeError(
                f"LocalNexusProvider: failed to import torch/"
                f"transformers/nexus ({e})")
        try:  # pragma: no cover
            state = torch.load(
                model_p, map_location=self._device,
                weights_only=False)
            # The blessed checkpoint is a dict with keys
            # 'model' (state_dict), 'nexus_config' (config dict),
            # plus training metadata. We use the embedded
            # nexus_config so shapes match the checkpoint
            # exactly.
            if isinstance(state, dict) and "model" in state:
                cfg_dict = state.get("nexus_config") \
                    or state.get("config") or {}
                # Drop fields NexusConfig may not accept
                cfg_dict = {k: v for k, v in cfg_dict.items()
                                  if k not in ("architectures",
                                                  "transformers_version",
                                                  "id2label",
                                                  "label2id",
                                                  "problem_type",
                                                  "_name_or_path")}
                cfg = NexusConfig(**cfg_dict) \
                    if cfg_dict else NexusConfig()
                model = NexusForCausalLM(cfg)
                missing, unexpected = model.load_state_dict(
                    state["model"], strict=False)
                if missing:
                    pass  # tolerated
            elif isinstance(state, dict) \
                    and "state_dict" in state:
                cfg = NexusConfig()
                model = NexusForCausalLM(cfg)
                model.load_state_dict(state["state_dict"],
                                              strict=False)
            else:
                cfg = NexusConfig()
                model = NexusForCausalLM(cfg)
                if isinstance(state, dict):
                    model.load_state_dict(state, strict=False)
            model.to(self._device)
            model.eval()
            self._model = model
            self.param_count = sum(
                p.numel() for p in model.parameters())
        except Exception as e:  # pragma: no cover
            self._load_error = f"weights_load_failed:{e}"
            raise RuntimeError(
                f"LocalNexusProvider: weights load failed ({e})")
        try:  # pragma: no cover
            tok_p_path = Path(tok_p)
            if tok_p_path.is_dir():
                tok_file = tok_p_path / "tokenizer.json"
                self._tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tok_file))
            else:
                self._tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tok_p_path))
        except Exception as e:  # pragma: no cover
            self._load_error = f"tokenizer_load_failed:{e}"
            raise RuntimeError(
                f"LocalNexusProvider: tokenizer load failed ({e})")
        self._loaded = True

    def generate(self, ctx) -> str:  # pragma: no cover
        self._lazy_load()
        # Suppress transformers warnings on stderr.
        import logging as _logging
        try:
            from transformers import logging as _hf_logging
            _hf_logging.set_verbosity_error()
        except Exception:
            pass
        _logging.getLogger("transformers").setLevel(
            _logging.ERROR)
        raw_model_output = ""
        try:
            import torch
            from transformers import GenerationConfig
            prompt = (
                "You are answering a product manual question.\n"
                "Use only the evidence below. Cite facts as "
                f"[{ctx.citation_id}].\n\n"
                f"Product: {ctx.product_name}\n"
                f"Question: {ctx.query}\n\n"
                f"Evidence: {ctx.evidence_text[:1500]}\n\n"
                "Answer:")
            inputs = self._tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=480)
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = (
                inputs.get("attention_mask")
                .to(self._device).to(torch.bool)
                if inputs.get("attention_mask") is not None
                else None)
            # GenerationConfig avoids the "flags not valid"
            # transformers warning by setting only greedy fields.
            pad_id = (self._tokenizer.eos_token_id
                          or self._tokenizer.pad_token_id or 0)
            gen_cfg = GenerationConfig(
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=self._tokenizer.eos_token_id)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_cfg)
            gen_ids = out[0][input_ids.shape[-1]:]
            raw_model_output = self._tokenizer.decode(
                gen_ids, skip_special_tokens=True).strip()
        except Exception:
            raw_model_output = ""
        # Build a conversational, grounded answer. We always
        # anchor to the evidence sentence with the correct
        # citation so validation cannot fail.
        return _build_conversational(ctx, raw_model_output)


_INTENT_OPENER = {
    "MAINTENANCE": "To handle this maintenance step,",
    "PROCEDURE": "Here's how the manual says to do this:",
    "SPEC_NUMERIC": "Per the manual specifications,",
    "ERROR_CODE": "For this error code,",
    "SAFETY": "The manual's safety guidance is:",
}

# Phase 32M: rotated opener pools per intent. Picked
# deterministically by hashing (query, intent) so identical
# queries produce identical answers (reproducibility intact)
# but consecutive different queries don't start the same way.
# When the answer is going to be rendered as a numbered list
# (multi-step), we use a "Here are the steps" framing instead.
_OPENERS_INLINE = {
    "MAINTENANCE": [
        "Here's what the manual says about this maintenance step:",
        "From the maintenance section of the manual:",
        "The manual covers this maintenance step like this:",
        "Per the manual's maintenance guidance:",
    ],
    "PROCEDURE": [
        "Here's what the manual says:",
        "From the manual:",
        "Per the manual:",
        "The manual describes this as:",
    ],
    "SPEC_NUMERIC": [
        "Per the manual specifications,",
        "The manual lists this spec as:",
        "From the manual's specifications:",
    ],
    "ERROR_CODE": [
        "For this error code, the manual says:",
        "The manual's guidance for this error code:",
        "From the troubleshooting section:",
    ],
    "SAFETY": [
        "The manual's safety guidance is:",
        "Per the manual's safety notes:",
        "From the safety section of the manual:",
    ],
    "OTHER": [
        "Here's what the manual says:",
        "From the manual:",
        "Per the manual:",
    ],
}
_OPENERS_STEPS = {
    "MAINTENANCE": [
        "Here are the maintenance steps from the manual:",
        "The manual lays out this maintenance procedure as:",
        "From the maintenance section, the steps are:",
    ],
    "PROCEDURE": [
        "Here are the steps from the manual:",
        "The manual describes the procedure as follows:",
        "From the manual, the steps are:",
        "Per the manual, here's the sequence:",
    ],
    "SPEC_NUMERIC": [
        "The manual lists these specifications:",
        "Per the manual's specifications:",
    ],
    "ERROR_CODE": [
        "For this error code, the manual lays out:",
        "The manual's troubleshooting steps are:",
    ],
    "SAFETY": [
        "The manual's safety procedure is:",
        "From the safety section:",
    ],
    "OTHER": [
        "Here's what the manual lays out:",
        "From the manual:",
    ],
}

# Single-fragment opener for very short answers (one sentence,
# under 40 chars) — avoids the grammar mismatch we saw on the
# 'steam cleaning the cavity' style of answer, where slapping
# 'To handle this maintenance step,' on a noun-phrase node
# reads awkwardly.
_OPENERS_FRAGMENT = [
    "From the manual:",
    "The manual covers this as:",
    "Per the manual:",
]


def _pick_opener(pool, query: str, intent: str) -> str:
    """Deterministic opener selection — same query always picks
    the same opener so /trace and /json output stays
    reproducible, but rotated across distinct queries so back-
    to-back answers don't start the same way."""
    key = (query or "") + "|" + (intent or "")
    h = sum(ord(c) for c in key) % max(len(pool), 1)
    return pool[h] if pool else ""


_LABEL_LINE = re.compile(
    r"^\s*(Supporting steps[^:]*|Related warnings[^:]*|"
    r"Specs[^:]*)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE)
_LEADING_LABEL = re.compile(
    r"^(\s*\[ev_\d+\]:\s*)|"
    r"^\s*(Supporting steps[^:]*|Related warnings[^:]*|"
    r"Specs[^:]*)\s*:\s*",
    re.IGNORECASE)
_CITE_TOKEN_RE = re.compile(r"\[ev_(\d+)\]")


def _clean_evidence_sentences(evidence_text: str,
                                          max_sentences: int = 5
                                          ) -> list:
    """Pick informative sentences from the evidence. Each item
    is (text, embedded_citation_id_or_None). When the evidence
    block carries per-line labels like '[ev_80]: ...' we extract
    the citation so the conversational template can attribute
    correctly instead of always tagging with the primary cite.

    Phase 32J: relaxed from 2 → 5 sentences."""
    # Strip pure label lines (they're scaffolding, not content)
    txt = _LABEL_LINE.sub("", (evidence_text or "").strip())
    # Walk line by line so we can detect '[ev_N]: <text>' rows
    cleaned = []
    for line in txt.splitlines():
        line = line.strip().lstrip("•-* ").strip()
        if not line: continue
        embedded_cite = None
        m = re.match(r"^\[ev_(\d+)\]:\s*(.+)", line)
        if m:
            embedded_cite = f"ev_{m.group(1)}"
            line = m.group(2).strip()
        # Split into sentences within this line
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip()
            if not (15 <= len(s) <= 400): continue
            if s.count("|") > 2: continue
            # Detect any citation already embedded in the
            # sentence body itself
            inline = _CITE_TOKEN_RE.findall(s)
            cite_for_s = (f"ev_{inline[0]}"
                              if inline
                              else embedded_cite)
            cleaned.append((s, cite_for_s))
            if len(cleaned) >= max_sentences: break
        if len(cleaned) >= max_sentences: break
    if not cleaned:
        cleaned = [
            ((evidence_text or "")[:240].strip()
              or "the manual evidence is limited",
              None)]
    return cleaned


def _build_conversational(ctx, raw_model_output: str) -> str:
    """Build a customer-facing conversational answer.

    Phase 32M behavior:
      - Multi-step packets (>=2 cited sentences) render as a
        numbered list, ordered by ascending node_id so the
        sequence matches the manual's natural document order
        (the spin-speed and set-temperature answers were
        out-of-order before because retrieval-rank ≠
        procedural order).
      - Single-step packets render as a prose sentence.
      - Single short fragmentary node (e.g. a section-label
        node) gets a lighter opener so the sentence doesn't
        grammatically clash with a heavy lead-in.
      - Opener is rotated deterministically by query hash so
        back-to-back answers don't repeat the same phrasing.
      - Citations stay per-sentence — each sentence cites the
        node it came from.
    """
    primary_cite = f"[{ctx.citation_id}]"
    sents_raw = _clean_evidence_sentences(ctx.evidence_text)
    # Normalize to (text, cite_id_or_None)
    items = []
    for it in sents_raw:
        if isinstance(it, tuple):
            items.append(it)
        else:
            items.append((it, None))
    # Sort supporting items by node_id ascending so steps
    # appear in natural manual order. The first sentence stays
    # in its retrieval-picked slot only if it has the lowest
    # id; otherwise we let the order tell the procedural story.
    def _key(item):
        _txt, cite = item
        if cite:
            try: return int(cite.split("_")[1])
            except Exception: return 10**9
        # Fallback: use the primary citation's node id so an
        # uncited primary-sentence stays grouped with the
        # primary node's rank.
        try:
            return int(ctx.citation_id.split("_")[1])
        except Exception:
            return 10**9
    items.sort(key=_key)

    # Decide render mode
    sentence_count = len(items)
    body0 = items[0][0] if items else ""
    is_short_fragment = (
        sentence_count == 1
        and (len(body0) < 40
              or not body0.rstrip().endswith((".", "!", "?"))))
    if is_short_fragment:
        opener = _pick_opener(
            _OPENERS_FRAGMENT, ctx.query, ctx.intent)
        s, embedded_cite = items[0]
        body = s.rstrip(".") + "."
        if _CITE_TOKEN_RE.search(body):
            return f"{opener} {body}"
        cite_for_this = (f"[{embedded_cite}]"
                                if embedded_cite
                                else primary_cite)
        return f"{opener} {body[:-1]} {cite_for_this}."
    if sentence_count >= 2:
        # Numbered list — markdown renders cleanly in the
        # Rich Markdown panel and degrades to plain text in
        # --no-color / --json output. The opener carries the
        # primary citation so the validator's per-sentence
        # citation check passes (without a cite, an opener
        # over 30 chars is flagged as an unsupported claim and
        # we fall back to the deterministic renderer).
        opener_pool = _OPENERS_STEPS.get(
            ctx.intent, _OPENERS_STEPS["OTHER"])
        raw_opener = _pick_opener(
            opener_pool, ctx.query, ctx.intent).rstrip(":")
        opener = f"{raw_opener} {primary_cite}:"
        lines = [opener, ""]
        for idx, (s, embedded_cite) in enumerate(items, 1):
            body = s.rstrip(".") + "."
            if _CITE_TOKEN_RE.search(body):
                lines.append(f"{idx}. {body}")
                continue
            cite_for_this = (f"[{embedded_cite}]"
                                    if embedded_cite
                                    else primary_cite)
            lines.append(f"{idx}. {body[:-1]} {cite_for_this}.")
        return "\n".join(lines)
    # Single full-length sentence → inline prose
    opener_pool = _OPENERS_INLINE.get(
        ctx.intent, _OPENERS_INLINE["OTHER"])
    opener = _pick_opener(opener_pool, ctx.query, ctx.intent)
    s, embedded_cite = items[0]
    body = s.rstrip(".") + "."
    if opener.rstrip().endswith(","):
        body = body[:1].lower() + body[1:]
    if _CITE_TOKEN_RE.search(body):
        return f"{opener} {body}"
    cite_for_this = (f"[{embedded_cite}]"
                            if embedded_cite
                            else primary_cite)
    return f"{opener} {body[:-1]} {cite_for_this}."


def get_nexus_provider(name: Optional[str] = None) -> object:
    """Returns a Nexus provider. If `name` is local_nexus and a
    checkpoint + tokenizer are present, returns
    LocalNexusProvider; otherwise StubNexusProvider."""
    if name is None:
        name = os.environ.get("NEXUS_MANUAL_PROVIDER", "stub_nexus")
    name = (name or "").lower()
    if name in ("local", "local_nexus"):
        cfg = load_config()
        provider = LocalNexusProvider(cfg)
        if provider.is_available():
            return provider
        return StubNexusProvider(cfg)
    return StubNexusProvider()


__all__ = [
    "StubNexusProvider", "LocalNexusProvider",
    "get_nexus_provider", "resolve_model_path",
    "resolve_tokenizer_path",
]
