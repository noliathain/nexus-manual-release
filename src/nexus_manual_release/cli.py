"""Inference CLI for the Manual Graph-RAG release.

Commands:
  prewarm    — pre-load local decoder + build per-product
               semantic indexes so the first answer is warm.
  ask        — one-shot question, prints answer + optional
               trace + evidence.
  chat       — interactive REPL bound to one product.
  demo-chat  — polished interactive demo with suggested
               questions on startup.

All inference happens on the local machine. No external API.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False)


def _stream_answer_panel(answer_text, *, console,
                                    words_per_second=18):
    """Phase 32K: stream the validated answer text word-by-word
    into a Live Rich panel. The full answer has already been
    generated + validated; this is presentation-only streaming
    so the customer sees the LLM-chat-style typing experience
    without exposing any unvalidated raw model output."""
    import re as _re
    import time as _time
    from rich.live import Live
    from rich.panel import Panel
    from rich.markdown import Markdown
    tokens = _re.split(r"(\s+)", answer_text or "")
    accumulated = ""
    delay = 1.0 / max(words_per_second, 1)
    with Live(
            Panel(Markdown(""), title="Answer",
                    border_style="green"),
            console=console, refresh_per_second=24,
            transient=False) as live:
        for tok in tokens:
            accumulated += tok
            live.update(Panel(
                Markdown(accumulated), title="Answer",
                border_style="green"))
            if tok.strip():
                _time.sleep(delay)


def _render_runtime_answer(answer, *, show_evidence=False,
                                   trace=False, no_color=False,
                                   compact=False, stream=False):
    """Render a RuntimeAnswer with Rich panels.

    When stream=True and the decision is ALLOW, the Answer
    panel streams word-by-word. Header / Question / refusal /
    trace panels render statically."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.console import Console
    from rich.markdown import Markdown
    cons = Console(no_color=no_color, force_terminal=True)
    decision = answer.decision
    color = {"ALLOW": "green", "BLOCK": "red",
                "REVIEW": "yellow"}.get(decision, "white")
    header = (f"[bold]{answer.product_id}[/bold] · "
                  f"gate_v23c · "
                  f"[{color}]{decision}[/{color}]")
    if answer.evidence_packet_hash:
        header += (f" · packet "
                       f"[cyan]{answer.evidence_packet_hash}[/cyan]")
    cons.print(Panel(header, title="Pilot Runtime",
                          border_style=color))
    cons.print(Panel(answer.query, title="Question",
                          border_style="blue"))
    if decision == "ALLOW":
        if stream and answer.answer:
            _stream_answer_panel(answer.answer, console=cons)
        else:
            cons.print(Panel(Markdown(answer.answer),
                                  title="Answer",
                                  border_style="green"))
        if show_evidence and answer.selected_evidence_node_ids:
            t = Table(title="Evidence",
                          show_header=True,
                          header_style="bold cyan")
            t.add_column("citation")
            t.add_column("node_id")
            # Phase 32K: list every packet node by its true id.
            # selected_evidence_node_ids carries the enriched
            # packet from Phase 32J; citations may be a subset.
            for nid in answer.selected_evidence_node_ids:
                t.add_row(f"ev_{nid}", str(nid))
            cons.print(t)
    elif decision == "BLOCK":
        reason_text = {
            "prompt_injection_detected":
                "Query appears to contain a prompt-injection "
                "pattern.",
            "unsupported_repair_request":
                "I can't provide guidance for unsupported "
                "repairs or modifications.",
            "wrong_product_query":
                "This question seems to be about a different "
                "product.",
            "safety_veto":
                "Safety could not be verified with the available "
                "evidence.",
            "safety_fp_guard":
                "I can't answer this safety question without "
                "confident evidence in the manual.",
            "wrong_entity_veto":
                "I don't have evidence that closely matches "
                "this question for the selected product.",
            "no_relevant_evidence":
                "The manual doesn't appear to cover this.",
            "unsupported_product_" + (answer.product_id or ""):
                "This product is not part of the validated "
                "pilot scope.",
        }.get(answer.refusal_reason or "",
                "I can't answer this with the available evidence.")
        cons.print(Panel(reason_text,
                              title=f"Refusal ({answer.refusal_reason})",
                              border_style="red"))
        cons.print("[dim]Renderer was not called for this "
                      "refusal.[/dim]")
    else:  # REVIEW
        cons.print(Panel(
            "This needs human review — I won't emit a new "
            "answer.", title="Review", border_style="yellow"))
    if trace:
        t = Table(title="Trace",
                      show_header=True,
                      header_style="bold")
        t.add_column("field"); t.add_column("value")
        for k in ("decision", "intent", "provenance_mode",
                     "renderer_called"):
            v = getattr(answer, k, None)
            t.add_row(k, str(v))
        for k in ("renderer_mode",
                     "safety_veto_score",
                     "wrong_entity_veto_score",
                     "evidence_overlap", "latency_ms",
                     "evidence_packet_emitted",
                     "decoder_called",
                     "product_binding_status",
                     "llm_called", "llm_provider",
                     "llm_output_accepted",
                     "nexus_called", "nexus_provider",
                     "nexus_model_basename", "nexus_model_hash",
                     "nexus_output_accepted",
                     "nexus_output_rejected_reason",
                     "renderer_fallback_used",
                     "answer_validation_passed"):
            v = answer.telemetry.get(k)
            if v is not None and v is not False:
                t.add_row(k, str(v))
        cons.print(t)
    if not compact and decision == "ALLOW":
        renderer_mode = answer.telemetry.get(
            "renderer_mode", "deterministic")
        nexus_called = answer.telemetry.get(
            "nexus_called", False)
        nexus_accepted = answer.telemetry.get(
            "nexus_output_accepted", False)
        llm_called = answer.telemetry.get("llm_called", False)
        llm_accepted = answer.telemetry.get(
            "llm_output_accepted", False)
        fallback = answer.telemetry.get(
            "renderer_fallback_used", False)
        if renderer_mode == "nexus":
            if nexus_called and nexus_accepted:
                footer = (
                    "Renderer: local Nexus decoder verbalizer "
                    "over approved EvidencePacket. Nexus "
                    "called: true. Gate passed before "
                    "generation.")
            elif fallback:
                footer = (
                    "Renderer: deterministic fallback. Nexus "
                    "output failed validation and was not "
                    "shown.")
            else:
                footer = (
                    "Renderer: deterministic. Nexus called: "
                    f"{nexus_called}.")
        elif renderer_mode == "llm":
            if llm_called and llm_accepted:
                footer = (
                    "Renderer: LLM verbalizer over approved "
                    "EvidencePacket. LLM called: true. Gate "
                    "passed before generation.")
            elif fallback:
                footer = (
                    "Renderer: deterministic fallback. LLM "
                    "output failed validation and was not "
                    "shown.")
            else:
                footer = (
                    "Renderer: deterministic. LLM called: "
                    f"{llm_called}.")
        else:
            footer = (
                "Renderer: deterministic EvidencePacket "
                "renderer. Nexus called: false. External LLM "
                "called: false.")
        cons.print(f"[dim]{footer}[/dim]")
    elif not compact and decision != "ALLOW":
        cons.print("[dim]Gate blocked this query. Renderer / "
                      "Nexus / LLM was not called.[/dim]")


@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="User question"),
    product: str = typer.Option(
        ..., "--product", "-p",
        help="Product id (electrolux_steam_oven | "
        "electrolux_washer_dryer)"),
    trace: bool = typer.Option(False, "--trace",
                                          help="Show gate trace"),
    show_evidence: bool = typer.Option(
        False, "--show-evidence",
        help="Show evidence IDs"),
    json_out: bool = typer.Option(
        False, "--json", help="JSON output"),
    no_color: bool = typer.Option(False, "--no-color"),
    compact: bool = typer.Option(False, "--compact"),
    telemetry_out: Optional[Path] = typer.Option(
        None, "--telemetry-out",
        help="Append telemetry to this JSONL file"),
    renderer: str = typer.Option(
        "deterministic", "--renderer",
        help="deterministic | llm | auto. LLM is called only on "
        "ALLOW; falls back to deterministic if validation "
        "fails."),
    stream: Optional[bool] = typer.Option(
        None, "--stream/--no-stream",
        help="Stream the answer word-by-word + show a 'thinking' "
        "spinner during model generation. Default: on for "
        "nexus/llm, off for deterministic and --json."),
    retrieval: str = typer.Option(
        "lexical", "--retrieval",
        help="lexical (Phase 31 default — token overlap) | "
        "semantic (Phase 32L — weighted RRF of Model2Vec "
        "static embeddings + lexical; same encoder runs on "
        "desktop and embedded targets)."),
) -> None:
    """Ask a question to the pilot runtime (frozen gate v23c)."""
    from nexus_manual_release.runtime import answer_query
    from rich.console import Console as _Console
    if renderer not in ("deterministic", "llm", "nexus",
                                "auto"):
        typer.echo(
            f"Unknown renderer: {renderer}. Use deterministic | "
            "llm | nexus | auto.", err=True)
        raise typer.Exit(code=2)
    if retrieval not in ("lexical", "semantic"):
        typer.echo(
            f"Unknown retrieval mode: {retrieval}. Use "
            "lexical | semantic.", err=True)
        raise typer.Exit(code=2)
    if stream is None:
        stream = (renderer in ("nexus", "llm", "auto")
                       and not json_out)
    if stream and not json_out:
        _spinner_cons = _Console(no_color=no_color,
                                              force_terminal=True)
        with _spinner_cons.status(
                "[cyan]Nexus is thinking…[/cyan]"
                if renderer == "nexus"
                else "[cyan]Verbalizing answer…[/cyan]",
                spinner="dots"):
            ans = answer_query(product, query,
                                      renderer=renderer,
                                      retrieval=retrieval)
    else:
        ans = answer_query(product, query, renderer=renderer,
                                retrieval=retrieval)
    if telemetry_out is not None:
        telemetry_out.parent.mkdir(parents=True, exist_ok=True)
        with telemetry_out.open("a") as f:
            f.write(json.dumps(ans.telemetry) + "\n")
    if json_out:
        typer.echo(json.dumps(ans.to_dict(), indent=2,
                                    default=str))
    else:
        _render_runtime_answer(
            ans, show_evidence=show_evidence, trace=trace,
            no_color=no_color, compact=compact,
            stream=bool(stream))


# =============================================================
# Phase 32E — interactive chat / demo-chat (product aliases +
# slash commands + product selector)
# =============================================================

_VALIDATED_PRODUCT_LABELS = {
    "electrolux_steam_oven": "Electrolux Steam Oven",
    "electrolux_washer_dryer": "Electrolux Washer-Dryer",
}
_EXPERIMENTAL_PRODUCT_LABELS = {
    "coffee_machine":
        "Coffee Machine — experimental / safety-valid but "
        "coverage not ready",
}
_PRODUCT_ALIASES = {
    # steam oven
    "electrolux_steam_oven": "electrolux_steam_oven",
    "steam_oven": "electrolux_steam_oven",
    "steam-oven": "electrolux_steam_oven",
    "oven": "electrolux_steam_oven",
    "steam": "electrolux_steam_oven",
    "1": "electrolux_steam_oven",
    # washer dryer
    "electrolux_washer_dryer": "electrolux_washer_dryer",
    "washer_dryer": "electrolux_washer_dryer",
    "washer-dryer": "electrolux_washer_dryer",
    "washer": "electrolux_washer_dryer",
    "dryer": "electrolux_washer_dryer",
    "wd": "electrolux_washer_dryer",
    "2": "electrolux_washer_dryer",
    # coffee (experimental only)
    "coffee_machine": "coffee_machine",
    "coffee": "coffee_machine",
    "coffee-machine": "coffee_machine",
}
_EXAMPLE_QUESTIONS = {
    "electrolux_steam_oven": [
        "How do I preheat the oven?",
        "How do I remove the shelves?",
        "How do I switch on the oven?",
        "How do I disassemble the motor and bypass safety interlocks?",
    ],
    "electrolux_washer_dryer": [
        "How do I clean the filter?",
        "How do I clean the detergent drawer?",
        "How do I select a washing programme?",
        "How do I add detergent?",
        "How do I disassemble the motor?",
    ],
    "coffee_machine": [
        "How do I clean the brew group?",
        "How do I descale it?",
    ],
}
_GATE_HASH = ("2d6d28c07dd1353c12336dfda2a99c735ca26392c25"
                  "7742caafc11bfcca6ddab")


def _resolve_product(value, *, include_experimental):
    if not value: return None
    canonical = _PRODUCT_ALIASES.get(value.lower().strip())
    if canonical is None: return None
    if (canonical not in _VALIDATED_PRODUCT_LABELS
          and not include_experimental):
        return None
    return canonical


def _available_products_text(include_experimental):
    lines = ["Available validated products:"]
    for pid, label in _VALIDATED_PRODUCT_LABELS.items():
        lines.append(f"  - {pid}  ({label})")
    if include_experimental:
        lines.append("")
        lines.append("Experimental:")
        for pid, label in _EXPERIMENTAL_PRODUCT_LABELS.items():
            lines.append(f"  - {pid}  ({label})")
    else:
        lines.append("")
        lines.append("Use --include-experimental-products to "
                          "show experimental products.")
    return "\n".join(lines)


def _product_selector(include_experimental, console):
    """Show a numbered selector and return the chosen product
    id. Returns None on EOF / cancel."""
    from rich.panel import Panel
    options = list(_VALIDATED_PRODUCT_LABELS.items())
    if include_experimental:
        options += list(_EXPERIMENTAL_PRODUCT_LABELS.items())
    lines = ["Select product:"]
    for i, (pid, label) in enumerate(options, start=1):
        lines.append(f"  {i}. {label}")
    console.print(Panel("\n".join(lines),
                              title="Product Selector",
                              border_style="cyan"))
    try:
        choice = input("Choice (number or id) › ").strip()
    except EOFError:
        return None
    if not choice: return None
    # numeric pick
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx][0]
    return _resolve_product(choice,
        include_experimental=include_experimental)


def _interactive_loop(product, *, demo_mode, no_color,
                              telemetry_out, include_experimental,
                              renderer="deterministic",
                              retrieval="lexical"):
    """Single interactive chat loop. Shared by chat + demo-chat.

    Returns exit code (0 on clean exit)."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from nexus_manual_release.runtime import answer_query
    cons = Console(no_color=no_color, force_terminal=True)
    if renderer not in ("deterministic", "llm", "nexus",
                              "auto"):
        renderer = "deterministic"
    trace = False
    show_evidence = False
    json_mode = False
    compact = False
    renderer_mode = renderer
    retrieval_mode = (retrieval if retrieval in (
        "lexical", "semantic") else "lexical")
    # Phase 32K: streaming defaults to on for nexus/llm/auto
    # since those have non-trivial generation latency; off for
    # deterministic which is instant.
    stream_mode = renderer_mode in ("nexus", "llm", "auto")
    telemetry_path = (Path(telemetry_out) if telemetry_out
                            else None)
    if telemetry_path is not None:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    label = _VALIDATED_PRODUCT_LABELS.get(
        product, _EXPERIMENTAL_PRODUCT_LABELS.get(product,
                                                                product))
    # Welcome
    renderer_label = {
        "deterministic":
            "deterministic EvidencePacket renderer (no LLM)",
        "llm":
            "LLM verbalizer over approved EvidencePackets "
            "(safety gate runs first)",
        "nexus":
            "local Nexus decoder verbalizer over approved "
            "EvidencePackets (safety gate runs first)",
        "auto":
            "auto (LLM if available, else deterministic)",
    }.get(renderer_mode, renderer_mode)
    welcome_lines = [
        "[bold]Manual Graph-RAG Pilot Demo[/bold]",
        f"Product: [cyan]{label}[/cyan]",
        "Gate: gate_v23c_proven_overlay_baseline_first_guardrail",
        f"Renderer: {renderer_label}",
        "Type your question below. Type /help for commands or "
        "/exit to quit.",
    ]
    cons.print(Panel("\n".join(welcome_lines),
                          title="Welcome",
                          border_style="green"))
    if demo_mode:
        examples = _EXAMPLE_QUESTIONS.get(product, [])
        if examples:
            cons.print(Panel(
                "Try asking:\n"
                + "\n".join(f"  • {q}" for q in examples),
                title="Suggested questions",
                border_style="blue"))
    current_product = product
    prompt_label = lambda: (
        _VALIDATED_PRODUCT_LABELS.get(current_product,
            _EXPERIMENTAL_PRODUCT_LABELS.get(current_product,
                                                    current_product)))
    while True:
        try:
            q = input(f"{prompt_label()} › ")
        except (EOFError, KeyboardInterrupt):
            cons.print("[dim]bye[/dim]")
            return 0
        q = q.strip()
        if not q:
            continue
        # slash commands
        if q in ("/exit", "/quit", ":q"):
            cons.print("[dim]bye[/dim]")
            return 0
        if q == "/help":
            t = Table(title="Commands", show_header=True,
                          header_style="bold")
            t.add_column("command"); t.add_column("purpose")
            t.add_row("/help", "show this list")
            t.add_row("/exit | /quit", "end the session")
            t.add_row("/products",
                          "list validated products + scope")
            t.add_row("/product <id-or-alias>",
                          "switch product")
            t.add_row("/trace on | off", "toggle gate trace")
            t.add_row("/evidence on | off",
                          "toggle evidence display")
            t.add_row("/json on | off",
                          "echo machine-readable JSON after each "
                          "answer")
            t.add_row("/compact on | off",
                          "hide trailing grounding footer")
            t.add_row("/telemetry <path>",
                          "append telemetry JSONL to a file")
            t.add_row("/status",
                          "show current product + gate hash + "
                          "modes")
            t.add_row("/examples",
                          "show suggested demo questions")
            t.add_row("/stream on|off|status",
                          "toggle word-by-word answer streaming")
            t.add_row("/retrieval lexical|semantic|status",
                          "switch retrieval mode")
            t.add_row("/clear", "clear the screen")
            cons.print(t)
            continue
        if q == "/products":
            cons.print(Panel(
                _available_products_text(include_experimental),
                title="Products", border_style="cyan"))
            continue
        if q.startswith("/product "):
            new = q.split(maxsplit=1)[1].strip()
            resolved = _resolve_product(new,
                include_experimental=include_experimental)
            if resolved is None:
                cons.print(Panel(
                    f"Unknown product: {new}\n\n"
                    + _available_products_text(
                        include_experimental),
                    title="Product not found",
                    border_style="red"))
                continue
            current_product = resolved
            cons.print(f"[dim]Product switched to "
                          f"{prompt_label()}.[/dim]")
            cons.print("[dim]All future questions are now bound "
                          "to this product.[/dim]")
            continue
        if q == "/trace on": trace = True; continue
        if q == "/trace off": trace = False; continue
        if q == "/evidence on":
            show_evidence = True; continue
        if q == "/evidence off":
            show_evidence = False; continue
        if q == "/json on": json_mode = True; continue
        if q == "/json off": json_mode = False; continue
        if q == "/compact on": compact = True; continue
        if q == "/compact off": compact = False; continue
        if q.startswith("/telemetry "):
            telemetry_path = Path(q.split(maxsplit=1)[1].strip())
            telemetry_path.parent.mkdir(parents=True,
                                                exist_ok=True)
            cons.print(f"[dim]telemetry → "
                          f"{telemetry_path}[/dim]")
            continue
        if q == "/status":
            t = Table(title="Status", show_header=True,
                          header_style="bold")
            t.add_column("field"); t.add_column("value")
            t.add_row("product", current_product)
            t.add_row("gate",
                          "gate_v23c_proven_overlay_baseline_"
                          "first_guardrail")
            t.add_row("gate_hash_short", _GATE_HASH[:16] + "…")
            t.add_row("renderer", renderer_mode)
            if renderer_mode == "nexus":
                from nexus_manual_release.runtime import (
                    get_nexus_provider, nexus_model_hash)
                t.add_row("nexus_provider",
                              get_nexus_provider().name)
                h = nexus_model_hash()
                t.add_row("nexus_model_hash",
                              h if h else "(stub — no checkpoint)")
            t.add_row("trace", "on" if trace else "off")
            t.add_row("evidence",
                          "on" if show_evidence else "off")
            t.add_row("json", "on" if json_mode else "off")
            t.add_row("compact", "on" if compact else "off")
            t.add_row("telemetry",
                          str(telemetry_path)
                          if telemetry_path else "off")
            t.add_row("validated_scope",
                          ", ".join(_VALIDATED_PRODUCT_LABELS))
            cons.print(t)
            continue
        if q == "/examples":
            examples = _EXAMPLE_QUESTIONS.get(current_product,
                                                          [])
            if examples:
                cons.print(Panel(
                    "Try asking:\n"
                    + "\n".join(f"  • {x}" for x in examples),
                    title=f"Examples for {prompt_label()}",
                    border_style="blue"))
            else:
                cons.print(
                    "[dim]no examples for this product[/dim]")
            continue
        if q == "/clear":
            cons.clear(); continue
        if q.startswith("/renderer "):
            new_mode = q.split(maxsplit=1)[1].strip().lower()
            if new_mode not in ("deterministic", "llm", "nexus",
                                       "auto", "status"):
                cons.print(
                    f"[red]Unknown renderer mode: {new_mode}"
                    "[/red]. Use deterministic | llm | nexus | "
                    "auto | status.")
                continue
            if new_mode == "status":
                cons.print(
                    f"[dim]renderer = {renderer_mode}[/dim]")
            else:
                renderer_mode = new_mode
                # Auto-flip streaming default to match the new
                # renderer (still overridable via /stream).
                stream_mode = renderer_mode in (
                    "nexus", "llm", "auto")
                cons.print(
                    f"[dim]renderer → {renderer_mode} "
                    f"(stream {'on' if stream_mode else 'off'})"
                    f"[/dim]")
            continue
        if q.startswith("/stream"):
            arg = q[len("/stream"):].strip().lower()
            if arg in ("", "status"):
                cons.print(
                    f"[dim]stream = "
                    f"{'on' if stream_mode else 'off'}[/dim]")
            elif arg == "on":
                stream_mode = True
                cons.print("[dim]stream → on[/dim]")
            elif arg == "off":
                stream_mode = False
                cons.print("[dim]stream → off[/dim]")
            else:
                cons.print(
                    f"[red]Unknown /stream arg: {arg}[/red]. "
                    "Use /stream on|off|status.")
            continue
        if q.startswith("/retrieval"):
            arg = q[len("/retrieval"):].strip().lower()
            if arg in ("", "status"):
                cons.print(
                    f"[dim]retrieval = {retrieval_mode}[/dim]")
            elif arg in ("lexical", "semantic"):
                retrieval_mode = arg
                cons.print(
                    f"[dim]retrieval → {retrieval_mode}[/dim]")
            else:
                cons.print(
                    f"[red]Unknown /retrieval arg: {arg}[/red]. "
                    "Use /retrieval lexical|semantic|status.")
            continue
        # Real query → run the gate
        if stream_mode and not json_mode:
            with cons.status(
                    "[cyan]Nexus is thinking…[/cyan]"
                    if renderer_mode == "nexus"
                    else "[cyan]Verbalizing answer…[/cyan]",
                    spinner="dots"):
                ans = answer_query(current_product, q,
                                          renderer=renderer_mode,
                                          retrieval=retrieval_mode)
        else:
            ans = answer_query(current_product, q,
                                    renderer=renderer_mode,
                                    retrieval=retrieval_mode)
        if telemetry_path is not None:
            with telemetry_path.open("a") as f:
                f.write(json.dumps(ans.telemetry) + "\n")
        _render_runtime_answer(
            ans, show_evidence=show_evidence, trace=trace,
            no_color=no_color, compact=compact,
            stream=stream_mode and not json_mode)
        if json_mode:
            cons.print(json.dumps(ans.to_dict(), indent=2,
                                        default=str))


@app.command("chat")
def chat(
    product: Optional[str] = typer.Option(
        None, "--product", "-p",
        help="Product id or alias (steam_oven, washer, etc.). "
        "If omitted you'll see a selector."),
    no_color: bool = typer.Option(False, "--no-color"),
    telemetry_out: Optional[Path] = typer.Option(
        None, "--telemetry-out",
        help="Append telemetry JSONL to this file."),
    include_experimental_products: bool = typer.Option(
        False, "--include-experimental-products",
        help="Show coffee_machine (experimental, not validated "
        "for customer pilot)."),
    renderer: str = typer.Option(
        "deterministic", "--renderer",
        help="deterministic | llm | nexus | auto."),
    retrieval: str = typer.Option(
        "lexical", "--retrieval",
        help="lexical | semantic. Semantic uses Model2Vec "
        "static embeddings (same encoder offline + on-device)."),
) -> None:
    """Interactive operator-focused pilot chat."""
    from rich.console import Console
    cons = Console(no_color=no_color, force_terminal=True)
    if product is None:
        resolved = _product_selector(
            include_experimental_products, cons)
    else:
        resolved = _resolve_product(product,
            include_experimental=include_experimental_products)
    if resolved is None:
        cons.print(
            f"[red]Unknown product:[/red] {product}\n\n"
            + _available_products_text(
                include_experimental_products))
        raise typer.Exit(code=2)
    exit_code = _interactive_loop(
        resolved, demo_mode=False, no_color=no_color,
        telemetry_out=str(telemetry_out)
            if telemetry_out else None,
        include_experimental=include_experimental_products,
        renderer=renderer,
        retrieval=retrieval)
    raise typer.Exit(code=exit_code)


@app.command("demo-chat")
def demo_chat(
    product: Optional[str] = typer.Option(
        None, "--product", "-p",
        help="Product id or alias for the live demo."),
    no_color: bool = typer.Option(False, "--no-color"),
    telemetry_out: Optional[Path] = typer.Option(
        None, "--telemetry-out",
        help="Append telemetry JSONL to this file."),
    include_experimental_products: bool = typer.Option(
        False, "--include-experimental-products",
        help="Show experimental products in selector."),
    renderer: str = typer.Option(
        "deterministic", "--renderer",
        help="deterministic | llm | nexus | auto."),
    retrieval: str = typer.Option(
        "lexical", "--retrieval",
        help="lexical | semantic. Semantic uses Model2Vec "
        "static embeddings (same encoder offline + on-device)."),
) -> None:
    """Polished interactive demo for customer/meeting use.

    Shows suggested questions at startup. Same runtime + gate as
    `chat`; just a friendlier presentation."""
    from rich.console import Console
    cons = Console(no_color=no_color, force_terminal=True)
    if product is None:
        resolved = _product_selector(
            include_experimental_products, cons)
    else:
        resolved = _resolve_product(product,
            include_experimental=include_experimental_products)
    if resolved is None:
        cons.print(
            f"[red]Unknown product:[/red] {product}\n\n"
            + _available_products_text(
                include_experimental_products))
        raise typer.Exit(code=2)
    exit_code = _interactive_loop(
        resolved, demo_mode=True, no_color=no_color,
        telemetry_out=str(telemetry_out)
            if telemetry_out else None,
        include_experimental=include_experimental_products,
        renderer=renderer,
        retrieval=retrieval)
    raise typer.Exit(code=exit_code)


@app.command("prewarm")
def prewarm(
    no_color: bool = typer.Option(False, "--no-color"),
    products: str = typer.Option(
        "electrolux_washer_dryer,electrolux_steam_oven",
        "--products",
        help="Comma-separated product ids to prewarm."),
) -> None:
    """Pre-cache all runtime artifacts (Nexus checkpoint,
    Model2Vec encoder, per-product semantic index) so the
    first answer in a live demo doesn't pay any download or
    lazy-load cost.

    Recommended before customer demos:
        uv run mgr prewarm
    """
    from rich.console import Console
    from nexus_manual_release.runtime import (
        answer_query, get_nexus_provider)
    cons = Console(no_color=no_color, force_terminal=True)
    cons.print("[bold cyan]Pre-warming runtime artifacts...[/]")
    with cons.status(
            "[cyan]Loading local Nexus decoder…[/cyan]",
            spinner="dots"):
        p = get_nexus_provider("local_nexus")
        try:
            p._lazy_load()
            cons.print("  [green]✓[/green] Nexus decoder loaded")
        except Exception as e:
            cons.print(f"  [yellow]![/yellow] Nexus decoder: {e}")
    for pid in [s.strip() for s in products.split(",")
                    if s.strip()]:
        with cons.status(
                f"[cyan]Building semantic index for "
                f"{pid}…[/cyan]", spinner="dots"):
            try:
                _ = answer_query(
                    pid, "warmup", renderer="deterministic",
                    retrieval="semantic")
                cons.print(
                    f"  [green]✓[/green] {pid} index ready")
            except Exception as e:
                cons.print(
                    f"  [yellow]![/yellow] {pid}: {e}")
    cons.print(
        "[bold green]Pre-warm complete.[/] "
        "Run demo-chat now — first answer will be steady-state "
        "latency (~3-4s), not cold-load (~10s).")
    cons.print(
        "[dim]Tip — for a truly silent demo "
        "(no hub revision-check pings), prefix your demo "
        "command with [bold]HF_HUB_OFFLINE=1[/bold]:[/dim]")
    cons.print(
        "[dim]  HF_HUB_OFFLINE=1 uv run mgr demo-chat "
        "--product electrolux_washer_dryer "
        "--renderer nexus --retrieval semantic[/dim]")



def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":
    main()
