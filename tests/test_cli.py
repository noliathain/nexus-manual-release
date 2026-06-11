"""CLI smoke tests — verify the entry-point commands behave."""
from __future__ import annotations

import json
import subprocess

from tests.conftest import REPO_ROOT


def _run(args, timeout=180):
    return subprocess.run(
        ["nexus-manual", *args],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=timeout)


def test_help_lists_demo_commands():
    r = _run(["--help"], timeout=30)
    assert r.returncode == 0
    for cmd in ("ask", "chat", "demo-chat", "prewarm"):
        assert cmd in r.stdout, cmd


def test_ask_help():
    r = _run(["ask", "--help"], timeout=30)
    assert r.returncode == 0
    assert "--product" in r.stdout
    assert "--renderer" in r.stdout
    assert "--retrieval" in r.stdout
    assert "--json" in r.stdout


def test_demo_chat_help():
    r = _run(["demo-chat", "--help"], timeout=30)
    assert r.returncode == 0


def test_ask_json_output_is_parseable():
    r = _run([
        "ask",
        "--product", "electrolux_washer_dryer",
        "--renderer", "deterministic",
        "--retrieval", "lexical",
        "--json",
        "--no-color",
        "How do I clean the filter?",
    ])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["decision"] == "ALLOW"
    assert "telemetry" in payload
    assert payload["telemetry"]["runtime_config_hash"]
    assert isinstance(payload["citations"], list)


def test_ask_unknown_product_fails_cleanly():
    r = _run([
        "ask",
        "--product", "totally_made_up_product",
        "--renderer", "deterministic",
        "--no-color",
        "Hello?",
    ])
    # Either non-zero exit or a typed refusal in stdout
    if r.returncode == 0:
        assert "unsupported_product" in r.stdout \
            or "Unknown" in r.stdout
    else:
        assert r.returncode != 0


def test_ask_unknown_renderer_rejected():
    r = _run([
        "ask",
        "--product", "electrolux_washer_dryer",
        "--renderer", "bogus",
        "--no-color",
        "Hello?",
    ], timeout=30)
    assert r.returncode != 0


def test_ask_unknown_retrieval_rejected():
    r = _run([
        "ask",
        "--product", "electrolux_washer_dryer",
        "--retrieval", "bogus",
        "--no-color",
        "Hello?",
    ], timeout=30)
    assert r.returncode != 0
