#!/usr/bin/env bash
# Setup helper for non-technical users.
#
# Usage:
#   bash setup.sh
#
# This script does what the Setup Guide describes step-by-step. It is
# idempotent — running it twice is safe.

set -euo pipefail

GREEN="$(printf '\033[1;32m')"
YELLOW="$(printf '\033[1;33m')"
RED="$(printf '\033[1;31m')"
RESET="$(printf '\033[0m')"

say() { printf "%s%s%s\n" "$GREEN" "$1" "$RESET"; }
warn() { printf "%s%s%s\n" "$YELLOW" "$1" "$RESET"; }
err() { printf "%s%s%s\n" "$RED" "$1" "$RESET" >&2; }

say "==> Manual Graph-RAG release setup"
echo

# 1. Check uv is available.
if ! command -v uv >/dev/null 2>&1; then
    warn "uv is not installed. Installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer adds uv to ~/.cargo/bin or ~/.local/bin; nudge PATH.
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv install completed, but 'uv' is still not on PATH."
        err "Close this terminal, open a fresh one, and re-run this script."
        exit 1
    fi
fi

UV_VERSION="$(uv --version 2>/dev/null | head -1)"
say "✓ uv available: $UV_VERSION"
echo

# 2. Verify we're in the project root.
if [[ ! -f pyproject.toml ]]; then
    err "This script must be run from the nexus-manual-release directory."
    err "  cd into the project folder first, then re-run: bash setup.sh"
    exit 1
fi

if [[ ! -f models/local_decoder.pt ]]; then
    err "Bundled model files are missing. Did you clone the full repository?"
    err "If you downloaded the ZIP, make sure it extracted completely."
    exit 1
fi

# 3. Install dependencies.
say "==> Installing dependencies (this takes 2-5 minutes the first time)..."
uv sync
echo

say "✓ dependencies installed"
echo

# 4. Pre-warm.
say "==> Pre-warming the local decoder (one-time, ~30 seconds)..."
uv run nexus-manual prewarm
echo

# 5. Verify with a quick sanity test.
say "==> Verifying the install with a quick test..."
HF_HUB_OFFLINE=1 uv run nexus-manual ask \
    --product electrolux_steam_oven \
    --renderer nexus --retrieval semantic \
    --no-stream --no-color --json \
    "How do I clean the cavity?" \
    > /tmp/nexus_manual_smoke.json 2>/dev/null

if grep -q '"decision": "ALLOW"' /tmp/nexus_manual_smoke.json; then
    say "✓ smoke test passed — system is working"
else
    err "Smoke test produced an unexpected result. See /tmp/nexus_manual_smoke.json"
    exit 1
fi

rm -f /tmp/nexus_manual_smoke.json
echo

# 6. Print next steps.
cat <<'EOF'
================================================================
                  Setup complete!
================================================================

Run the interactive demo:

  HF_HUB_OFFLINE=1 uv run nexus-manual demo-chat \
      --product electrolux_washer_dryer \
      --renderer nexus --retrieval semantic

When the prompt appears, try these questions:

  How do I select the spin speed?
  How do I add detergent?
  How do I bypass the door lock?
  /exit

For the full demo walkthrough, see docs/demo_script.md.
For help with any issues, see docs/troubleshooting.md.

================================================================
EOF
