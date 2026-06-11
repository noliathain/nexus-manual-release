# Troubleshooting Guide

Common issues encountered during setup and how to fix them. Each section has the error message, why it happens, and what to do.

If your issue isn't listed here, see the "Getting more help" section at the bottom.

---

## Installation problems

### `uv: command not found`

**What you see:**

```
$ uv --version
zsh: command not found: uv
```

(or `bash: uv: command not found`)

**Why it happens:** uv was installed, but your terminal hasn't picked up the new `PATH` yet.

**Fix:** Close the terminal window completely and open a fresh one. Then try `uv --version` again.

If that still doesn't work, the installer probably put `uv` in a non-default location. Try:

```bash
~/.cargo/bin/uv --version
```

or

```bash
~/.local/bin/uv --version
```

If one of those works, you can either use that full path every time, or add the directory to your PATH (advanced — search "add to PATH macOS" or "add to PATH Linux" for instructions).

---

### `git: command not found`

**What you see:**

```
$ git clone https://github.com/noliathain/nexus-manual-release.git
zsh: command not found: git
```

**Why it happens:** git isn't installed.

**Fix — Option A (easiest)**: skip git entirely. Download the repository as a ZIP file:

1. Go to [https://github.com/noliathain/nexus-manual-release](https://github.com/noliathain/nexus-manual-release) in your browser.
2. Click the green **"Code"** button.
3. Click **"Download ZIP"**.
4. Unzip the file (double-click on macOS, right-click → Extract on Windows).
5. In your terminal:

   ```bash
   cd ~/Downloads/nexus-manual-release-master
   ```

   (or wherever you unzipped it). Note the `-master` suffix on the folder name when downloaded as ZIP — this is normal.

**Fix — Option B**: install git.

- **macOS**: Open a terminal, type `git --version`. If git isn't installed, macOS will offer to install Apple's Developer Command Line Tools (which includes git). Click "Install" and wait ~5 minutes.
- **Linux (Ubuntu/Debian)**: `sudo apt install git -y`
- **Linux (Fedora/RHEL)**: `sudo dnf install git -y`

---

### `uv sync` is very slow or hangs

**What you see:** progress bars that stop moving for several minutes.

**Why it happens:** PyTorch is large (~700 MB). On a slow connection it can take 10+ minutes.

**Fix:**
1. Be patient. Wait at least 10 minutes.
2. If it hangs entirely (no progress for 5+ minutes), press **Ctrl+C** to cancel, then re-run `uv sync`. The downloads resume from where they stopped.
3. If your connection is genuinely slow, try running `uv sync` overnight.

---

### `uv sync` fails with "No space left on device"

**Why it happens:** You don't have enough disk space. The full install needs ~2 GB.

**Fix:**
1. Check available space: `df -h` (Linux/macOS) or `Get-PSDrive` (PowerShell).
2. Free up at least 2 GB of space.
3. Re-run `uv sync`.

---

### `uv sync` fails with "SSL certificate problem"

**Why it happens:** Your network has a proxy or firewall that intercepts HTTPS traffic.

**Fix:**
1. If you're on a corporate network, contact IT. They can either whitelist `*.pypi.org` and `*.astral.sh`, or give you a CA certificate to install.
2. If you're on a public network (coffee shop, hotel), switch to a personal hotspot from your phone temporarily.

---

## Running-the-demo problems

### "command not found: nexus-manual"

**Why it happens:** You forgot the `uv run` prefix. After `uv sync`, the `nexus-manual` command lives inside the project's virtual environment, not in your global PATH.

**Wrong:**

```bash
nexus-manual demo-chat ...
```

**Right:**

```bash
uv run nexus-manual demo-chat ...
```

`uv run` activates the project's environment in the background and runs the command there.

---

### The first answer takes 10+ seconds

**This is normal.** The first time you ask a question, the AI model is loaded from disk into memory. This takes 7-10 seconds on a typical laptop.

**Subsequent answers** are 3-4 seconds.

**To avoid the wait**, run `uv run nexus-manual prewarm` once before the demo starts. This pre-loads the model.

---

### The demo says "downloading..." even though I have no internet

**What you see:**

```
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 10 files: 100% ...
```

**Why it happens:** A library is checking the HuggingFace server for updates, even though everything we need is bundled.

**Fix:** Always run the demo with `HF_HUB_OFFLINE=1` at the start:

```bash
HF_HUB_OFFLINE=1 uv run nexus-manual demo-chat ...
```

The setup guide has this in every example.

---

### "I see weird characters like `[2K[1A` in the output"

**What you see:**

```
[2K[1A[2K[?25l...
```

**Why it happens:** Your terminal doesn't fully support the colored / animated text the demo uses.

**Fix:** Add `--no-color` to the command:

```bash
HF_HUB_OFFLINE=1 uv run nexus-manual demo-chat \
    --product electrolux_washer_dryer \
    --renderer nexus \
    --retrieval semantic \
    --no-color
```

---

### The demo answer text is cut off mid-sentence

**Why it happens:** Your terminal window is too narrow.

**Fix:** Resize your terminal window to be wider (drag the right edge). The demo uses a 130-character-wide layout.

---

### "Error: unknown product: electrolux_washer_dryer"

**Why it happens:** You're not in the project directory. The system looks for product graphs in a `artifacts/products/` folder relative to the current directory.

**Fix:**

```bash
cd ~/Desktop/nexus-manual-release   # or wherever you cloned it
```

Then run the demo command again.

---

## Test-suite problems

### `pytest: command not found`

**Why it happens:** Same as the `nexus-manual` issue — you forgot the `uv run` prefix.

**Wrong:**

```bash
pytest tests/
```

**Right:**

```bash
uv run pytest tests/ -v
```

---

### Tests fail with "Frozen safety gate hash mismatch"

**What you see:**

```
FAILED tests/test_bundle_integrity.py::test_frozen_safety_gate_hash_matches
AssertionError: safety gate config has been modified.
  expected: 2d6d28c0...ca6ddab
  got:      a1b2c3d4...
```

**Why it happens:** Someone changed `configs/safety_gate.yaml`. This is a deliberate fail-loud check.

**Fix:** Restore the original gate config:

```bash
git checkout configs/safety_gate.yaml
```

If you didn't clone via git, re-download the repository (it'll have the correct file).

---

### Tests fail with "Bundled encoder not found"

**What you see:**

```
FAILED tests/test_bundle_integrity.py::test_static_embedding_encoder_bundled
AssertionError: models/encoder/model.safetensors not found
```

**Why it happens:** When you downloaded the repository as a ZIP, the large encoder file might not have downloaded correctly.

**Fix:** Re-clone via git, or re-download the ZIP. The encoder file should be ~30 MB.

---

## Windows-specific problems

### "PowerShell execution policy" error

**What you see:**

```
File ... cannot be loaded because running scripts is disabled on this system.
```

**Why it happens:** Windows blocks PowerShell scripts by default.

**Fix:** Run PowerShell as Administrator (search "PowerShell" → right-click → "Run as administrator") and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run the uv install command.

---

### Paths with spaces cause errors

If your username has a space (e.g. "C:\Users\John Smith\Desktop"), some commands fail.

**Fix:** Always wrap paths with spaces in quotes:

```powershell
cd "C:\Users\John Smith\Desktop\nexus-manual-release"
```

Or move the project to a path without spaces (e.g. `C:\Projects\nexus-manual-release`).

---

## Performance problems

### Answers take 30+ seconds, not 3-4

**Why it happens:** You're on a very old CPU, or another program is hogging the CPU.

**Fix:**
1. Close other applications (browsers with many tabs, video editors, etc.).
2. Make sure your laptop is plugged in (some laptops throttle the CPU on battery).
3. Check CPU usage: on macOS use "Activity Monitor", on Windows use "Task Manager".

A 3.5-second answer assumes a typical 2018+ laptop CPU. On older hardware, expect 10-15 seconds per answer.

---

### My laptop fan is going wild

**This is normal.** The AI model uses 100% of one CPU core during generation. The fan is doing its job.

The fan will quiet down between questions.

---

## "I broke something" recovery

### I want to start over from scratch

```bash
cd ~/Desktop/nexus-manual-release   # or wherever it is
rm -rf .venv
uv sync
```

This deletes the virtual environment and rebuilds it.

---

### I want to delete the entire project

```bash
rm -rf ~/Desktop/nexus-manual-release   # or wherever it is
```

This deletes the project folder and everything in it. `uv` and any other tools you installed are not affected.

---

## Getting more help

If your issue isn't listed here:

1. **Re-read the error message carefully.** The actual error usually has a hint about what went wrong, even if the surrounding text is intimidating.
2. **Search the error online.** Copy the most-specific line of the error and paste it into a search engine.
3. **Ask the team.** If you got this repository from a colleague, ask them — they've probably hit the same issue.
4. **File an issue.** On the [GitHub issues page](https://github.com/noliathain/nexus-manual-release/issues), with:
   - The command you ran
   - The full error output
   - Your operating system (macOS / Linux / Windows + version)
   - The output of `uv --version` and `python --version`

---

## What's normal vs what's an actual problem

| symptom | normal or problem? |
|---|---|
| First answer takes ~10 seconds | **normal** (model loading) |
| Subsequent answers take ~3-4 seconds | **normal** (CPU inference) |
| Refusal responses appear instantly | **normal** (sub-millisecond by design) |
| Laptop fan spins up during answers | **normal** (CPU at 100% on one core) |
| Some questions get refused | **normal** (the safety architecture working) |
| Browser tabs slow down during demo | **normal** (CPU contention) |
| Demo answers are sometimes shorter than expected | **normal** (the manual section is brief) |
| Demo crashes with a Python traceback | **problem** (file an issue) |
| Demo hangs for 60+ seconds with no output | **problem** (check CPU usage, possibly restart) |
| All answers look identical regardless of question | **problem** (something wrong with retrieval) |
| Test suite fails | **problem** (see test-suite section above) |

If you're in the "problem" column, see "Getting more help" above.
