#!/usr/bin/env bash
# setup.sh — one-shot setup for the oripic bot.
# Creates .venv, installs requirements, warns about missing config.
#
# Usage:
#     chmod +x setup.sh     # first time only
#     ./setup.sh

set -euo pipefail

# Prefer python3 (standard on modern Linux/macOS); fall back to `python`
# (common on Windows/Git Bash where there's no `python3` alias).
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "error: no python interpreter found on PATH" >&2
    exit 1
fi

echo "using interpreter: $("$PY" --version 2>&1) ($(command -v "$PY"))"

# Create venv if it doesn't exist; otherwise reuse so reruns are cheap.
if [ ! -d .venv ]; then
    echo "creating .venv..."
    "$PY" -m venv .venv
else
    echo ".venv already exists, reusing"
fi

# Pick the venv's python directly — works without activating the venv,
# and activation differs across shells (source vs .ps1 vs .bat).
# Linux/macOS layout first, then Windows (Git Bash / MSYS) layout.
if [ -x .venv/bin/python ]; then
    VENV_PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
    VENV_PY=.venv/Scripts/python.exe
else
    echo "error: couldn't find the venv's python executable" >&2
    exit 1
fi

echo "upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip --quiet

echo "installing requirements..."
"$VENV_PY" -m pip install -r requirements.txt

# On first run, create an empty .env so the user has somewhere to put their
# token. We don't prefill a placeholder because load_dotenv would then see
# `DISCORD_TOKEN=` as a real (empty) value and mask a clearer error.
if [ ! -f .env ]; then
    touch .env
    echo
    echo "    created an empty .env in this directory."
    echo "    Open it and add token"
    echo
    echo "        DISCORD_TOKEN=token"

fi

echo
echo "setup complete."
echo
echo "to activate the venv:"
echo "    source .venv/bin/activate        # mac/linux/git bash"
echo "    .venv\\Scripts\\Activate.ps1       # windows powershell"
echo
echo "to run the bot (no activation needed):"
echo "    $VENV_PY bot.py"
