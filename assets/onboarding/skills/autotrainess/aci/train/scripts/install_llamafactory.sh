#!/usr/bin/env bash
set -euo pipefail

# Adapted from AutoTrainess (MIT) — https://github.com/simple-agent-lab/AutoTrainess

# Reuse an existing installation if the CLI is already available.
if command -v llamafactory-cli >/dev/null 2>&1; then
  echo "LlamaFactory already available: $(command -v llamafactory-cli)"
  exit 0
fi

if ! command -v python >/dev/null 2>&1; then
  echo "No python interpreter found for installing llamafactory" >&2
  exit 1
fi

python -m pip install -U llamafactory

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo 'llamafactory-cli is still unavailable after installation.' >&2
  exit 1
fi

llamafactory-cli version >/dev/null
