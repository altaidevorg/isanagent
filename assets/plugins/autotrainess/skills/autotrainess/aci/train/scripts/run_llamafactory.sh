#!/usr/bin/env bash
set -euo pipefail

# Adapted from AutoTrainess (MIT) — https://github.com/simple-agent-lab/AutoTrainess

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <llamafactory-config.yaml> [extra args...]" >&2
  exit 1
fi

CONFIG_PATH="$1"
shift || true

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "LlamaFactory config not found: $CONFIG_PATH" >&2
  exit 1
fi

if command -v llamafactory-cli >/dev/null 2>&1; then
  DISABLE_VERSION_CHECK=1 exec llamafactory-cli train "$CONFIG_PATH" "$@"
fi

if command -v python >/dev/null 2>&1; then
  if python -m llamafactory.cli train -h >/dev/null 2>&1; then
    DISABLE_VERSION_CHECK=1 exec python -m llamafactory.cli train "$CONFIG_PATH" "$@"
  fi
fi

echo "LlamaFactory is not available in the current environment." >&2
echo "Run skills/autotrainess/aci/train/scripts/install_llamafactory.sh before launching training." >&2
exit 1
