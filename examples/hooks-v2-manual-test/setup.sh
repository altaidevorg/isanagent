#!/usr/bin/env bash
# Install (or refresh) the hooks v2 manual-test kit in an isanagent workspace.
# Usage: bash setup.sh [workspace]   (default: ~/.isanagent-hooktest)
set -euo pipefail
WS="${1:-$HOME/.isanagent-hooktest}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="$WS/config.toml"
BEGIN="# --- isanagent hooks v2 manual test"
END="# --- end hooks v2 manual test ---"

if [ ! -f "$CFG" ]; then
  echo "No $CFG yet. Create the workspace first:"
  echo "  cargo run --release -- onboard --interactive --workspace $WS"
  exit 1
fi

# Hook script + scenarios
mkdir -p "$WS/workspace/hooks-test"
cp "$HERE/hooktest.py" "$WS/workspace/hooks-test/hooktest.py"
[ -f "$WS/workspace/hooks-test/scenarios" ] || cp "$HERE/scenarios" "$WS/workspace/hooks-test/scenarios"
: > "$WS/workspace/hooks-test/events.jsonl"

# Example plugin with a Claude Code-style hooks.json
mkdir -p "$WS/.agents/plugins"
rm -rf "$WS/.agents/plugins/hooktest-guard"
cp -R "$HERE/plugin/hooktest-guard" "$WS/.agents/plugins/hooktest-guard"

# Config: replace a previous kit block, refuse to touch a hand-written one.
if grep -qF "$BEGIN" "$CFG"; then
  cp "$CFG" "$CFG.bak-hooktest"
  awk -v b="$BEGIN" -v e="$END" '
    index($0, b) == 1 { skip = 1; next }
    skip && index($0, e) == 1 { skip = 0; next }
    !skip { print }' "$CFG.bak-hooktest" > "$CFG"
elif grep -qE '^\[harness\.hooks\.(steering|observation)\]' "$CFG"; then
  echo "config.toml already has its own [harness.hooks.*] tables; not touching it."
  echo "Merge $HERE/hooks.toml by hand."
  exit 1
else
  cp "$CFG" "$CFG.bak-hooktest"
fi
{ echo; cat "$HERE/hooks.toml"; } >> "$CFG"
echo "Hook config written to $CFG (backup: $CFG.bak-hooktest)."

echo
echo "Scenarios file: $WS/workspace/hooks-test/scenarios"
echo "Hook I/O log:   $WS/workspace/hooks-test/events.jsonl"
echo "Audit trail:    $WS/workspace/hooks-test/audit.jsonl"
echo "Run the agent:  cargo run --release -- --workspace $WS"
