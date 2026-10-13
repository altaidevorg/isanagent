#!/bin/sh
# Blocks exec commands that use curl / wget (exit 2 = block, stderr = reason).
payload=$(cat)
case "$payload" in
  *'"command":"curl'*|*'"command":"wget'*|*'"command": "curl'*|*'"command": "wget'*)
    echo "hooktest-guard plugin: network commands (curl/wget) are blocked" >&2
    exit 2 ;;
esac
exit 0
