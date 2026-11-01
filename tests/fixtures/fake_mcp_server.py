"""Minimal stdio MCP server for isanagent's MCP client tests (stdlib only).

Misbehaves on purpose: prints a banner to stdout before speaking JSON-RPC (like a library that
logs at import time), answers requests from worker threads (so responses can arrive out of
order), and can ask the client for its roots.
"""
import json
import os
import sys
import threading
import time

print("[Registry] registering 3 tools", flush=True)  # not JSON-RPC: the client must skip it
sys.stderr.write("fake server starting\n")
sys.stderr.flush()

_lock = threading.Lock()
_next_id = [1000]
_waiting = {}
_cancelled = []

TOOLS = [
    {"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "slow", "description": "Sleep", "inputSchema": {"type": "object", "properties": {"secs": {"type": "number"}}}},
    {"name": "roots", "description": "Ask the client for roots", "inputSchema": {"type": "object"}},
    {"name": "env", "description": "Read an env var", "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}}},
    {"name": "cancelled", "description": "Ids cancelled so far", "inputSchema": {"type": "object"}},
    {"name": "crash", "description": "Exit", "inputSchema": {"type": "object"}},
]


def send(msg):
    with _lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def text(t):
    return {"content": [{"type": "text", "text": t}]}


def ask_client(method):
    with _lock:
        _next_id[0] += 1
        rid = _next_id[0]
        ev = threading.Event()
        _waiting[rid] = [ev, None]
    send({"jsonrpc": "2.0", "id": rid, "method": method, "params": {}})
    ev.wait(10)
    return _waiting.pop(rid)[1]


def call(rid, params):
    name = params.get("name")
    args = params.get("arguments") or {}
    if name == "echo":
        return text(args.get("text", ""))
    if name == "slow":
        time.sleep(float(args.get("secs", 1)))
        return text("slept")
    if name == "roots":
        return text(json.dumps(ask_client("roots/list")))
    if name == "env":
        return text(os.environ.get(args.get("key", ""), "<unset>"))
    if name == "cancelled":
        return text(json.dumps(_cancelled))
    if name == "crash":
        sys.stderr.write("fatal: rdkit import failed\n")
        sys.stderr.flush()
        os._exit(3)
    return {"isError": True, "content": [{"type": "text", "text": "unknown tool"}]}


def handle(msg):
    method = msg.get("method")
    rid = msg.get("id")
    if method is None:  # response to our own request
        slot = _waiting.get(rid)
        if slot:
            slot[1] = msg.get("result")
            slot[0].set()
        return
    if rid is None:  # notification
        if method == "notifications/cancelled":
            _cancelled.append(msg["params"]["requestId"])
        return
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "0"}}
    elif method == "tools/list":
        cursor = (msg.get("params") or {}).get("cursor")
        if cursor is None:
            result = {"tools": TOOLS[:3], "nextCursor": "page2"}
        else:
            result = {"tools": TOOLS[3:]}
    elif method == "tools/call":
        result = call(rid, msg.get("params") or {})
    else:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "nope"}})
        return
    send({"jsonrpc": "2.0", "id": rid, "result": result})


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    threading.Thread(target=handle, args=(msg,), daemon=True).start()
