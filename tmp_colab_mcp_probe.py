#!/usr/bin/env python3
"""
Debug/probe script for googlecolab/colab-mcp over stdio MCP.

Primary goal: discover the *correct* sequence to execute Python in Colab MCP,
and document **timeouts** (several independent clocks exist).

## Architecture (upstream googlecolab/colab-mcp)

- Local **FastMCP** stdio process proxies most tools to the **Colab tab** over WebSocket
  after `open_colab_browser_connection` (see upstream `session.py`).
- **`tools/list` after the browser connects** is authoritative; tool names/schemas come
  from the Colab front-end build, not only from the Python package.
- Upstream constants in `colab_mcp/session.py`: `UI_CONNECTION_TIMEOUT = 60.0` (wait for
  user to open/connect Colab), not cell execution time.

## Different “timeouts” (easy to confuse)

1. **MCP host / IDE `timeout` (e.g. mcp.json `30000`)** — milliseconds the *client app*
   waits for a JSON-RPC reply on stdio. Increase if the *server process* is slow to
   answer; this is **not** the same as notebook cell wall clock.

2. **This script’s `--timeout` / per-`request()` timeout** — how long the probe waits
   for each MCP message. Must exceed expected `run_code_cell` duration or you will see
   probe-side timeouts even when Colab is still running.

3. **isanagent `execution_run`** — uses `timeout_secs` on the run (default from
   `[harness.execution] default_execution_timeout_secs`, typically 600) for *both*
   `add_code_cell` and `run_code_cell` waits. Set `timeout_secs` high for training.

4. **isanagent `colab_mcp_tool_call`** — separate wall clock for the raw MCP call;
   **defaults to 120s** if `timeout_secs` is omitted. Error text looks like
   `colab_mcp_tool_call timed out after 120s`. For long `run_code_cell`, pass a larger
   `timeout_secs` in the tool args.

5. **Colab browser tool implementation** — may enforce its own cap (e.g. messages like
   `run_code_cell timed out in 120s` in the tool result). That is **not** configurable
   from the Python `colab-mcp` repo; check the live `inputSchema` for `run_code_cell`
   (`--dump-schemas`) for optional parameters on your Colab build.

## Empirical behavior (`--training-sim`, ~130s cell)

- Expect **one** JSON-RPC `tools/call` **response** for `run_code_cell` when the cell finishes
  (agent blocks until then). Stdio does not carry per-print MCP messages.
- **Interim** MCP messages on stdio during the run are **rare**; you may see
  `notifications/tools/list_changed` (notebook edits), not streaming stdout.
- Structured result often has **one** `stream` stdout object with `text` as a **list** of
  strings (Jupyter-style: one element per `print`), i.e. all lines batched in one payload.

## Probe flow

1) Launch `colab-mcp` (default: `uvx git+https://github.com/googlecolab/colab-mcp`)
2) MCP init + tools/list
3) open_colab_browser_connection
4) tools/list again (often expands after connect)
5) Optional: `--dump-schemas` / `--inspect-only`
6) Try execution strategies in order:
   A) direct execution tool (if present)
   B) add_code_cell -> run_code_cell
   C) get_cells -> pick code cell -> update_cell -> run_code_cell

Run:
  uv run --no-project python tmp_colab_mcp_probe.py --timeout 45
  uv run --no-project python tmp_colab_mcp_probe.py --inspect-only --dump-schemas
  uv run --no-project python tmp_colab_mcp_probe.py --long-run-seconds 130 --request-timeout 400
  uv run --no-project python tmp_colab_mcp_probe.py --training-sim --timeout 90 --request-timeout 350
    # ~130s cell: 13 x (sleep 10s + print). Logs every MCP stdio line that is not the
    # tools/call JSON-RPC response (usually none until completion).
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any


def _now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}")


# Simulated training: 13 steps × 10s sleep ≈ 130s wall, prints each step (like metric logging).
TRAINING_SIM_TOTAL_SECS = 130
TRAINING_SIM_CODE = """import time, sys
# Probe: fake training loop - incremental sleeps + prints (total ~130s)
epochs = 13
step_secs = 10
for epoch in range(1, epochs + 1):
    time.sleep(step_secs)
    loss = 2.5 / epoch + 0.01 * epoch
    acc = min(0.5 + 0.03 * epoch, 0.99)
    print(f"epoch={epoch}/{epochs} loss={loss:.4f} acc={acc:.4f}", flush=True)
print("training_sim_done", flush=True)
"""


def _compact_json_for_log(msg: dict[str, Any], max_len: int = 700) -> str:
    try:
        raw = json.dumps(msg, ensure_ascii=False)
        if len(raw) <= max_len:
            return raw
        return raw[: max_len - 1] + "…"
    except Exception:
        return repr(msg)[:max_len]


def _make_interim_mcp_tap(label: str) -> tuple[Callable[[dict[str, Any]], None], Callable[[], int]]:
    """Tap JSON-RPC messages that arrive while waiting for a matching response (notifications, stray responses)."""
    count = {"n": 0}

    def tap(msg: dict[str, Any]) -> None:
        count["n"] += 1
        log(f"[MCP interim {label} msg#{count['n']}] {_compact_json_for_log(msg)}")

    def get_count() -> int:
        return int(count["n"])

    return tap, get_count


def _analyze_training_cell_output(result_text: str) -> None:
    """Heuristic: did notebook stdout arrive as one blob (batch) vs many lines."""
    n_epoch_lines = result_text.count("epoch=")
    n_done = 1 if "training_sim_done" in result_text else 0
    stream_chunks = 0
    try:
        j = json.loads(result_text)
        outs = j.get("outputs")
        if isinstance(outs, list):
            stream_chunks = sum(
                1
                for o in outs
                if isinstance(o, dict) and o.get("output_type") == "stream" and o.get("name") == "stdout"
            )
            for o in outs:
                if not isinstance(o, dict) or o.get("output_type") != "stream":
                    continue
                t = o.get("text")
                if isinstance(t, list):
                    log(f"output analysis: stdout stream chunk is list of {len(t)} part(s) (Jupyter-style)")
                elif isinstance(t, str):
                    log(f"output analysis: stdout stream chunk is single string, len={len(t)}")
    except json.JSONDecodeError:
        log("output analysis: result is not top-level JSON (using substring counts only)")

    log(
        f"output analysis: substring 'epoch=' occurrences={n_epoch_lines} "
        f"(expect 13 if all prints captured); training_sim_done={n_done}; "
        f"stdout stream output objects={stream_chunks}"
    )
    if n_epoch_lines == 0 and "training_sim_done" not in result_text:
        log("output analysis: no training markers in text — check JSON structure below")
    # One MCP tools/call response typically means one batch of structured content from Colab.
    log(
        "interpretation: Colab MCP returns one JSON-RPC result per run_code_cell when the "
        "cell finishes; this probe's stdio reader only delivers lines as the local colab-mcp "
        "process writes them. Interim MCP notifications during execution are uncommon unless "
        "the server streams them (see [MCP interim run_code_cell] lines above). Notebook stdout "
        "may appear as one stream object with Jupyter-style text[] segments."
    )


class McpClient:
    def __init__(self, process: subprocess.Popen[bytes], read_timeout: float = 30.0):
        self.process = process
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.read_timeout = read_timeout
        self.next_id = 1
        self._inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._reader_stop = threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._reader_stop.set()
        if self.process.poll() is None:
            try:
                self.process.kill()
            except Exception:
                pass

    def _reader_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                msg = self._read_one_message_raw()
                if msg is None:
                    return
                self._inbox.put(msg)
        except Exception as e:
            self._inbox.put({"_reader_error": str(e)})

    def _readline_with_timeout(self, timeout: float) -> bytes:
        q: "queue.Queue[bytes | Exception]" = queue.Queue(maxsize=1)

        def _work() -> None:
            try:
                q.put(self.stdout.readline())
            except Exception as e:
                q.put(e)

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        try:
            item = q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("timed out waiting for MCP header line")
        if isinstance(item, Exception):
            raise item
        return item

    def _read_exact_with_timeout(self, n: int, timeout: float) -> bytes:
        start = time.time()
        out = bytearray()
        while len(out) < n:
            remaining_time = timeout - (time.time() - start)
            if remaining_time <= 0:
                raise TimeoutError("timed out waiting for MCP body")
            q: "queue.Queue[bytes | Exception]" = queue.Queue(maxsize=1)

            def _work() -> None:
                try:
                    chunk = self.stdout.read(n - len(out))
                    q.put(chunk)
                except Exception as e:
                    q.put(e)

            t = threading.Thread(target=_work, daemon=True)
            t.start()
            try:
                item = q.get(timeout=remaining_time)
            except queue.Empty:
                raise TimeoutError("timed out waiting for MCP body chunk")
            if isinstance(item, Exception):
                raise item
            if not item:
                raise RuntimeError("EOF while reading MCP body")
            out.extend(item)
        return bytes(out)

    def _read_one_message_raw(self) -> dict[str, Any] | None:
        while True:
            line = self._readline_with_timeout(self.read_timeout)
            if line == b"":
                return None
            line_s = line.decode("utf-8", errors="replace").strip()
            if line_s == "":
                continue
            # FastMCP stdio commonly sends one JSON message per line.
            try:
                return json.loads(line_s)
            except json.JSONDecodeError:
                pass
            # Compatibility: Content-Length framing.
            lower = line_s.lower()
            if lower.startswith("content-length:"):
                content_length = int(lower.split(":", 1)[1].strip())
                delim = self._readline_with_timeout(self.read_timeout)
                if delim not in (b"\n", b"\r\n", b""):
                    # If server sent extra headers, keep consuming until blank.
                    while delim not in (b"\n", b"\r\n", b""):
                        delim = self._readline_with_timeout(self.read_timeout)
                body = self._read_exact_with_timeout(content_length, self.read_timeout)
                return json.loads(body.decode("utf-8", errors="replace"))

    def _send(self, payload: dict[str, Any]) -> None:
        # FastMCP stdio accepts one JSON payload per line.
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.stdin.write(body)
        self.stdin.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float = 30.0,
        *,
        tap_interim: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        req_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.time() + timeout
        buffered: list[dict[str, Any]] = []
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"timeout waiting for response to {method} (id={req_id})")
            try:
                msg = self._inbox.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"timeout waiting for response to {method} (id={req_id})")
            if "_reader_error" in msg:
                raise RuntimeError(f"MCP reader error: {msg['_reader_error']}")
            if msg.get("id") == req_id:
                for m in buffered:
                    self._inbox.put(m)
                return msg
            if tap_interim is not None:
                tap_interim(msg)
            buffered.append(msg)


def _tool_list_from_response(resp: dict[str, Any]) -> list[dict[str, Any]]:
    result = resp.get("result", {})
    tools = result.get("tools", [])
    return tools if isinstance(tools, list) else []


def _schema_keys(tool: dict[str, Any]) -> list[str]:
    schema = tool.get("inputSchema", {})
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return []
    return [str(k) for k in props.keys()]


def _schema_required(tool: dict[str, Any]) -> list[str]:
    schema = tool.get("inputSchema", {})
    if not isinstance(schema, dict):
        return []
    req = schema.get("required", [])
    if not isinstance(req, list):
        return []
    return [str(x) for x in req]


def _pick_exec_tool(tools: list[dict[str, Any]], explicit: str | None) -> tuple[str, str]:
    names = [t.get("name") for t in tools if isinstance(t.get("name"), str)]
    if explicit:
        if explicit in names:
            return explicit, "code"
        raise RuntimeError(f"explicit execution tool not found: {explicit}; available={names}")

    candidate_order = [
        "execute_python",
        "run_python",
        "run_python_cell",
        "execute_cell",
        "run_code",
    ]
    picked = None
    for c in candidate_order:
        if c in names:
            picked = c
            break
    if picked is None:
        for n in names:
            low = n.lower()
            if ("run" in low or "execute" in low) and any(k in low for k in ("python", "code", "cell")):
                picked = n
                break
    if picked is None:
        raise RuntimeError(f"no execution-like tool found; available={names}")

    code_key = "code"
    for t in tools:
        if t.get("name") != picked:
            continue
        schema = t.get("inputSchema", {})
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for key in ("code", "source", "cell", "input"):
            if isinstance(props, dict) and key in props:
                code_key = key
                return picked, code_key
        if isinstance(props, dict):
            for k in props.keys():
                k_low = str(k).lower()
                if any(x in k_low for x in ("code", "source", "cell", "input")):
                    code_key = str(k)
                    return picked, code_key
    return picked, code_key


def _find_tool(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for t in tools:
        if t.get("name") == name:
            return t
    return None


def _first_present_key(keys: list[str], preferred: tuple[str, ...]) -> str | None:
    keys_set = {k.lower(): k for k in keys}
    for p in preferred:
        if p.lower() in keys_set:
            return keys_set[p.lower()]
    return None


def _extract_cell_id_any(resp: dict[str, Any]) -> str | None:
    result = resp.get("result", {})
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        # Colab FE returns newCellId from add_code_cell (matches isanagent extract_cell_id_from_tool_result).
        for key in ("newCellId", "cellId", "cell_id", "id"):
            v = structured.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    text = _extract_text_content(resp)
    markers = ['"newCellId":"', '"cellId":"', '"cell_id":"', '"id":"']
    for m in markers:
        idx = text.find(m)
        if idx >= 0:
            rest = text[idx + len(m) :]
            end = rest.find('"')
            if end > 0:
                return rest[:end]
    return None


def _extract_text_content(tool_call_resp: dict[str, Any]) -> str:
    result = tool_call_resp.get("result", {})
    content = result.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        if chunks:
            return "\n".join(chunks)
    structured = result.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _extract_structured(tool_call_resp: dict[str, Any]) -> Any:
    result = tool_call_resp.get("result", {})
    return result.get("structuredContent")


def _collect_timeoutish_schema_paths(node: Any, path: str = "$") -> list[str]:
    """JSON-schema walk: paths whose key names suggest execution/MCP timeouts."""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            kl = str(k).lower()
            p = f"{path}.{k}"
            if "timeout" in kl or "deadline" in kl or kl.endswith("_ms") and "time" in kl:
                out.append(p)
            out.extend(_collect_timeoutish_schema_paths(v, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_collect_timeoutish_schema_paths(v, f"{path}[{i}]"))
    return out


def _tools_schemas_blob(tools: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in tools:
        n = t.get("name")
        if isinstance(n, str):
            out[n] = t.get("inputSchema")
    return out


def _log_schema_timeout_hints(tools: list[dict[str, Any]]) -> None:
    for t in tools:
        n = t.get("name")
        schema = t.get("inputSchema")
        if not isinstance(n, str) or not isinstance(schema, dict):
            continue
        hints = _collect_timeoutish_schema_paths(schema)
        if hints:
            log(f"schema timeout-ish paths for {n}: {hints[:40]}{' …' if len(hints) > 40 else ''}")


def _spawn_stderr_reader(proc: subprocess.Popen[bytes], buf: list[str]) -> threading.Thread:
    def _work() -> None:
        err = proc.stderr
        if err is None:
            return
        try:
            for raw in iter(err.readline, b""):
                if not raw:
                    break
                buf.append(raw.decode("utf-8", errors="replace"))
        except Exception:
            pass

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="uvx", help="MCP launcher command (default: uvx)")
    ap.add_argument(
        "--args",
        nargs="*",
        default=["git+https://github.com/googlecolab/colab-mcp"],
        help="Args passed to launcher command",
    )
    ap.add_argument("--timeout", type=float, default=45.0, help="Default per-request timeout (seconds)")
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Minimum MCP tools/call wait (seconds); defaults to max(--timeout, 60). "
        "Raise for long run_code_cell so this probe does not abort before Colab responds.",
    )
    ap.add_argument(
        "--post-connect-wait",
        type=float,
        default=4.0,
        help="Seconds to sleep after reconnect tools/list before execution attempts",
    )
    ap.add_argument(
        "--dump-schemas",
        action="store_true",
        help="Print full inputSchema JSON per tool (post-connect list) and timeout-ish key paths",
    )
    ap.add_argument(
        "--schemas-out",
        default=None,
        help="If set, write pretty JSON schemas blob to this file (in addition to stdout when --dump-schemas)",
    )
    ap.add_argument(
        "--inspect-only",
        action="store_true",
        help="After connect + tools/list + optional schema dump, exit 0 without running code",
    )
    ap.add_argument(
        "--long-run-seconds",
        type=int,
        default=0,
        help="If >0, notebook probe runs `time.sleep(N)` in the cell to test Colab-side vs client timeouts",
    )
    ap.add_argument(
        "--training-sim",
        action="store_true",
        help=(
            "Run a ~130s fake training loop in Colab (13×10s sleep + loss/acc prints). "
            "Raises MCP client timeouts automatically. Enables MCP interim message tap on run_code_cell."
        ),
    )
    ap.add_argument(
        "--tap-mcp-during-run",
        action="store_true",
        help="Log any MCP stdio JSON-RPC messages received while waiting for run_code_cell (besides the final response)",
    )
    ap.add_argument(
        "--execution-tool",
        default=None,
        help="Explicit MCP execution tool name (skip auto-detect)",
    )
    args = ap.parse_args()

    long_pad = 180.0
    effective_cell_wall = 0.0
    if args.training_sim:
        effective_cell_wall = float(TRAINING_SIM_TOTAL_SECS)
    elif args.long_run_seconds > 0:
        effective_cell_wall = float(args.long_run_seconds)

    req_floor = args.request_timeout if args.request_timeout is not None else max(args.timeout, 60.0)
    if effective_cell_wall > 0:
        req_floor = max(req_floor, effective_cell_wall + long_pad)
    # Stdio reader must tolerate a single hung line until the JSON-RPC result arrives.
    mcp_read_timeout = max(args.timeout, req_floor)

    cmd = [args.command, *args.args]
    log(f"starting MCP process: {' '.join(cmd)}")
    log(
        f"MCP read/request ceiling: {mcp_read_timeout}s "
        f"(training_sim={args.training_sim}, long_run_seconds={args.long_run_seconds}, "
        f"effective_cell_wall={effective_cell_wall}, --request-timeout={args.request_timeout})"
    )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None or proc.stdout is None or proc.stderr is None:
        print("failed to attach stdio pipes", file=sys.stderr)
        return 2

    stderr_buf: list[str] = []
    _spawn_stderr_reader(proc, stderr_buf)

    client = McpClient(proc, read_timeout=mcp_read_timeout)
    try:
        init = client.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "isanagent-colab-debug", "version": "0.1.0"},
            },
            timeout=args.timeout,
        )
        if "error" in init:
            raise RuntimeError(f"initialize error: {json.dumps(init['error'])}")
        log("initialize OK")
        log(f"server initialize result: {json.dumps(init.get('result', {}), ensure_ascii=False)[:800]}")

        client.notify("notifications/initialized", {})
        log("sent notifications/initialized")

        resp = client.request("tools/list", {}, timeout=args.timeout)
        if "error" in resp:
            raise RuntimeError(f"tools/list error: {json.dumps(resp['error'])}")
        tools = _tool_list_from_response(resp)
        names = [t.get("name") for t in tools if isinstance(t.get("name"), str)]
        log(f"tools/list returned {len(names)} tools: {names}")
        for t in tools:
            n = t.get("name")
            if isinstance(n, str):
                log(
                    f"tool {n} schema keys: {_schema_keys(t)} required: {_schema_required(t)}"
                )

        connect_tool = "open_colab_browser_connection"
        if connect_tool in names:
            log(f"calling {connect_tool}()")
            # Upstream waits up to ~60s for the browser session.
            c_resp = client.request(
                "tools/call",
                {"name": connect_tool, "arguments": {}},
                timeout=max(args.timeout, 65.0),
            )
            if "error" in c_resp:
                log(f"{connect_tool} returned error: {json.dumps(c_resp['error'])}")
            else:
                text = _extract_text_content(c_resp)
                log(f"{connect_tool} result: {text}")
        else:
            log(f"{connect_tool} not present in current tool list")

        # Re-list after connect attempt (colab-mcp can hot-change tool list)
        time.sleep(1.5)
        resp2 = client.request("tools/list", {}, timeout=args.timeout)
        tools2 = _tool_list_from_response(resp2)
        names2 = [t.get("name") for t in tools2 if isinstance(t.get("name"), str)]
        log(f"tools/list (after connect attempt) -> {len(names2)} tools: {names2}")
        for t in tools2:
            n = t.get("name")
            if isinstance(n, str):
                log(
                    f"tool {n} schema keys: {_schema_keys(t)} required: {_schema_required(t)}"
                )

        _log_schema_timeout_hints(tools2)
        if args.dump_schemas or args.schemas_out:
            blob = _tools_schemas_blob(tools2)
            pretty = json.dumps(blob, ensure_ascii=False, indent=2)
            if args.schemas_out:
                with open(args.schemas_out, "w", encoding="utf-8") as fo:
                    fo.write(pretty)
                log(f"wrote tool inputSchemas to {args.schemas_out!r}")
            if args.dump_schemas:
                print("=== tool inputSchema (post-connect tools/list) ===")
                print(pretty)

        if args.inspect_only:
            log("inspect-only: skipping execution strategies")
            return 0

        # Give browser-side plugin a bit more time after first connect.
        log(f"waiting {args.post_connect_wait}s for browser-side readiness before execution calls...")
        time.sleep(args.post_connect_wait)

        if args.training_sim:
            code = TRAINING_SIM_CODE
        elif args.long_run_seconds > 0:
            n = args.long_run_seconds
            code = (
                "import time, sys\n"
                f"print('probe_long_run start sleep {n}s', flush=True)\n"
                f"time.sleep({n})\n"
                "print('probe_long_run end', flush=True)\n"
            )
        else:
            code = "print('hello from colab mcp probe')"

        call_tc = req_floor
        tap_run_code_cell = args.training_sim or args.tap_mcp_during_run

        # Strategy A: direct execution if tool supports code-like arg.
        if args.training_sim or args.long_run_seconds > 0:
            log("[A] skipped: long notebook cell probe uses add_code_cell -> run_code_cell")
        else:
            try:
                exec_tool, code_key = _pick_exec_tool(tools2, args.execution_tool)
                if code_key.lower() not in ("cellid", "cell_id", "id"):
                    log(f"[A] direct run via {exec_tool!r} key={code_key!r}")
                    run_resp = client.request(
                        "tools/call",
                        {"name": exec_tool, "arguments": {code_key: code}},
                        timeout=call_tc,
                    )
                    if "error" in run_resp:
                        log(f"[A] direct run error: {json.dumps(run_resp['error'])}")
                    else:
                        log("[A] direct run success")
                        print(_extract_text_content(run_resp))
                        return 0
                else:
                    log(f"[A] skipped direct run because {exec_tool} expects cell id")
            except Exception as e:
                log(f"[A] direct strategy failed: {e!r}")

        # Strategy B: add_code_cell -> run_code_cell
        try:
            add_tool = _find_tool(tools2, "add_code_cell")
            run_tool = _find_tool(tools2, "run_code_cell")
            if add_tool and run_tool:
                add_key = _first_present_key(_schema_keys(add_tool), ("code", "source", "content", "text"))
                run_key = _first_present_key(_schema_keys(run_tool), ("cellId", "cell_id", "id"))
                if add_key and run_key:
                    log(f"[B] add_code_cell({add_key}) -> run_code_cell({run_key})")
                    add_args = {add_key: code, "language": "python", "cellIndex": 0}
                    log(f"[B] add_code_cell args={add_args}")
                    add_resp = client.request(
                        "tools/call",
                        {"name": "add_code_cell", "arguments": add_args},
                        timeout=call_tc,
                    )
                    if "error" in add_resp:
                        log(f"[B] add_code_cell error: {json.dumps(add_resp['error'])}")
                    else:
                        cell_id = _extract_cell_id_any(add_resp)
                        log(f"[B] add_code_cell response text: {_extract_text_content(add_resp)}")
                        if cell_id:
                            req_kw: dict[str, Any] = {}
                            tap_fn, tap_cnt_get = _make_interim_mcp_tap("run_code_cell")
                            if tap_run_code_cell:
                                req_kw["tap_interim"] = tap_fn
                            hint = f"~{TRAINING_SIM_TOTAL_SECS}s cell" if args.training_sim else "cell"
                            log(
                                f"[B] run_code_cell start ({hint}, MCP client timeout={call_tc}s, "
                                f"tap_interim={tap_run_code_cell})"
                            )
                            t_cell = time.time()
                            run_resp = client.request(
                                "tools/call",
                                {"name": "run_code_cell", "arguments": {run_key: cell_id}},
                                timeout=call_tc,
                                **req_kw,
                            )
                            elapsed = time.time() - t_cell
                            interim_n = tap_cnt_get() if tap_run_code_cell else 0
                            log(f"[B] run_code_cell returned after {elapsed:.1f}s; interim MCP msgs: {interim_n}")
                            if "error" in run_resp:
                                log(f"[B] run_code_cell error: {json.dumps(run_resp['error'])}")
                            else:
                                log("[B] run_code_cell success")
                                out_txt = _extract_text_content(run_resp)
                                print(out_txt)
                                if args.training_sim:
                                    _analyze_training_cell_output(out_txt)
                                return 0
                        else:
                            log("[B] no cell id could be extracted from add_code_cell")
                else:
                    log("[B] missing add/run schema keys for notebook-cell strategy")
            else:
                log("[B] notebook add/run tools not present")
        except Exception as e:
            log(f"[B] notebook add/run strategy failed: {e!r}")

        # Strategy C: get_cells -> update_cell -> run_code_cell
        try:
            get_tool = _find_tool(tools2, "get_cells")
            update_tool = _find_tool(tools2, "update_cell")
            run_tool = _find_tool(tools2, "run_code_cell")
            if get_tool and update_tool and run_tool:
                update_cell_id_key = _first_present_key(_schema_keys(update_tool), ("cellId", "cell_id", "id"))
                update_content_key = _first_present_key(_schema_keys(update_tool), ("content", "code", "source", "text"))
                run_key = _first_present_key(_schema_keys(run_tool), ("cellId", "cell_id", "id"))
                if update_cell_id_key and update_content_key and run_key:
                    log(f"[C] get_cells -> update_cell({update_content_key}) -> run_code_cell")
                    get_args = {"includeOutputs": False, "cellIndexStart": 0, "cellIndexEnd": 50}
                    log(f"[C] get_cells args={get_args}")
                    get_resp = client.request(
                        "tools/call",
                        {"name": "get_cells", "arguments": get_args},
                        timeout=call_tc,
                    )
                    if "error" in get_resp:
                        log(f"[C] get_cells error: {json.dumps(get_resp['error'])}")
                    else:
                        structured = _extract_structured(get_resp)
                        candidate_id = None
                        if isinstance(structured, dict):
                            cells = structured.get("cells")
                            if isinstance(cells, list):
                                for c in cells:
                                    if not isinstance(c, dict):
                                        continue
                                    cid = c.get("id") or c.get("cellId")
                                    ctype = c.get("cellType") or c.get("type")
                                    if isinstance(cid, str) and isinstance(ctype, str) and "code" in ctype.lower():
                                        candidate_id = cid
                                        break
                        if not candidate_id:
                            candidate_id = _extract_cell_id_any(get_resp)
                        log(f"[C] selected cell id: {candidate_id!r}")
                        if candidate_id:
                            upd_resp = client.request(
                                "tools/call",
                                {
                                    "name": "update_cell",
                                    "arguments": {
                                        update_cell_id_key: candidate_id,
                                        update_content_key: code,
                                    },
                                },
                                timeout=call_tc,
                            )
                            if "error" in upd_resp:
                                log(f"[C] update_cell error: {json.dumps(upd_resp['error'])}")
                            else:
                                req_kw2: dict[str, Any] = {}
                                tap_fn2, tap_cnt_get2 = _make_interim_mcp_tap("run_code_cell")
                                if tap_run_code_cell:
                                    req_kw2["tap_interim"] = tap_fn2
                                log(f"[C] run_code_cell start (MCP timeout={call_tc}s, tap_interim={tap_run_code_cell})")
                                t_c = time.time()
                                run_resp = client.request(
                                    "tools/call",
                                    {"name": "run_code_cell", "arguments": {run_key: candidate_id}},
                                    timeout=call_tc,
                                    **req_kw2,
                                )
                                log(
                                    f"[C] run_code_cell returned after {time.time() - t_c:.1f}s; "
                                    f"interim MCP msgs: {tap_cnt_get2() if tap_run_code_cell else 0}"
                                )
                                if "error" in run_resp:
                                    log(f"[C] run_code_cell error: {json.dumps(run_resp['error'])}")
                                else:
                                    log("[C] update+run success")
                                    out_c = _extract_text_content(run_resp)
                                    print(out_c)
                                    if args.training_sim:
                                        _analyze_training_cell_output(out_c)
                                    return 0
                        else:
                            log("[C] could not extract any code cell id")
                else:
                    log("[C] missing schema keys for get/update/run strategy")
            else:
                log("[C] get/update/run tools not present")
        except Exception as e:
            log(f"[C] get/update/run strategy failed: {e!r}")

        log("all strategies failed")
        return 4
    except Exception as e:
        log(f"probe failed: {e!r}")
        if stderr_buf:
            tail = "".join(stderr_buf)[-6000:]
            log("recent colab-mcp stderr:")
            print(tail, file=sys.stderr)
        else:
            try:
                err = proc.stderr.read1(8192) if hasattr(proc.stderr, "read1") else proc.stderr.read(8192)
                if err:
                    log("stderr from colab-mcp:")
                    print(err.decode("utf-8", errors="replace"), file=sys.stderr)
            except Exception:
                pass
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

