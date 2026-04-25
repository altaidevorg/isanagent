#!/usr/bin/env python3
"""
Debug/probe script for googlecolab/colab-mcp over stdio MCP.

Primary goal: discover the *correct* sequence to execute Python in Colab MCP.

Flow:
1) Launch `colab-mcp` (default: `uvx git+https://github.com/googlecolab/colab-mcp`)
2) MCP init + tools/list
3) open_colab_browser_connection
4) tools/list again
5) Try execution strategies in order:
   A) direct execution tool (if present)
   B) add_code_cell -> run_code_cell
   C) get_cells -> pick code cell -> update_cell -> run_code_cell

Run:
  uv run --no-project python tmp_colab_mcp_probe.py --timeout 45
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from typing import Any


def _now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}")


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

    def request(self, method: str, params: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
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
        for key in ("cellId", "cell_id", "id"):
            v = structured.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    text = _extract_text_content(resp)
    markers = ['"cellId":"', '"cell_id":"', '"id":"']
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="uvx", help="MCP launcher command (default: uvx)")
    ap.add_argument(
        "--args",
        nargs="*",
        default=["git+https://github.com/googlecolab/colab-mcp"],
        help="Args passed to launcher command",
    )
    ap.add_argument("--timeout", type=float, default=45.0, help="Per-request timeout seconds")
    ap.add_argument(
        "--execution-tool",
        default=None,
        help="Explicit MCP execution tool name (skip auto-detect)",
    )
    args = ap.parse_args()

    cmd = [args.command, *args.args]
    log(f"starting MCP process: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None or proc.stdout is None or proc.stderr is None:
        print("failed to attach stdio pipes", file=sys.stderr)
        return 2

    client = McpClient(proc, read_timeout=args.timeout)
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
            c_resp = client.request("tools/call", {"name": connect_tool, "arguments": {}}, timeout=args.timeout)
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

        # Give browser-side plugin a bit more time after first connect.
        log("waiting 4s for browser-side readiness before execution calls...")
        time.sleep(4)

        code = "print('hello from colab mcp probe')"

        # Strategy A: direct execution if tool supports code-like arg.
        try:
            exec_tool, code_key = _pick_exec_tool(tools2, args.execution_tool)
            if code_key.lower() not in ("cellid", "cell_id", "id"):
                log(f"[A] direct run via {exec_tool!r} key={code_key!r}")
                run_resp = client.request(
                    "tools/call",
                    {"name": exec_tool, "arguments": {code_key: code}},
                    timeout=max(args.timeout, 60.0),
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
                        timeout=max(args.timeout, 60.0),
                    )
                    if "error" in add_resp:
                        log(f"[B] add_code_cell error: {json.dumps(add_resp['error'])}")
                    else:
                        cell_id = _extract_cell_id_any(add_resp)
                        log(f"[B] add_code_cell response text: {_extract_text_content(add_resp)}")
                        if cell_id:
                            run_resp = client.request(
                                "tools/call",
                                {"name": "run_code_cell", "arguments": {run_key: cell_id}},
                                timeout=max(args.timeout, 60.0),
                            )
                            if "error" in run_resp:
                                log(f"[B] run_code_cell error: {json.dumps(run_resp['error'])}")
                            else:
                                log("[B] run_code_cell success")
                                print(_extract_text_content(run_resp))
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
                        timeout=max(args.timeout, 60.0),
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
                                timeout=max(args.timeout, 60.0),
                            )
                            if "error" in upd_resp:
                                log(f"[C] update_cell error: {json.dumps(upd_resp['error'])}")
                            else:
                                run_resp = client.request(
                                    "tools/call",
                                    {"name": "run_code_cell", "arguments": {run_key: candidate_id}},
                                    timeout=max(args.timeout, 60.0),
                                )
                                if "error" in run_resp:
                                    log(f"[C] run_code_cell error: {json.dumps(run_resp['error'])}")
                                else:
                                    log("[C] update+run success")
                                    print(_extract_text_content(run_resp))
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
        try:
            err = proc.stderr.read1(8192) if hasattr(proc.stderr, "read1") else proc.stderr.read(8192)
            if err:
                log("stderr from colab-mcp:")
                print(err.decode("utf-8", errors="replace"))
        except Exception:
            pass
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

