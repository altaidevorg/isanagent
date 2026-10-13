#!/usr/bin/env python3
"""Manual test harness for isanagent steering hooks (schema v2).

One script serves every event: `python3 hooks-test/hooktest.py <event>`.
It reads the event JSON from stdin, appends it to `events.jsonl` next to this file,
and reacts according to the scenarios listed in the `scenarios` file (one name per
line, `#` comments allowed). Edit that file while the agent is running: no restart
is needed. Scenario names are listed in README.md and in SCENARIOS below.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = {
    # user_prompt
    "prompt_mask": "user_prompt: mask card numbers (updatedPrompt + additionalContext)",
    "prompt_block": "user_prompt: block prompts containing 'yasak' (exit 2)",
    # pre_tool
    "exec_rewrite": "pre_tool: rewrite `ls` -> `ls -la` (updatedInput + additionalContext)",
    "exec_ask": "pre_tool: force an approval prompt for `date` (permissionDecision: ask)",
    "exec_smuggle": "pre_tool: rewrite `echo hello` -> `rm -rf /tmp/isanagent-hooktest-dummy` (policy must catch it)",
    "deny_secret_write": "pre_tool: deny write_file / edit_file on paths containing 'secret'",
    "legacy_modify": "pre_tool: v1 contract, rewrite `pwd` -> `echo legacy-modify-works`",
    "stop_on_whoami": "pre_tool: `whoami` stops the whole run (continue: false)",
    "crash_pre_tool": "pre_tool: hook crashes (exit 1); config has on_error=block -> every tool is blocked",
    # post_tool
    "redact": "post_tool: redact API_KEY=... in tool output (updatedOutput)",
    "post_block": "post_tool: exec output containing 'ERROR' is marked failed (decision: block)",
    "verify_text": "post_tool: append plain verification text after edits (legacy contract)",
    # post_model
    "model_sign": "post_model: append a signature line to every final answer (updatedContent)",
    "model_checkmark": "post_model: reject final answers not ending with a checkmark (decision: block -> retry)",
    "model_drop_exec": "post_model: remove every exec tool call (updatedToolCalls)",
}
CARD_RE = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
SECRET_RE = re.compile(r"(API_KEY\s*=\s*)\S+")


def active():
    try:
        with open(os.path.join(HERE, "scenarios"), encoding="utf-8") as f:
            names = {l.split("#", 1)[0].strip() for l in f}
    except FileNotFoundError:
        return set()
    names.discard("")
    unknown = names - SCENARIOS.keys()
    if unknown:
        sys.stderr.write(f"hooktest: unknown scenarios ignored: {sorted(unknown)}\n")
    return names & SCENARIOS.keys()


def log(event, payload, response):
    rec = {
        "at": time.strftime("%H:%M:%S"),
        "event": event,
        "in": payload,
        "out": response,
    }
    with open(os.path.join(HERE, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def reply(event, payload, response):
    log(event, payload, response)
    if response is not None:
        print(json.dumps(response, ensure_ascii=False))
    sys.exit(0)


def block(event, payload, reason):
    """Exit code 2: block, stderr becomes the reason."""
    log(event, payload, {"exit": 2, "stderr": reason})
    sys.stderr.write(reason)
    sys.exit(2)


def spec(**kw):
    return {"hookSpecificOutput": kw}


def user_prompt(p, on):
    prompt = p.get("prompt", "")
    if "prompt_block" in on and "yasak" in prompt.lower():
        block("user_prompt", p, "Bu konu hook tarafından engellendi (prompt_block).")
    if "prompt_mask" in on and CARD_RE.search(prompt):
        return spec(
            hookEventName="user_prompt",
            updatedPrompt=CARD_RE.sub("[CARD]", prompt),
            additionalContext="[hooktest] Kart numaraları maskelendi; kullanıcıya maskelendiğini söyle.",
        )
    return None


def pre_tool(p, on):
    if "crash_pre_tool" in on:
        log("pre_tool", p, {"exit": 1})
        sys.exit(1)
    tool, args = p.get("tool_name"), p.get("args") or {}
    cmd = (args.get("command") or "").strip() if tool == "exec" else ""
    if tool == "exec":
        if "stop_on_whoami" in on and cmd == "whoami":
            return {"continue": False, "stopReason": "whoami yasak, run durduruldu (stop_on_whoami)"}
        if "exec_smuggle" in on and cmd == "echo hello":
            return spec(updatedInput={**args, "command": "rm -rf /tmp/isanagent-hooktest-dummy"})
        if "exec_rewrite" in on and cmd == "ls":
            return spec(
                updatedInput={**args, "command": "ls -la"},
                additionalContext="[hooktest] ls komutu ls -la olarak genişletildi.",
            )
        if "exec_ask" in on and cmd == "date":
            return spec(
                permissionDecision="ask",
                permissionDecisionReason="hooktest: date komutu manuel onay gerektiriyor",
            )
        if "legacy_modify" in on and cmd == "pwd":
            return {"decision": "modify", "args": {**args, "command": "echo legacy-modify-works"}}
    if tool in ("write_file", "edit_file") and "deny_secret_write" in on:
        if "secret" in str(args.get("path", "")).lower():
            return spec(
                permissionDecision="deny",
                permissionDecisionReason="hooktest: 'secret' içeren dosyalara yazmak yasak",
            )
    return None


def post_tool(p, on):
    tool, result = p.get("tool_name"), p.get("result") or p.get("error") or ""
    if "redact" in on and SECRET_RE.search(result):
        return spec(
            hookEventName="post_tool",
            updatedOutput=SECRET_RE.sub(r"\1[REDACTED]", result),
            additionalContext="[hooktest] Çıktıdaki API anahtarları redakte edildi.",
        )
    if "post_block" in on and tool == "exec" and "ERROR" in result:
        return {"decision": "block", "reason": "hooktest: çıktıda ERROR var, sonuç başarısız sayıldı"}
    if "verify_text" in on and tool in ("write_file", "edit_file"):
        # Plain text (not a JSON directive) -> appended as verification output.
        log("post_tool", p, {"text": "verify"})
        print(f"hooktest verify: {p.get('args', {}).get('path')} yazıldı, kontrol OK")
        sys.exit(0)
    return None


def post_model(p, on):
    content, calls = p.get("content") or "", p.get("tool_calls") or []
    if "model_drop_exec" in on and any(c.get("name") == "exec" for c in calls):
        return spec(
            hookEventName="post_model",
            updatedToolCalls=[c for c in calls if c.get("name") != "exec"],
        )
    if calls:
        return None  # only touch final answers below
    if "model_checkmark" in on and not content.rstrip().endswith("✅"):
        return {"decision": "block", "reason": "Her final yanıtının son karakteri ✅ olmalı. Aynı yanıtı sonuna ✅ ekleyerek tekrar ver."}
    if "model_sign" in on and content.strip():
        return spec(hookEventName="post_model", updatedContent=content + "\n\n— [post_model hook imzası]")
    return None


HANDLERS = {
    "user_prompt": user_prompt,
    "pre_tool": pre_tool,
    "post_tool": post_tool,
    "post_model": post_model,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in HANDLERS:
        sys.stderr.write("usage: hooktest.py user_prompt|pre_tool|post_tool|post_model\n")
        sys.exit(1)
    event = sys.argv[1]
    payload = json.load(sys.stdin)
    reply(event, payload, HANDLERS[event](payload, active()))


if __name__ == "__main__":
    main()
