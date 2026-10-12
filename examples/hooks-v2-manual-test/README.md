# Hooks v2: manual test kit

A single Python hook (`hooktest.py`) handles all four steering events. It logs every event to `events.jsonl` and reacts according to the scenarios listed in the `scenarios` file. You can change the active scenarios while the agent is running. No restart needed.

## Setup (once)

```bash
# 1. A throwaway workspace, so your real ~/.isanagent is untouched
cargo run --release -- onboard --interactive --workspace ~/.isanagent-hooktest

# 2. Install the kit (copies hooktest.py, appends hooks.toml to config.toml, keeps a backup)
bash examples/hooks-v2-manual-test/setup.sh ~/.isanagent-hooktest

# 3. Run the agent (terminal UI)
cargo run --release -- --workspace ~/.isanagent-hooktest

# 4. In a second terminal, watch what the hooks receive and return
tail -f ~/.isanagent-hooktest/workspace/hooks-test/events.jsonl

# 5. In a third, watch the audit trail (HookDecision events: what each hook changed)
tail -f ~/.isanagent-hooktest/workspace/hooks-test/audit.jsonl
```

`setup.sh` can be run again at any time. It refreshes the script, the example plugin and its own config block. It never overwrites the `scenarios` file.

To switch scenarios, edit `~/.isanagent-hooktest/workspace/hooks-test/scenarios`. Put one name per line. Several can be active at once.

If a scenario does nothing, check `events.jsonl` first. The hook only reacts to the exact command shown in the table. If the model ran something else (e.g. `ls .` instead of `ls`), ask it to "run exactly `ls`".

## Scenarios

| Scenario | Say to the agent | Expected |
|---|---|---|
| `prompt_mask` | `Kartım 4111 1111 1111 1111, numarayı aynen tekrar et` | The model only sees `[CARD]` and says the number was masked. In `events.jsonl`, `in.prompt` holds the original text and `out.updatedPrompt` the masked one. |
| `prompt_block` | `yasak bir şey sor` | The turn ends with "Bu konu hook tarafından engellendi". The model is never called. |
| `exec_rewrite` | `exec ile tam olarak ls çalıştır` | `ls -la` runs (hidden files, permissions). The model mentions the hook note / rewritten args. |
| `exec_ask` | `exec ile date çalıştır` | An approval prompt appears even though `date` is harmless, ending with "A policy hook requested review: …". Approve → it runs. Deny → it's skipped. |
| `exec_smuggle` | `exec ile echo hello çalıştır` | **Security check.** The hook rewrites the command to `rm -rf /tmp/isanagent-hooktest-dummy`. The approval prompt must show the **rm** command, not `echo hello`. Deny it. Before the fix, it would have run without a prompt. |
| `deny_secret_write` | `my_secret.txt dosyasına "x" yaz` | Blocked: "'secret' içeren dosyalara yazmak yasak". No edit prompt. |
| `legacy_modify` | `exec ile pwd çalıştır` | Output is `legacy-modify-works` (the v1 `decision: modify` contract still works). |
| `stop_on_whoami` | `exec ile whoami çalıştır` | The whole run stops with "Stopped by hook: whoami yasak…". |
| `crash_pre_tool` | anything that uses a tool | The hook exits 1 and `on_error = "block"` → every tool call fails with "pre_tool hook failed and on_error=block". Remove the scenario afterwards. |
| `redact` | first `echo 'API_KEY=sk-test-123' > ~/.isanagent-hooktest/workspace/creds.txt`, then `creds.txt dosyasını oku` | The model sees `API_KEY=[REDACTED]`. The real key is only in `events.jsonl` `in.result`. |
| `post_block` | `exec ile echo ERROR test çalıştır` | The tool result becomes an error: "[post-tool hook rejected this result] …". |
| `verify_text` | `notes.txt dosyasına "merhaba" yaz` (approve the edit) | The result ends with "[post-tool hook] hooktest verify: notes.txt yazıldı, kontrol OK" (plain text output is appended as before). |
| `model_sign` | `merhaba` | Every final answer ends with "— [post_model hook imzası]". |
| `model_checkmark` | `merhaba` | The first answer is rejected. The model gets the feedback and answers again ending with ✅. If it fails 3 times in a row, the turn ends with an error. |
| `model_drop_exec` | `exec ile date çalıştır` | The exec call is removed before it runs. In `events.jsonl`, `out.updatedToolCalls` has no `exec` entry. |

## Plugin hooks and audit trail

The kit also installs an example plugin, `hooktest-guard`, under `.agents/plugins/`. Its `dev.altai.isanagent/hooks.json` uses the **Claude Code format** (`PreToolUse` → `{"type": "command"}`) and finds its own script through `${CLAUDE_PLUGIN_ROOT}`. It is always active, whatever the `scenarios` file says.

| Test | Do | Expected |
|---|---|---|
| Plugin blocks | `exec ile curl https://example.com çalıştır` | Blocked: "hooktest-guard plugin: network commands (curl/wget) are blocked". In `audit.jsonl` the record has `"source": "plugin:hooktest-guard"` and `"decision": "block"`. |
| Allowlist | Delete the `plugin_hooks = [...]` line from `config.toml` and restart | curl runs again. At startup the log says the plugin "ships hooks … but is not listed in plugin_hooks; skipped". |
| Audit of config hooks | Turn on `exec_rewrite` and run `ls` | `audit.jsonl` gets `"source": "config"`, `"decision": "modify"`, `"changed": ["args","context"]`. |
| Audit of failures | Turn on `crash_pre_tool` and use any tool | `"decision": "error_blocked"` with the error text. |

Only hook runs that change something are recorded. A hook that answers "no opinion" leaves no line.

## Automated tests

```bash
cargo test --lib hooks                    # engine, plugin hooks.json, audit trail
cargo test --lib hook_dispatch_tests      # dispatch order / ask / block
cargo test --lib post_model_tool_call_tests
```

## Cleanup

Restore `config.toml.bak-hooktest`, or delete the `~/.isanagent-hooktest` folder.
