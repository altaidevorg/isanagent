---
name: cron
description: Schedule deferred or recurring work via the isanagent cron tool—add/remove jobs with correct runtime context fields.
---

# Cron (isanagent `cron` tool)

Use this when the task requires **you** to register, update, or tear down a scheduled follow‑up message to the **same** or a known chat/channel.

## Tool contract

- Tool name: **`cron`**.
- Actions: **`add`** | **`remove`**. (There is no **`list`** yet—do not assume you can enumerate jobs.)
- **`add`** requires: **`action`**, **`chat_id`**, **`channel`**, **`message`**, plus **exactly one** schedule among **`every_seconds`**, **`at`**, **`cron_expr`**.
- **`remove`** requires: **`action`**, **`job_id`**, **`chat_id`**, **`channel`**.

Always copy **`chat_id`** and **`channel`** from the **RUNTIME CONTEXT** block you were given—do not invent them.

## Schedule choice

- **`every_seconds`**: simple periodic nudges. **Forbidden** when the deployment has **`[multi_tenant_edge].cron_scheduling_enabled = true`**—the tool will error; use **`at`** or **`cron_expr`** instead.
- **`at`**: one shot at an ISO datetime. You **must** include the correct **numeric timezone offset** from runtime context (e.g. `2026-04-24T15:00:00+03:00`). Do **not** default to `Z` unless context is UTC.
- **`cron_expr`**: recurring. With multi‑tenant edge cron enabled, use a **6‑field UTC** expression (`second minute hour day month weekday`). Otherwise follow the host’s 7‑field cron rules.

## Hygiene

- **`list`** allows you to view all currently active cron jobs across all chats and channels. Use this when you need to verify if a job is running or retrieve a lost `job_id`.
- **`add`** assigns the **`job_id`** (returned in the success string, e.g. `Successfully scheduled job abcdef12 …`). Store it in your reasoning if you may need **`remove`** later; you cannot choose the id yourself today.
- When the user asks to stop a schedule, call **`remove`** with that exact **`job_id`**.
- Write **`message`** so a future turn knows what to do when the job fires—include intent, constraints, and any ids/paths the follow‑up needs.
