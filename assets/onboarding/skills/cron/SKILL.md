---
name: cron
description: Use the built-in cron tool to schedule one-time, recurring, or cron-based follow-up actions.
---

# Cron

Use this skill when the user wants a follow-up action to happen later, repeatedly, or on a fixed schedule.

Guidelines:
- Use `every_seconds` for simple repeating jobs.
- Use `at` for one-time scheduled execution.
- Use `cron_expr` for calendar-style recurring schedules.
- Always use the exact timezone offset from the runtime context when scheduling with `at`.
- Choose stable, descriptive job ids so the same job can be updated or removed later.
- Remove obsolete jobs explicitly when the user asks to stop or replace an existing schedule.
