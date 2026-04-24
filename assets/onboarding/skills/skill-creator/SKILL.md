---
name: skill-creator
description: Author or refine workspace-local skills (SKILL.md) so future turns load concise, agent-oriented procedures.
---

# Skill creator (isanagent workspace skills)

Use this when you need a **durable procedure** (checklist, tool order, project conventions) that should apply across sessions without pasting the same instructions every time.

## Where to write

- Path pattern: **`workspace/skills/<skill-name>/SKILL.md`** relative to the onboarded workspace root (same tree the **`load_skill_instructions`** tool indexes).
- One skill = one directory = one **`SKILL.md`**. Avoid scattering extra markdown unless it is truly shared assets.

## Frontmatter (required)

Every **`SKILL.md`** must start with YAML:

- **`name`**: short identifier, usually matches the directory.
- **`description`**: one line, optimized for **skill search**—describe *when another model should load this* using concrete triggers (tools, file types, error symptoms), not marketing language.

Optional keys (`requires`, `always`, …) follow the same conventions as built-in templates—omit unknown keys.

## Body: write for another model

- Use **imperative** steps (**“Call … then …”**), not documentation addressed to operators.
- Name **isanagent tools exactly** as they appear in your tool list (`execution_run`, `glob_files`, …).
- Keep skills **short and skimmable**; link or cite repo docs for long policy (sandbox rules, harness config).
- Put **execution-relevant** guardrails here (ordering, caps, anti-patterns). Do not duplicate entire `AGENTS.md`—reference it for global behavior.
- Prefer **decision tables** or bullet checklists over prose essays.

## Editing flow

1. Read the existing **`SKILL.md`** if present; minimally extend rather than rewriting unrelated sections.
2. Save; no compile step—skills are picked up on next load/search.
3. If the skill is niche, set **`always: false`** (default) so it only loads when search matches.
