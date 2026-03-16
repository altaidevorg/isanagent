---
name: skill-creator
description: Create or update local workspace skills under `workspace/skills`.
---

# Skill Creator

Use this skill when the user wants a new reusable workflow or wants to improve an existing local skill.

Rules:
- Create skills under `workspace/skills/<skill-name>/SKILL.md`.
- Every skill must start with YAML frontmatter containing `name` and `description`.
- Keep skill instructions concise and reusable.
- Put only execution-relevant guidance in the skill; avoid extra README-style files unless they are truly needed.
- Prefer clear steps, minimal examples, and direct instructions over long explanations.
