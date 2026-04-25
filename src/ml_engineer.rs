//! Static ML-engineer policy appended when `[harness.ml_engineer] enabled = true`.
//!
//! `isanagent onboard` also writes `workspace/ML_ENGINEER_OVERLAY.md` with the same text plus a
//! short preamble so operators can read or grep it without opening the repo.

/// Workspace-level overlay (HF ml-intern–informed policy). Kept in `assets/` for easy editing.
pub const HARNESS_OVERLAY: &str = include_str!("../assets/ml_engineer_overlay.md");

/// Extra system instructions for sub-agent runs when ML harness is on (research-style behavior).
pub const SUBAGENT_RESEARCH_APPEND: &str = r#"

--- Sub-agent (research / delegation) ---
You are a focused sub-task runner. Prefer structured outputs: bullet findings, cited URLs or arXiv IDs, and explicit unknowns.
Default deep-research flow:
1) discovery (`web_search` / `arxiv_search`) to shortlist candidates,
2) primary-source reading (`web_fetch` / `arxiv_fetch` / `hf_hub_file_fetch`),
3) contradiction check across at least two sources,
4) synthesis with confidence level and open questions.
Extract methodology-relevant details (datasets, metrics, hyperparameters, compute setup) for literature or recipe-style tasks.
If evidence is weak, say so and propose a verification experiment. Do not spawn nested sub-agents.
"#;
