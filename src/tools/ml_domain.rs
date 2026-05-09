//! arXiv and Hugging Face Hub read-only helpers for ML engineering workflows.

use async_trait::async_trait;
use reqwest::header::AUTHORIZATION;
use serde_json::Value;

use crate::traits::Tool;

const HF_USER_AGENT: &str =
    "isanagent-ml-domain/0.1 (https://github.com/huggingface; research tool)";

fn hf_path_safe(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("path must be non-empty".to_string());
    }
    if path.contains("..") {
        return Err("path must not contain '..'".to_string());
    }
    if path.starts_with('/') {
        return Err("path must be relative (no leading slash)".to_string());
    }
    Ok(())
}

/// Search arXiv via the public Atom API (`export.arxiv.org`).
pub struct ArxivSearchTool {
    pub max_output_chars: usize,
}

#[async_trait]
impl Tool for ArxivSearchTool {
    fn name(&self) -> &str {
        "arxiv_search"
    }

    fn description(&self) -> &str {
        "Search arXiv (papers) via the public API. Discovery tool: returns Atom/XML snippets (titles, ids, summaries) so you can shortlist candidates. Follow up with `arxiv_fetch` to read details before citing claims. Prefer precise English keywords."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search terms (e.g. 'DPO preference optimization')" },
                "max_results": { "type": "integer", "description": "1–30 (default 10)" }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing query")?;
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(10)
            .clamp(1, 30);

        let q = urlencoding::encode(query);
        let url = format!(
            "https://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}",
            q, max_results
        );

        let client = reqwest::Client::builder()
            .user_agent(HF_USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;

        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("arxiv_search request: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("arxiv_search HTTP {}", resp.status()));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| format!("arxiv_search body: {}", e))?;

        let mut out = String::new();
        let entry_re = regex::Regex::new(r"(?s)<entry>.*?</entry>").map_err(|e| e.to_string())?;
        let id_re = regex::Regex::new(r"(?s)<id>(.*?)</id>").map_err(|e| e.to_string())?;
        let title_re = regex::Regex::new(r"(?s)<title>(.*?)</title>").map_err(|e| e.to_string())?;
        let summary_re =
            regex::Regex::new(r"(?s)<summary>(.*?)</summary>").map_err(|e| e.to_string())?;

        for entry_match in entry_re.find_iter(&body) {
            let entry = entry_match.as_str();
            let id = id_re
                .captures(entry)
                .and_then(|c| c.get(1))
                .map_or("", |m| m.as_str())
                .trim();
            let title = title_re
                .captures(entry)
                .and_then(|c| c.get(1))
                .map_or("", |m| m.as_str())
                .trim()
                .replace('\n', " ");
            let summary = summary_re
                .captures(entry)
                .and_then(|c| c.get(1))
                .map_or("", |m| m.as_str())
                .trim()
                .replace('\n', " ");

            let id_clean = id.split('/').next_back().unwrap_or(id);

            out.push_str(&format!(
                "ID: {}\nTitle: {}\nSummary: {}\n\n---\n",
                id_clean, title, summary
            ));
        }

        if out.is_empty() {
            out = "No results found.".to_string();
        }

        crate::utils::truncate_utf8_safe(&mut out, self.max_output_chars, "\n... [TRUNCATED]");
        Ok(out)
    }
}

/// Fetch one arXiv abstract page (abs HTML) by id.
pub struct ArxivFetchTool {
    pub workspace_dir: std::path::PathBuf,
}

#[async_trait]
impl Tool for ArxivFetchTool {
    fn name(&self) -> &str {
        "arxiv_fetch"
    }

    fn description(&self) -> &str {
        "Fetch an arXiv paper by id (e.g. `2401.0001` or `cs.CL/0001001`). Returns Markdown text (truncated). Use after `arxiv_search` for downloading full text and cross-verification; do not rely on search snippets alone."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "arxiv_id": { "type": "string", "description": "arXiv paper id" }
            },
            "required": ["arxiv_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let id = args
            .get("arxiv_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing arxiv_id")?
            .trim()
            .replace(' ', "");
        if id.is_empty() || id.contains("..") {
            return Err("invalid arxiv_id".to_string());
        }

        let arxiv2md_url = format!("https://arxiv2md.org/api/markdown?url={}", id);

        let client = reqwest::Client::builder()
            .user_agent(HF_USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;

        let arxiv2md_resp = client.get(&arxiv2md_url).send().await;

        let mut html_markdown_content = String::new();
        if let Ok(resp) = arxiv2md_resp {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    if text.lines().count() >= 30 {
                        html_markdown_content = text;
                    }
                }
            }
        }

        let full_content = if html_markdown_content.is_empty() {
            let pdf_url = format!("https://arxiv.org/pdf/{}.pdf", id);
            let pdf_resp = client
                .get(&pdf_url)
                .send()
                .await
                .map_err(|e| format!("arxiv_fetch pdf request: {}", e))?;

            if !pdf_resp.status().is_success() {
                return Err(format!(
                    "arxiv_fetch HTTP {} (PDF not found for {})",
                    pdf_resp.status(),
                    id
                ));
            }

            let pdf_bytes = pdf_resp
                .bytes()
                .await
                .map_err(|e| format!("arxiv_fetch pdf body: {}", e))?
                .to_vec();

            crate::utils::extract_markdown_from_pdf_bytes(&pdf_bytes)?
        } else {
            html_markdown_content
        };

        let downloads_dir = self
            .workspace_dir
            .join("workspace")
            .join("downloads")
            .join("arxiv");
        let _ = tokio::fs::create_dir_all(&downloads_dir).await;
        let file_path = downloads_dir.join(format!("{id}.md"));
        tokio::fs::write(&file_path, &full_content)
            .await
            .map_err(|e| e.to_string())?;

        let total_lines = full_content.lines().count();
        let max_preview_lines = 50;

        if total_lines <= max_preview_lines {
            Ok(format!(
                "{full_content}\n\n---\nFull paper ({total_lines} lines) saved to `{}`.",
                file_path.display()
            ))
        } else {
            let preview: String = full_content
                .lines()
                .take(max_preview_lines)
                .collect::<Vec<_>>()
                .join("\n");

            Ok(format!(
                "{preview}\n\n---\n[TRUNCATED] Showing first {max_preview_lines} of {total_lines} lines. \
                Full content saved to `{}`. Use `read_file` with `start_line` and `end_line` to read \
                the rest, or `search_text` to find specific information.",
                file_path.display()
            ))
        }
    }
}

/// GET a file from the Hugging Face Hub `resolve` URL (read-only). Uses `HF_TOKEN` when set for private repos.
pub struct HfHubFileFetchTool {
    pub max_output_chars: usize,
}

#[async_trait]
impl Tool for HfHubFileFetchTool {
    fn name(&self) -> &str {
        "hf_hub_file_fetch"
    }

    fn description(&self) -> &str {
        "Download a **single text file** from the Hugging Face Hub via the public `resolve` URL (e.g. config.json, README). Pass `repo` like `org/model`, optional `revision` (branch/commit, default `main`), and relative `path`. Requires network; respects `HF_TOKEN` for gated models. Not for huge binaries—use `web_fetch` on the raw URL if needed."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "repo": { "type": "string", "description": "Hub repo id, e.g. `meta-llama/Llama-2-7b-hf`" },
                "path": { "type": "string", "description": "Repo-relative path, e.g. `config.json` or `README.md`" },
                "revision": { "type": "string", "description": "Git revision (default `main`)" }
            },
            "required": ["repo", "path"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let repo = args
            .get("repo")
            .and_then(|v| v.as_str())
            .ok_or("Missing repo")?
            .trim();
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing path")?
            .trim();
        hf_path_safe(path)?;

        let revision = args
            .get("revision")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("main");

        if repo.is_empty() || repo.contains("..") || repo.starts_with('/') {
            return Err("invalid repo id".to_string());
        }

        let enc_path = path
            .split('/')
            .map(urlencoding::encode)
            .collect::<Vec<_>>()
            .join("/");
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            repo, revision, enc_path
        );

        let client = reqwest::Client::builder()
            .user_agent(HF_USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;

        let mut req = client.get(&url);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            let t = token.trim();
            if !t.is_empty() {
                req = req.header(AUTHORIZATION, format!("Bearer {}", t));
            }
        }

        let resp = req
            .send()
            .await
            .map_err(|e| format!("hf_hub_file_fetch request: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "hf_hub_file_fetch HTTP {} for {}",
                resp.status(),
                url
            ));
        }

        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ctype.contains("application/octet-stream")
            && !path.ends_with(".json")
            && !path.ends_with(".md")
            && !path.ends_with(".txt")
            && !path.ends_with(".py")
            && !path.ends_with(".toml")
            && !path.ends_with(".yaml")
            && !path.ends_with(".yml")
        {
            return Err(
                "Refusing to dump large binary: use a text path (json/md/txt/py) or open the file in the browser."
                    .to_string(),
            );
        }

        let mut body = resp
            .text()
            .await
            .map_err(|e| format!("hf_hub_file_fetch body: {}", e))?;
        crate::utils::truncate_utf8_safe(&mut body, self.max_output_chars, "\n... [TRUNCATED]");
        Ok(body)
    }
}
