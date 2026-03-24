use async_trait::async_trait;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::JinaWebBackend;
use crate::traits::Tool;
use crate::NodeHandle;

/// Resolves a path against the workspace and enforces boundary restrictions.
pub fn resolve_path(path: &str, workspace_dir: &Path, restrict: bool) -> Result<PathBuf, String> {
    // 1. Expand naive relativity to the workspace dir.
    let base_path = Path::new(path);
    let resolved = if base_path.is_absolute() {
        base_path.to_path_buf()
    } else {
        workspace_dir.join(base_path)
    };

    // 2. Canonicalize if it exists to cleanly remove `..` and `.`.
    // If it doesn't exist yet (e.g., writing a new file), we canonicalize the nearest existing parent
    // and append the remainder.
    let canonical = if resolved.exists() {
        std::fs::canonicalize(&resolved).map_err(|e| format!("Path normalization error: {}", e))?
    } else {
        // Find nearest existing parent
        let mut parent = resolved.parent();
        let mut missing_components = Vec::new();
        while let Some(p) = parent {
            if p.exists() {
                break;
            }
            missing_components.push(p.file_name().unwrap_or_default());
            parent = p.parent();
        }
        
        let mut safe_base = std::fs::canonicalize(parent.unwrap_or_else(|| Path::new(".")))
            .map_err(|e| format!("Base path normalization error: {}", e))?;
            
        for comp in missing_components.into_iter().rev() {
            safe_base.push(comp);
        }
        if let Some(name) = resolved.file_name() {
            safe_base.push(name);
        }
        safe_base
    };

    // 3. Enforce sandbox boundary if restricted.
    if restrict {
        let canonical_workspace = std::fs::canonicalize(workspace_dir)
            .map_err(|e| format!("Workspace normalization error: {}", e))?;
            
        if !canonical.starts_with(&canonical_workspace) {
            return Err(format!(
                "PermissionError: Path {:?} is outside allowed workspace directory {:?}",
                canonical, canonical_workspace
            ));
        }
    }

    Ok(canonical)
}

pub struct ReadFileTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a local file. Provide the absolute or relative path to the file."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args.get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'path' argument")?;
            
        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;
        
        fs::read_to_string(&actual_path).map_err(|e| e.to_string())
    }
}

pub struct WriteFileTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a local file. Be careful, this will overwrite the file if it exists."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write into the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args.get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        
        let content = args.get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content' argument")?;
            
        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;
        
        if let Some(parent) = actual_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create parent directories: {}", e))?;
        }
            
        fs::write(&actual_path, content)
            .map(|_| format!("Successfully wrote to {}", actual_path.display()))
            .map_err(|e| e.to_string())
    }
}

pub struct EditFileTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing old_text with new_text. The old_text must match exactly."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace it with"
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args.get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
            
        let old_text = args.get("old_text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'old_text' argument")?;
            
        let new_text = args.get("new_text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'new_text' argument")?;

        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;
        
        let content = fs::read_to_string(&actual_path)
            .map_err(|e| format!("Error reading file: {}", e))?;

        if !content.contains(old_text) {
            return Ok(format!("Error: old_text not found in {:?}", actual_path.display()));
        }

        let count = content.matches(old_text).count();
        if count > 1 {
            return Ok(format!("Error: old_text appears {} times. Please provide more context to make it unique.", count));
        }

        let new_content = content.replacen(old_text, new_text, 1);
        fs::write(&actual_path, new_content)
            .map(|_| format!("Successfully edited {}", actual_path.display()))
            .map_err(|e| format!("Error saving edits: {}", e))
    }
}

pub struct ListDirTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List the contents of a directory."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args.get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
            
        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;
        
        if !actual_path.is_dir() {
            return Ok(format!("Error: Not a directory: {:?}", actual_path.display()));
        }

        let mut entries = match fs::read_dir(&actual_path) {
            Ok(iter) => iter,
            Err(e) => return Ok(format!("Error reading dir: {}", e)),
        };

        let mut items = Vec::new();
        while let Some(Ok(entry)) = entries.next() {
            let metadata = entry.metadata().map_err(|e| e.to_string())?;
            let prefix = if metadata.is_dir() { "📁" } else { "📄" };
            items.push(format!("{} {}", prefix, entry.file_name().to_string_lossy()));
        }

        items.sort();

        if items.is_empty() {
            return Ok(format!("Directory {:?} is empty", actual_path.display()));
        }

        Ok(items.join("\n"))
    }
}

pub struct ShellExecTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

impl ShellExecTool {
    fn check_safety_guards(command: &str) -> Result<(), String> {
        let lower_cmd = command.to_lowercase();
        // Mimic Nanobot destructive safety guards
        let blocked_patterns = [
            "rm -rf", "rm -fr", "del /f", "del /q", "rmdir /s",
            "format ", "mkfs", "diskpart", "dd if=", "> /dev/sd",
            "shutdown", "reboot", "poweroff", ":(){ :|:& };:"
        ];

        for pattern in blocked_patterns.iter() {
            if lower_cmd.contains(pattern) {
                return Err(format!("Command blocked by safety guard (detected dangerous pattern: {})", pattern));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for ShellExecTool {
    fn name(&self) -> &str {
        "exec"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output. Use with caution. Bounded by a 60 second timeout."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional relative working directory for the command"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let command = args.get("command")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'command' argument")?;
            
        Self::check_safety_guards(command)?;

        let cwd_str = args.get("working_dir")
            .and_then(|v| v.as_str())
            .unwrap_or(".");
            
        let actual_dir = resolve_path(cwd_str, &self.workspace_dir, self.restrict_to_workspace)?;

        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = tokio::process::Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(command);
            c
        };
        
        cmd.current_dir(actual_dir);

        let child = cmd.output();
        
        match tokio::time::timeout(std::time::Duration::from_secs(60), child).await {
            Ok(Ok(output)) => {
                let mut result = String::new();
                let stdout = String::from_utf8_lossy(&output.stdout);
                if !stdout.trim().is_empty() {
                    result.push_str(&stdout);
                }
                
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.trim().is_empty() {
                    if !result.is_empty() { result.push_str("\nSTDERR:\n"); }
                    result.push_str(&stderr);
                }
                
                if !output.status.success() {
                    result.push_str(&format!("\nExit code: {}", output.status.code().unwrap_or(-1)));
                }
                
                if result.is_empty() {
                    Ok("(no output)".to_string())
                } else {
                    // Truncate if massive
                    if result.len() > 10000 {
                        Ok(format!("{}\n... (truncated, {} more chars)", &result[..10000], result.len() - 10000))
                    } else {
                        Ok(result)
                    }
                }
            },
            Ok(Err(e)) => Err(format!("Failed to execute command: {}", e)),
            Err(_) => Err("Command timed out after 60 seconds".to_string()),
        }
    }
}

fn web_http_client(timeout_secs: u64) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0")
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}

fn parse_scraper_selector(sel: &str) -> Result<scraper::Selector, String> {
    scraper::Selector::parse(sel).map_err(|e| format!("Invalid CSS selector {:?}: {}", sel, e))
}

fn truncate_web_output(text: String, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text;
    }
    let mut end = max_chars;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n... (truncated, {} more chars)",
        &text[..end],
        text.len() - end
    )
}

fn apply_jina_bearer(
    req: reqwest::RequestBuilder,
    jina: Option<&JinaWebBackend>,
) -> reqwest::RequestBuilder {
    if let Some(j) = jina {
        if let Some(key) = j.api_key.as_deref() {
            return req.header("Authorization", format!("Bearer {}", key));
        }
    }
    req
}

/// DuckDuckGo `/html/` often blocks scrapers; `/lite/` via POST is more reliable.
async fn web_search_duckduckgo(query: &str, max_output_chars: usize) -> Result<String, String> {
    let url = "https://lite.duckduckgo.com/lite/";
    let client = web_http_client(45)?;

    let res = client
        .post(url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!("q={}", urlencoding::encode(query)))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let body = res.text().await.map_err(|e| e.to_string())?;

    let document = scraper::Html::parse_document(&body);
    let title_selector = parse_scraper_selector(".result-link")?;
    let snippet_selector = parse_scraper_selector(".result-snippet")?;

    let mut results = String::new();

    let titles: Vec<_> = document.select(&title_selector).take(5).collect();
    let snippets: Vec<_> = document.select(&snippet_selector).take(5).collect();

    for (i, (title_elem, snippet_elem)) in titles.into_iter().zip(snippets.into_iter()).enumerate() {
        let title = title_elem.text().collect::<Vec<_>>().join(" ");
        let link = title_elem.value().attr("href").unwrap_or("");
        let snippet = snippet_elem.text().collect::<Vec<_>>().join(" ").trim().to_string();

        results.push_str(&format!("{}. [{}]({})\n   {}\n\n", i + 1, title, link, snippet));
    }

    if results.is_empty() {
        return Ok("No results found.".to_string());
    }

    Ok(truncate_web_output(results, max_output_chars))
}

/// [Jina Search](https://s.jina.ai/) — useful when DuckDuckGo is unreachable from the host.
async fn web_search_jina(
    query: &str,
    jina: &JinaWebBackend,
    max_output_chars: usize,
) -> Result<String, String> {
    let url = format!("https://s.jina.ai/{}", urlencoding::encode(query));
    let client = web_http_client(45)?;
    let req = apply_jina_bearer(client.get(&url), Some(jina));
    let res = req.send().await.map_err(|e| e.to_string())?;
    if !res.status().is_success() {
        return Err(format!("Jina search HTTP error: {}", res.status()));
    }
    let body = res.text().await.map_err(|e| e.to_string())?;
    if body.trim().is_empty() {
        return Ok("No results found.".to_string());
    }
    Ok(truncate_web_output(body, max_output_chars))
}

async fn web_fetch_direct(url: &str, max_output_chars: usize) -> Result<String, String> {
    let client = web_http_client(30)?;

    let res = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch URL: {}", e))?;

    if !res.status().is_success() {
        return Err(format!("HTTP Error: {}", res.status()));
    }

    let content_type = res
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("application/json") {
        let json_body: Value = res.json().await.map_err(|e| format!("Invalid JSON: {}", e))?;
        let s = serde_json::to_string_pretty(&json_body).unwrap_or_default();
        return Ok(truncate_web_output(s, max_output_chars));
    }

    let body = res
        .text()
        .await
        .map_err(|e| format!("Failed to decode text: {}", e))?;

    let document = scraper::Html::parse_document(&body);
    let mut text_output = String::new();

    // Heuristic HTML→text for direct fetches: skip non-content tags (scripts, chrome, SVG) and
    // treat block-level tags as line breaks / light markdown markers. Not configurable; Jina path
    // avoids this entirely.
    let elements_to_ignore = ["script", "style", "noscript", "svg", "nav", "footer", "header"];
    let block_elements = [
        "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br",
    ];

    let body_selector = parse_scraper_selector("body")?;
    if let Some(body_node) = document.select(&body_selector).next() {
        for node in body_node.descendants() {
            if let scraper::Node::Element(elem) = node.value() {
                let tag = elem.name();

                if elements_to_ignore.contains(&tag) {
                    continue;
                }
                if block_elements.contains(&tag) {
                    text_output.push('\n');
                    if tag.starts_with('h') {
                        text_output.push_str("### ");
                    }
                    if tag == "li" {
                        text_output.push_str("- ");
                    }
                }
            } else if let scraper::Node::Text(text_node) = node.value() {
                let text = text_node.trim();
                if !text.is_empty() {
                    let mut ignore = false;
                    let mut parent = node.parent();
                    while let Some(p) = parent {
                        if let scraper::Node::Element(e) = p.value() {
                            if elements_to_ignore.contains(&e.name()) {
                                ignore = true;
                                break;
                            }
                        }
                        parent = p.parent();
                    }

                    if !ignore {
                        text_output.push_str(text);
                        text_output.push(' ');
                    }
                }
            }
        }
    }

    let cleaned = text_output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(truncate_web_output(cleaned, max_output_chars))
}

/// [Jina Reader](https://r.jina.ai/) returns LLM-friendly markdown for a target URL.
async fn web_fetch_jina(
    url: &str,
    jina: &JinaWebBackend,
    max_output_chars: usize,
) -> Result<String, String> {
    // Jina Reader expects the target URL as a path suffix with `:` and `/` intact — do not
    // percent-encode the whole URL or the service cannot resolve the target.
    let reader_url = format!("https://r.jina.ai/{}", url.trim());
    let client = web_http_client(60)?;
    let req = apply_jina_bearer(client.get(&reader_url), Some(jina));
    let res = req
        .send()
        .await
        .map_err(|e| format!("Failed to fetch URL via Jina: {}", e))?;
    if !res.status().is_success() {
        return Err(format!("HTTP Error (Jina reader): {}", res.status()));
    }

    let content_type = res
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("application/json") {
        let json_body: Value = res.json().await.map_err(|e| format!("Invalid JSON: {}", e))?;
        let s = serde_json::to_string_pretty(&json_body).unwrap_or_default();
        return Ok(truncate_web_output(s, max_output_chars));
    }

    let body = res
        .text()
        .await
        .map_err(|e| format!("Failed to decode text: {}", e))?;
    Ok(truncate_web_output(body, max_output_chars))
}

pub struct WebSearchTool {
    /// When `Some`, use [Jina Search](https://s.jina.ai/) (`[jina].enabled` in config).
    pub jina: Option<JinaWebBackend>,
    /// From `max_web_tool_output_chars` in config (see `AppConfig::effective_max_web_tool_output_chars`).
    pub max_output_chars: usize,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web. Uses Jina (s.jina.ai) when [jina].enabled is true in config; otherwise DuckDuckGo Lite."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query' argument")?;

        if let Some(ref jina) = self.jina {
            web_search_jina(query, jina, self.max_output_chars).await
        } else {
            web_search_duckduckgo(query, self.max_output_chars).await
        }
    }
}

pub struct WebFetchTool {
    /// When `Some`, use [Jina Reader](https://r.jina.ai/) (`[jina].enabled` in config).
    pub jina: Option<JinaWebBackend>,
    /// From `max_web_tool_output_chars` in config (see `AppConfig::effective_max_web_tool_output_chars`).
    pub max_output_chars: usize,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL. Uses Jina Reader (r.jina.ai) when [jina].enabled is true; otherwise direct GET with HTML text extraction or JSON pretty-print."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'url' argument")?;

        if let Some(ref jina) = self.jina {
            web_fetch_jina(url, jina, self.max_output_chars).await
        } else {
            web_fetch_direct(url, self.max_output_chars).await
        }
    }
}

pub struct CronTool {
    pub cron_node: NodeHandle<String>,
}

#[async_trait]
impl Tool for CronTool {
    fn name(&self) -> &str {
        "cron"
    }

    fn description(&self) -> &str {
        "Manage scheduled tasks. Supports repeating intervals, exact times, or standard cron expressions."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform: 'add' or 'remove'. 'list' is not yet supported."
                },
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job. Required only for 'remove' action."
                },
                "message": {
                    "type": "string",
                    "description": "The message to send back to you when triggered. Required for 'add' action."
                },
                "chat_id": {
                    "type": "string",
                    "description": "The target chat ID. You must extract this explicitly from the RUNTIME CONTEXT block."
                },
                "channel": {
                    "type": "string",
                    "description": "The target channel (e.g., 'terminal', 'slack'). Extract this from the RUNTIME CONTEXT block."
                },
                "every_seconds": {
                    "type": "integer",
                    "description": "Execute repeatedly every N seconds. Mutually exclusive with 'at' and 'cron_expr'."
                },
                "at": {
                    "type": "string",
                    "description": "Execute once at a specific ISO datetime. You MUST include the exact correct timezone offset from your RUNTIME CONTEXT (e.g. 2026-03-04T13:45:53+03:00, NOT ending in Z unless you are in UTC). Mutually exclusive with 'every_seconds' and 'cron_expr'."
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Execute using a 7-part cron string. Mutually exclusive with 'every_seconds' and 'at'."
                }
            },
            "required": ["action", "chat_id", "channel"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("add");

        if action == "remove" {
            let job_id = args.get("job_id").and_then(|v| v.as_str()).ok_or("Missing 'job_id' for remove action")?;
            let cmd = crate::scheduler::CronCommand::Remove { id: job_id.to_string() };
            let json_str = serde_json::to_string(&cmd).map_err(|e| e.to_string())?;
            self.cron_node.send_packet(json_str).await.map_err(|e| e.to_string())?;
            return Ok(format!("Requested removal of job {}", job_id));
        }

        if action == "add" {
            let message = args.get("message").and_then(|v| v.as_str()).ok_or("Missing 'message' for add action")?;
            let chat_id = args.get("chat_id").and_then(|v| v.as_str()).ok_or("Missing 'chat_id' for add action")?;
            let channel = args.get("channel").and_then(|v| v.as_str()).ok_or("Missing 'channel' for add action")?;
            let id = uuid::Uuid::new_v4().to_string()[..8].to_string();

            let schedule = if let Some(secs) = args.get("every_seconds").and_then(|v| v.as_i64()) {
                crate::scheduler::ScheduleKind::Every { every_ms: secs * 1000 }
            } else if let Some(at) = args.get("at").and_then(|v| v.as_str()) {
                let dt = chrono::DateTime::parse_from_rfc3339(at).map_err(|_| "Invalid ISO format for 'at'. Make sure you include the proper UTC offset as provided in context.")?;
                crate::scheduler::ScheduleKind::At { at_ms: dt.timestamp_millis() }
            } else if let Some(expr) = args.get("cron_expr").and_then(|v| v.as_str()) {
                crate::scheduler::ScheduleKind::Cron { cron_expr: expr.to_string() }
            } else {
                return Err("Must provide one of 'every_seconds', 'at', or 'cron_expr' for add action".to_string());
            };

            let cmd = crate::scheduler::CronCommand::Add {
                id: id.clone(),
                schedule,
                message: message.to_string(),
                chat_id: chat_id.to_string(),
                channel: channel.to_string(),
            };

            let json_str = serde_json::to_string(&cmd).map_err(|e| e.to_string())?;
            self.cron_node.send_packet(json_str).await.map_err(|e| e.to_string())?;
            return Ok(format!("Successfully scheduled job {} with action '{}'", id, message));
        }

        Err(format!("Unknown action '{}'", action))
    }
}

/// Message Tool: allows the agent to asynchronously emit proactive status messages
/// directly to the user/channel before the primary generation loop completes.
pub struct MessageTool {
    pub outbound_tx: tokio::sync::mpsc::Sender<crate::bus::BusMessage>,
}

#[async_trait]
impl Tool for MessageTool {
    fn name(&self) -> &str {
        "message"
    }

    fn description(&self) -> &str {
        "Send a message to the user asynchronously. Use this to provide proactive updates or intermediate results while working on long multi-step tasks."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Target channel (e.g., terminal, slack, email)."
                },
                "chat_id": {
                    "type": "string",
                    "description": "Target chat/user ID."
                },
                "thread_id": {
                    "type": "string",
                    "description": "Target thread ID if applicable."
                }
            },
            "required": ["content", "channel", "chat_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let content = args.get("content").and_then(|v| v.as_str()).ok_or("Missing 'content'")?;
        let channel = args.get("channel").and_then(|v| v.as_str()).ok_or("Missing 'channel'")?;
        let chat_id = args.get("chat_id").and_then(|v| v.as_str()).ok_or("Missing 'chat_id'")?;
        let thread_id = args.get("thread_id").and_then(|v| v.as_str()).map(|s| s.to_string());

        let msg = crate::bus::BusMessage::Outbound(crate::bus::OutboundMessage {
            channel: channel.to_string(),
            chat_id: chat_id.to_string(),
            thread_id,
            content: content.to_string(),
            metadata: std::collections::HashMap::new(),
        });

        match self.outbound_tx.send(msg).await {
            Ok(_) => Ok(format!("Message sent to {}:{}", channel, chat_id)),
            Err(e) => Err(format!("Failed to send message: {}", e)),
        }
    }
}

pub struct SearchMemoryTool {
    pub memory_node: NodeHandle<crate::memory::MemoryMessage>,
}

#[async_trait]
impl Tool for SearchMemoryTool {
    fn name(&self) -> &str {
        "search_memory"
    }

    fn description(&self) -> &str {
        "Search your long-term and short-term memory (session summaries) for past context, facts, or keywords."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let query = args.get("query").and_then(|v| v.as_str()).ok_or("Missing 'query'")?;
        
        // Use oneshot channel to await the reply from the MemoryActor
        let (tx, rx) = tokio::sync::oneshot::channel();
        let msg = crate::memory::MemoryMessage::SearchSummaries {
            query: query.to_string(),
            reply: crate::memory::SharedReply::new(tx),
        };
        
        self.memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
        
        let results = rx.await.map_err(|_| "Memory Actor Channel Closed")?.map_err(|e| e)?;
        
        if results.is_empty() {
            Ok(format!("No memory results found for '{}'.", query))
        } else {
            Ok(format!("Memory Search Results:\n\n{}", results.join("\n\n---\n\n")))
        }
    }
}

pub struct FetchMemoryByDateTool {
    pub memory_node: NodeHandle<crate::memory::MemoryMessage>,
}

#[async_trait]
impl Tool for FetchMemoryByDateTool {
    fn name(&self) -> &str {
        "fetch_memory_by_date"
    }

    fn description(&self) -> &str {
        "Fetch long-term and short-term memory (session summaries) from a specific relative time range, like the last 7 days."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "days_ago": {
                    "type": "integer",
                    "description": "Number of days in the past to search from. For example, 7 means 'within the last 7 days'."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of summaries to return."
                }
            },
            "required": ["days_ago", "limit"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let days_ago = args.get("days_ago").and_then(|v| v.as_u64()).ok_or("Missing or invalid 'days_ago'")?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        
        let (tx, rx) = tokio::sync::oneshot::channel();
        let msg = crate::memory::MemoryMessage::FetchSummariesByTimeRange {
            days_ago,
            limit,
            reply: crate::memory::SharedReply::new(tx),
        };
        
        self.memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
        
        let results = rx.await.map_err(|_| "Memory Actor Channel Closed")?.map_err(|e| e)?;
        
        if results.is_empty() {
            Ok(format!("No memory results found in the last {} days.", days_ago))
        } else {
            Ok(format!("Memory Results (Last {} days):\n\n{}", days_ago, results.join("\n\n---\n\n")))
        }
    }
}

