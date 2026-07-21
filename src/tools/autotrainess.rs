//! AutoTrainess experiment ledger tools — project bootstrap and iteration logging.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::AppConfig;
use crate::traits::Tool;
use crate::utils::resolve_path;

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectMeta {
    schema_version: u32,
    project_id: String,
    stage: String,
    base_model: Option<String>,
    benchmark: Option<String>,
    target_metric: Option<String>,
    best_iteration_id: Option<String>,
    best_metric_value: Option<f64>,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IterationEntry {
    iteration_id: String,
    stage: String,
    status: String,
    context: Option<String>,
    motivation: Option<String>,
    references: Option<String>,
    starting_checkpoint: Option<String>,
    training_data: Option<String>,
    method: Option<String>,
    training_config: Option<String>,
    evaluation: Option<String>,
    result: Option<String>,
    metric_name: Option<String>,
    metric_value: Option<f64>,
    analysis: Option<String>,
    artifacts: Option<String>,
    next_action: Option<String>,
    recorded_at: String,
}

pub struct AutoTrainessTools {
    pub sandbox_dir: PathBuf,
    pub config: Arc<AppConfig>,
}

impl AutoTrainessTools {
    fn project_root(&self, project_id: &str) -> Result<PathBuf, String> {
        let root_rel = self.config.autotrainess_default_project_root();
        let rel = format!("{}/{}", root_rel.trim_end_matches('/'), project_id.trim());
        resolve_path(&self.sandbox_dir, &rel)
            .ok_or_else(|| format!("Project path escapes sandbox or is invalid: {rel}"))
    }

    fn meta_path(&self, project_id: &str) -> Result<PathBuf, String> {
        Ok(self.project_root(project_id)?.join("database/meta.json"))
    }

    fn jsonl_path(&self, project_id: &str) -> Result<PathBuf, String> {
        Ok(self
            .project_root(project_id)?
            .join("database/iterations.jsonl"))
    }

    fn validate_project_id(project_id: &str) -> Result<(), String> {
        if project_id.is_empty()
            || project_id.contains('/')
            || project_id.contains('\\')
            || project_id.contains("..")
        {
            return Err("project_id must be a simple slug without path separators".into());
        }
        Ok(())
    }

    fn read_meta(&self, project_id: &str) -> Result<ProjectMeta, String> {
        let path = self.meta_path(project_id)?;
        if !path.exists() {
            return Err(format!(
                "Project meta not found at {}. Run train_db_init first.",
                path.display()
            ));
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse meta.json: {e}"))
    }

    fn write_meta_atomic(&self, path: &Path, meta: &ProjectMeta) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create dir {}: {e}", parent.display()))?;
        }
        let text =
            serde_json::to_string_pretty(meta).map_err(|e| format!("serialize meta: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("rename meta: {e}"))
    }

    fn read_iterations(&self, project_id: &str) -> Result<Vec<IterationEntry>, String> {
        let path = self.jsonl_path(project_id)?;
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file =
            std::fs::File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut entries = Vec::new();
        for line in std::io::BufReader::new(file).lines() {
            let line = line.map_err(|e| format!("read jsonl: {e}"))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: IterationEntry =
                serde_json::from_str(trimmed).map_err(|e| format!("parse iteration line: {e}"))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    fn count_iterations(path: &Path) -> Result<usize, String> {
        if !path.exists() {
            return Ok(0);
        }
        let file =
            std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut n = 0usize;
        for line in std::io::BufReader::new(file).lines() {
            let line = line.map_err(|e| format!("read jsonl: {e}"))?;
            if !line.trim().is_empty() {
                n += 1;
            }
        }
        Ok(n)
    }

    fn render_markdown_entry(entry: &IterationEntry) -> String {
        format!(
            "### Iteration: {id}\n\n\
- Context: {context}\n\
- Status: {status}\n\
- Motivation: {motivation}\n\
- References: {references}\n\
- Starting checkpoint: {checkpoint}\n\
- Training data: {data}\n\
- Method: {method}\n\
- Training config: {config}\n\
- Evaluation: {evaluation}\n\
- Result: {result}\n\
- Analysis: {analysis}\n\
- Artifacts: {artifacts}\n\
- Next action: {next}\n\n",
            id = entry.iteration_id,
            context = entry.context.as_deref().unwrap_or("N/A"),
            status = entry.status,
            motivation = entry.motivation.as_deref().unwrap_or("N/A"),
            references = entry.references.as_deref().unwrap_or("None"),
            checkpoint = entry.starting_checkpoint.as_deref().unwrap_or("N/A"),
            data = entry.training_data.as_deref().unwrap_or("N/A"),
            method = entry.method.as_deref().unwrap_or("N/A"),
            config = entry.training_config.as_deref().unwrap_or("N/A"),
            evaluation = entry.evaluation.as_deref().unwrap_or("N/A"),
            result = entry.result.as_deref().unwrap_or("N/A"),
            analysis = entry.analysis.as_deref().unwrap_or("N/A"),
            artifacts = entry.artifacts.as_deref().unwrap_or("N/A"),
            next = entry.next_action.as_deref().unwrap_or("N/A"),
        )
    }

    fn ensure_stage_heading(log_path: &Path, stage: &str) -> Result<(), String> {
        let heading = format!("## Stage: {stage}");
        if log_path.exists() {
            let existing = std::fs::read_to_string(log_path)
                .map_err(|e| format!("read {}: {e}", log_path.display()))?;
            if existing.contains(&heading) {
                return Ok(());
            }
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(log_path)
                .map_err(|e| format!("append {}: {e}", log_path.display()))?;
            writeln!(f, "\n{heading}\n").map_err(|e| format!("write heading: {e}"))?;
        } else {
            let body = format!("# Experiment Log\n\n{heading}\n\n");
            std::fs::write(log_path, body)
                .map_err(|e| format!("create {}: {e}", log_path.display()))?;
        }
        Ok(())
    }

    fn opt_string(args: &Value, key: &str) -> Option<String> {
        args.get(key)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }

    fn sandbox_rel(&self, path: &Path) -> String {
        path.strip_prefix(&self.sandbox_dir)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.display().to_string())
    }
}

macro_rules! train_tool {
    ($name:ident) => {
        pub struct $name {
            pub ctx: Arc<AutoTrainessTools>,
        }
    };
}

train_tool!(TrainDbInitTool);
train_tool!(TrainDbAppendTool);
train_tool!(TrainDbListTool);
train_tool!(TrainDbStatusTool);
train_tool!(TrainDbGetTool);

#[async_trait]
impl Tool for TrainDbInitTool {
    fn name(&self) -> &str {
        "train_db_init"
    }

    fn description(&self) -> &str {
        "Initialize an AutoTrainess training project under train/projects/{id}/ with experiment ledger, iterations/, and final_model/ layout."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string", "description": "Unique project slug (e.g. gsm8k_qwen3_4b)" },
                "stage": {
                    "type": "string",
                    "description": "Initial stage label (default stage0_task_definition)",
                    "default": "stage0_task_definition"
                },
                "base_model": { "type": "string", "description": "Base model path or Hugging Face id" },
                "benchmark": { "type": "string", "description": "Target benchmark or task name" },
                "target_metric": { "type": "string", "description": "Primary metric name to optimize" }
            },
            "required": ["project_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing project_id")?
            .trim();
        AutoTrainessTools::validate_project_id(project_id)?;
        let stage = args
            .get("stage")
            .and_then(|v| v.as_str())
            .unwrap_or("stage0_task_definition");
        let now = chrono::Utc::now().to_rfc3339();
        let root = self.ctx.project_root(project_id)?;
        for sub in [
            "database",
            "iterations",
            "final_model",
            "artifacts",
            "eval_results",
        ] {
            std::fs::create_dir_all(root.join(sub))
                .map_err(|e| format!("create {}: {e}", root.join(sub).display()))?;
        }
        let meta = ProjectMeta {
            schema_version: SCHEMA_VERSION,
            project_id: project_id.to_string(),
            stage: stage.to_string(),
            base_model: AutoTrainessTools::opt_string(&args, "base_model"),
            benchmark: AutoTrainessTools::opt_string(&args, "benchmark"),
            target_metric: AutoTrainessTools::opt_string(&args, "target_metric"),
            best_iteration_id: None,
            best_metric_value: None,
            created_at: now.clone(),
            updated_at: now,
        };
        let meta_path = root.join("database/meta.json");
        self.ctx.write_meta_atomic(&meta_path, &meta)?;
        let jsonl = root.join("database/iterations.jsonl");
        if !jsonl.exists() {
            std::fs::write(&jsonl, "").map_err(|e| format!("create iterations.jsonl: {e}"))?;
        }
        let log_path = root.join("experiment_log.md");
        if !log_path.exists() {
            std::fs::write(
                &log_path,
                "# Experiment Log\n\n## Stage: stage0_task_definition\n\n",
            )
            .map_err(|e| format!("create experiment_log.md: {e}"))?;
        }
        Ok(json!({
            "status": "initialized",
            "project_id": project_id,
            "project_root": self.ctx.sandbox_rel(&root),
            "stage": stage,
            "database": "database/meta.json",
            "ledger": "database/iterations.jsonl",
            "experiment_log": "experiment_log.md"
        })
        .to_string())
    }
}

#[async_trait]
impl Tool for TrainDbAppendTool {
    fn name(&self) -> &str {
        "train_db_append"
    }

    fn description(&self) -> &str {
        "Append one completed AutoTrainess iteration to the JSONL ledger and experiment_log.md."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string" },
                "iteration_id": { "type": "string", "description": "Unique iteration id (e.g. iter_001)" },
                "stage": { "type": "string", "description": "stage1_baseline | stage2_local | stage3_explore | ..." },
                "status": {
                    "type": "string",
                    "description": "completed | failed | blocked",
                    "default": "completed"
                },
                "context": { "type": "string" },
                "motivation": { "type": "string" },
                "references": { "type": "string" },
                "starting_checkpoint": { "type": "string" },
                "training_data": { "type": "string" },
                "method": { "type": "string" },
                "training_config": { "type": "string" },
                "evaluation": { "type": "string" },
                "result": { "type": "string" },
                "metric_name": { "type": "string" },
                "metric_value": { "type": "number" },
                "analysis": { "type": "string" },
                "artifacts": { "type": "string" },
                "next_action": { "type": "string" }
            },
            "required": ["project_id", "iteration_id", "stage", "status"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing project_id")?
            .trim();
        AutoTrainessTools::validate_project_id(project_id)?;
        let iteration_id = args
            .get("iteration_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing iteration_id")?
            .trim();
        if iteration_id.is_empty() {
            return Err("iteration_id must be non-empty".into());
        }
        let stage = args
            .get("stage")
            .and_then(|v| v.as_str())
            .ok_or("Missing stage")?
            .trim();
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("completed")
            .trim();
        let jsonl_path = self.ctx.jsonl_path(project_id)?;
        let count = AutoTrainessTools::count_iterations(&jsonl_path)?;
        if count >= self.ctx.config.autotrainess_max_log_entries() {
            return Err(format!(
                "Experiment ledger full (max {} entries). Start a new project or prune.",
                self.ctx.config.autotrainess_max_log_entries()
            ));
        }
        let existing = self.ctx.read_iterations(project_id)?;
        if existing.iter().any(|e| e.iteration_id == iteration_id) {
            return Err(format!(
                "iteration_id '{iteration_id}' already exists in the ledger"
            ));
        }
        let entry = IterationEntry {
            iteration_id: iteration_id.to_string(),
            stage: stage.to_string(),
            status: status.to_string(),
            context: AutoTrainessTools::opt_string(&args, "context"),
            motivation: AutoTrainessTools::opt_string(&args, "motivation"),
            references: AutoTrainessTools::opt_string(&args, "references"),
            starting_checkpoint: AutoTrainessTools::opt_string(&args, "starting_checkpoint"),
            training_data: AutoTrainessTools::opt_string(&args, "training_data"),
            method: AutoTrainessTools::opt_string(&args, "method"),
            training_config: AutoTrainessTools::opt_string(&args, "training_config"),
            evaluation: AutoTrainessTools::opt_string(&args, "evaluation"),
            result: AutoTrainessTools::opt_string(&args, "result"),
            metric_name: AutoTrainessTools::opt_string(&args, "metric_name"),
            metric_value: args.get("metric_value").and_then(|v| v.as_f64()),
            analysis: AutoTrainessTools::opt_string(&args, "analysis"),
            artifacts: AutoTrainessTools::opt_string(&args, "artifacts"),
            next_action: AutoTrainessTools::opt_string(&args, "next_action"),
            recorded_at: chrono::Utc::now().to_rfc3339(),
        };
        let line = serde_json::to_string(&entry).map_err(|e| e.to_string())?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&jsonl_path)
            .map_err(|e| format!("append iterations.jsonl: {e}"))?;
        writeln!(f, "{line}").map_err(|e| format!("write iterations.jsonl: {e}"))?;

        let root = self.ctx.project_root(project_id)?;
        let log_path = root.join("experiment_log.md");
        AutoTrainessTools::ensure_stage_heading(&log_path, stage)?;
        let mut log_f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| format!("append experiment_log.md: {e}"))?;
        write!(
            log_f,
            "{}",
            AutoTrainessTools::render_markdown_entry(&entry)
        )
        .map_err(|e| format!("write experiment_log.md: {e}"))?;

        let mut meta = self.ctx.read_meta(project_id)?;
        meta.stage = stage.to_string();
        meta.updated_at = chrono::Utc::now().to_rfc3339();
        if let Some(v) = entry.metric_value {
            let better = match meta.best_metric_value {
                None => true,
                Some(prev) => v > prev,
            };
            if better {
                meta.best_metric_value = Some(v);
                meta.best_iteration_id = Some(iteration_id.to_string());
            }
        }
        let meta_path = self.ctx.meta_path(project_id)?;
        self.ctx.write_meta_atomic(&meta_path, &meta)?;

        Ok(json!({
            "status": "appended",
            "project_id": project_id,
            "iteration_id": iteration_id,
            "stage": stage,
            "entry_count": count + 1,
            "best_iteration_id": meta.best_iteration_id,
            "best_metric_value": meta.best_metric_value,
        })
        .to_string())
    }
}

#[async_trait]
impl Tool for TrainDbListTool {
    fn name(&self) -> &str {
        "train_db_list"
    }

    fn description(&self) -> &str {
        "List recent AutoTrainess experiment iterations from the project ledger."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string" },
                "limit": { "type": "integer", "description": "Max iterations to return (default 20, most recent last)" }
            },
            "required": ["project_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing project_id")?;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .clamp(1, 200) as usize;
        let mut entries = self.ctx.read_iterations(project_id)?;
        let total = entries.len();
        if entries.len() > limit {
            entries = entries.split_off(entries.len() - limit);
        }
        Ok(json!({
            "project_id": project_id,
            "total": total,
            "returned": entries.len(),
            "iterations": entries,
        })
        .to_string())
    }
}

#[async_trait]
impl Tool for TrainDbStatusTool {
    fn name(&self) -> &str {
        "train_db_status"
    }

    fn description(&self) -> &str {
        "Summarize an AutoTrainess project: stage, iteration count, best metric, paths."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string" }
            },
            "required": ["project_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing project_id")?;
        let meta = self.ctx.read_meta(project_id)?;
        let entries = self.ctx.read_iterations(project_id)?;
        let root = self.ctx.project_root(project_id)?;
        Ok(json!({
            "project_id": meta.project_id,
            "schema_version": meta.schema_version,
            "stage": meta.stage,
            "base_model": meta.base_model,
            "benchmark": meta.benchmark,
            "target_metric": meta.target_metric,
            "iteration_count": entries.len(),
            "best_iteration_id": meta.best_iteration_id,
            "best_metric_value": meta.best_metric_value,
            "project_root": self.ctx.sandbox_rel(&root),
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
        })
        .to_string())
    }
}

#[async_trait]
impl Tool for TrainDbGetTool {
    fn name(&self) -> &str {
        "train_db_get"
    }

    fn description(&self) -> &str {
        "Fetch one AutoTrainess iteration entry by iteration_id."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string" },
                "iteration_id": { "type": "string" }
            },
            "required": ["project_id", "iteration_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing project_id")?;
        let iteration_id = args
            .get("iteration_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing iteration_id")?;
        let entries = self.ctx.read_iterations(project_id)?;
        let entry = entries
            .into_iter()
            .find(|e| e.iteration_id == iteration_id)
            .ok_or_else(|| format!("iteration_id '{iteration_id}' not found"))?;
        Ok(serde_json::to_string_pretty(&entry).map_err(|e| e.to_string())?)
    }
}

pub fn register_autotrainess_tools(
    tools: &mut crate::tools::ToolRegistry,
    sandbox_dir: PathBuf,
    config: Arc<AppConfig>,
) {
    let ctx = Arc::new(AutoTrainessTools {
        sandbox_dir,
        config,
    });
    tools.register(Box::new(TrainDbInitTool { ctx: ctx.clone() }));
    tools.register(Box::new(TrainDbAppendTool { ctx: ctx.clone() }));
    tools.register(Box::new(TrainDbListTool { ctx: ctx.clone() }));
    tools.register(Box::new(TrainDbStatusTool { ctx: ctx.clone() }));
    tools.register(Box::new(TrainDbGetTool { ctx }));
}
