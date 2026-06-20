//! MaxEvolve kernel porting tools — MAP-Elites project database and project bootstrap.

use std::collections::HashMap;
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
struct MapElitesArchive {
    schema_version: u32,
    project_id: String,
    target_hardware: String,
    dimensions: Vec<String>,
    bins_per_dimension: Vec<usize>,
    cells: HashMap<String, MapElitesCell>,
    global_best_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MapElitesCell {
    id: String,
    kernel_path: String,
    fitness_latency_ms: Option<f64>,
    fitness_mfu: Option<f64>,
    fitness_tflops: Option<f64>,
    complexity_loc: Option<usize>,
    complexity_ast_depth: Option<usize>,
    tile_volume: Option<usize>,
    mutation_class: Option<String>,
    parent_id: Option<String>,
    generation: Option<u32>,
    notes: Option<String>,
    inserted_at: String,
}

pub struct KernelPortingTools {
    pub sandbox_dir: PathBuf,
    pub config: Arc<AppConfig>,
}

impl KernelPortingTools {
    fn project_root(&self, project_id: &str) -> Result<PathBuf, String> {
        let root_rel = self.config.kernel_porting_default_project_root();
        let rel = format!("{}/{}", root_rel.trim_end_matches('/'), project_id.trim());
        resolve_path(&self.sandbox_dir, &rel)
            .ok_or_else(|| format!("Project path escapes sandbox or is invalid: {rel}"))
    }

    fn db_path(&self, project_id: &str) -> Result<PathBuf, String> {
        Ok(self
            .project_root(project_id)?
            .join("database/map_elites.json"))
    }

    fn empty_archive(project_id: &str, target_hardware: &str) -> MapElitesArchive {
        MapElitesArchive {
            schema_version: SCHEMA_VERSION,
            project_id: project_id.to_string(),
            target_hardware: target_hardware.to_string(),
            dimensions: vec![
                "fitness_latency_ms".into(),
                "complexity_loc".into(),
                "tile_volume".into(),
            ],
            bins_per_dimension: vec![10, 5, 5],
            cells: HashMap::new(),
            global_best_id: None,
        }
    }

    fn read_archive(&self, project_id: &str) -> Result<MapElitesArchive, String> {
        let path = self.db_path(project_id)?;
        if !path.exists() {
            return Err(format!(
                "MAP-Elites archive not found at {}. Run kernel_db_init first.",
                path.display()
            ));
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse map_elites.json: {e}"))
    }

    fn write_archive_atomic(&self, path: &Path, archive: &MapElitesArchive) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create dir {}: {e}", parent.display()))?;
        }
        let text =
            serde_json::to_string_pretty(archive).map_err(|e| format!("serialize archive: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("rename archive: {e}"))
    }

    fn update_global_best(archive: &mut MapElitesArchive) {
        archive.global_best_id = archive
            .cells
            .values()
            .filter(|c| c.fitness_latency_ms.is_some())
            .min_by(|a, b| {
                a.fitness_latency_ms
                    .unwrap_or(f64::MAX)
                    .partial_cmp(&b.fitness_latency_ms.unwrap_or(f64::MAX))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.id.clone());
    }

    fn elite_rank_key(cell: &MapElitesCell) -> (u64, usize, usize) {
        (
            cell.fitness_latency_ms
                .map(|v| v.to_bits())
                .unwrap_or(u64::MAX),
            cell.complexity_loc.unwrap_or(0),
            cell.tile_volume.unwrap_or(0),
        )
    }
}

macro_rules! kernel_tool {
    ($name:ident, $ctx:ident) => {
        pub struct $name {
            pub ctx: Arc<KernelPortingTools>,
        }
    };
}

kernel_tool!(KernelDbInitTool, ctx);
kernel_tool!(KernelDbSampleTool, ctx);
kernel_tool!(KernelDbInsertTool, ctx);
kernel_tool!(KernelDbStatusTool, ctx);

#[async_trait]
impl Tool for KernelDbInitTool {
    fn name(&self) -> &str {
        "kernel_db_init"
    }

    fn description(&self) -> &str {
        "Initialize a MaxEvolve kernel project directory with MAP-Elites database, artifacts/, candidates/, and source/ layout."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string", "description": "Unique project slug (e.g. vector_add_v1)" },
                "target_hardware": {
                    "type": "string",
                    "description": "cpu_interpret | gpu_hopper | tpu_v5e | tpu_v6e",
                    "default": "cpu_interpret"
                },
                "source_relative_path": {
                    "type": "string",
                    "description": "Optional sandbox-relative path to copy into project source/"
                }
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
        if project_id.is_empty()
            || project_id.contains('/')
            || project_id.contains('\\')
            || project_id.contains("..")
        {
            return Err("project_id must be a simple slug without path separators".into());
        }
        let target_hardware = args
            .get("target_hardware")
            .and_then(|v| v.as_str())
            .unwrap_or("cpu_interpret");
        let root = self.ctx.project_root(project_id)?;
        for sub in ["source", "database", "artifacts", "candidates"] {
            std::fs::create_dir_all(root.join(sub))
                .map_err(|e| format!("create {}: {e}", root.join(sub).display()))?;
        }
        let archive = KernelPortingTools::empty_archive(project_id, target_hardware);
        let db_path = root.join("database/map_elites.json");
        self.ctx
            .write_archive_atomic(&db_path, &archive)
            .map_err(|e| e.to_string())?;
        let lineage = root.join("database/lineage.jsonl");
        if !lineage.exists() {
            std::fs::write(&lineage, "").map_err(|e| format!("create lineage: {e}"))?;
        }
        if let Some(src) = args.get("source_relative_path").and_then(|v| v.as_str()) {
            if let Some(resolved) = resolve_path(&self.ctx.sandbox_dir, src.trim()) {
                if resolved.is_file() {
                    let dest = root.join("source").join(
                        resolved
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("input.py"),
                    );
                    std::fs::copy(&resolved, &dest).map_err(|e| format!("copy source: {e}"))?;
                }
            }
        }
        Ok(json!({
            "status": "initialized",
            "project_id": project_id,
            "project_root": root.strip_prefix(&self.ctx.sandbox_dir)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| root.display().to_string()),
            "target_hardware": target_hardware,
            "database": "database/map_elites.json"
        })
        .to_string())
    }
}

#[async_trait]
impl Tool for KernelDbSampleTool {
    fn name(&self) -> &str {
        "kernel_db_sample"
    }

    fn description(&self) -> &str {
        "Sample top-performing MAP-Elites elites and empty diversity cells for mutation prompts."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string" },
                "top_k": { "type": "integer", "description": "Top elites by latency (default 3)" },
                "include_global_best": { "type": "boolean", "default": true }
            },
            "required": ["project_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing project_id")?;
        let top_k = args
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(3)
            .clamp(1, 20) as usize;
        let archive = self.ctx.read_archive(project_id)?;
        let mut elites: Vec<&MapElitesCell> = archive.cells.values().collect();
        elites.sort_by_key(|c| KernelPortingTools::elite_rank_key(c));
        elites.truncate(top_k);
        let global_best = archive
            .global_best_id
            .as_ref()
            .and_then(|id| archive.cells.get(id));
        Ok(json!({
            "project_id": project_id,
            "target_hardware": archive.target_hardware,
            "cell_count": archive.cells.len(),
            "top_elites": elites,
            "global_best": global_best,
            "dimensions": archive.dimensions,
            "bins_per_dimension": archive.bins_per_dimension,
        })
        .to_string())
    }
}

#[async_trait]
impl Tool for KernelDbInsertTool {
    fn name(&self) -> &str {
        "kernel_db_insert"
    }

    fn description(&self) -> &str {
        "Insert or replace a MAP-Elites elite after correctness/profiling. Validates required fitness fields."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_id": { "type": "string" },
                "id": { "type": "string", "description": "Unique candidate id (uuid slug)" },
                "kernel_path": { "type": "string", "description": "Sandbox-relative path to converted_jax.py" },
                "fitness_latency_ms": { "type": "number" },
                "fitness_mfu": { "type": "number" },
                "fitness_tflops": { "type": "number" },
                "complexity_loc": { "type": "integer" },
                "complexity_ast_depth": { "type": "integer" },
                "tile_volume": { "type": "integer" },
                "mutation_class": { "type": "string" },
                "parent_id": { "type": "string" },
                "generation": { "type": "integer" },
                "notes": { "type": "string" }
            },
            "required": ["project_id", "id", "kernel_path", "fitness_latency_ms"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing project_id")?;
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing id")?
            .to_string();
        let kernel_path = args
            .get("kernel_path")
            .and_then(|v| v.as_str())
            .ok_or("Missing kernel_path")?
            .to_string();
        let fitness_latency_ms = args
            .get("fitness_latency_ms")
            .and_then(|v| v.as_f64())
            .ok_or("Missing fitness_latency_ms")?;
        let mut archive = self.ctx.read_archive(project_id)?;
        if archive.cells.len() >= self.ctx.config.kernel_porting_max_archive_entries() {
            return Err(format!(
                "Archive full (max {} entries). Prune or start a new project.",
                self.ctx.config.kernel_porting_max_archive_entries()
            ));
        }
        let cell = MapElitesCell {
            id: id.clone(),
            kernel_path,
            fitness_latency_ms: Some(fitness_latency_ms),
            fitness_mfu: args.get("fitness_mfu").and_then(|v| v.as_f64()),
            fitness_tflops: args.get("fitness_tflops").and_then(|v| v.as_f64()),
            complexity_loc: args
                .get("complexity_loc")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            complexity_ast_depth: args
                .get("complexity_ast_depth")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            tile_volume: args
                .get("tile_volume")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            mutation_class: args
                .get("mutation_class")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            parent_id: args
                .get("parent_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            generation: args
                .get("generation")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32),
            notes: args
                .get("notes")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            inserted_at: chrono::Utc::now().to_rfc3339(),
        };
        let map_elites_key = format!(
            "{}:{}:{}",
            (fitness_latency_ms * 100.0) as u64,
            cell.complexity_loc.unwrap_or(0),
            cell.tile_volume.unwrap_or(0)
        );
        let replaced = archive.cells.insert(map_elites_key, cell.clone());
        KernelPortingTools::update_global_best(&mut archive);
        let db_path = self.ctx.db_path(project_id)?;
        self.ctx.write_archive_atomic(&db_path, &archive)?;
        let lineage_path = self
            .ctx
            .project_root(project_id)?
            .join("database/lineage.jsonl");
        let lineage_line = serde_json::to_string(&cell).map_err(|e| e.to_string())?;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&lineage_path)
            .map_err(|e| format!("append lineage: {e}"))?;
        writeln!(f, "{lineage_line}").map_err(|e| format!("write lineage: {e}"))?;
        Ok(json!({
            "status": "inserted",
            "id": id,
            "replaced_previous": replaced.is_some(),
            "global_best_id": archive.global_best_id,
            "cell_count": archive.cells.len(),
        })
        .to_string())
    }
}

#[async_trait]
impl Tool for KernelDbStatusTool {
    fn name(&self) -> &str {
        "kernel_db_status"
    }

    fn description(&self) -> &str {
        "Summarize MAP-Elites archive stats: cell count, global best, hardware target."
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
        let archive = self.ctx.read_archive(project_id)?;
        let best = archive
            .global_best_id
            .as_ref()
            .and_then(|id| archive.cells.get(id));
        Ok(json!({
            "project_id": archive.project_id,
            "target_hardware": archive.target_hardware,
            "schema_version": archive.schema_version,
            "cell_count": archive.cells.len(),
            "global_best": best,
            "dimensions": archive.dimensions,
        })
        .to_string())
    }
}

pub fn register_kernel_porting_tools(
    tools: &mut crate::tools::ToolRegistry,
    sandbox_dir: PathBuf,
    config: Arc<AppConfig>,
) {
    let ctx = Arc::new(KernelPortingTools {
        sandbox_dir,
        config,
    });
    tools.register(Box::new(KernelDbInitTool { ctx: ctx.clone() }));
    tools.register(Box::new(KernelDbSampleTool { ctx: ctx.clone() }));
    tools.register(Box::new(KernelDbInsertTool { ctx: ctx.clone() }));
    tools.register(Box::new(KernelDbStatusTool { ctx }));
}
