use log::{info, warn};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Thread-safe skill registry shared across agents and the TUI.
pub type SharedSkillRegistry = Arc<RwLock<SkillRegistry>>;

/// Represents a parsed Anthropic Agent Skill
#[derive(Debug, Clone)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub path: PathBuf,
    pub always: bool,
    pub available: bool,
}

#[derive(Debug, Deserialize)]
struct SkillRequiresFrontmatter {
    bins: Option<Vec<String>>,
    env: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: String,
    description: String,
    always: Option<bool>,
    requires: Option<SkillRequiresFrontmatter>,
}

pub struct SkillRegistry {
    pub skills_dir: PathBuf,
    skills: HashMap<String, SkillDefinition>,
}

impl SkillRegistry {
    pub fn new(skills_dir: PathBuf) -> Self {
        let mut registry = Self {
            skills_dir,
            skills: HashMap::new(),
        };
        registry.scan_for_skills();
        registry
    }

    /// Scans the skills directory for folders containing a SKILL.md file.
    pub fn scan_for_skills(&mut self) {
        if !self.skills_dir.exists() {
            return;
        }

        let entries = match fs::read_dir(&self.skills_dir) {
            Ok(pwd) => pwd,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skill_md = path.join("SKILL.md");
                if skill_md.exists() {
                    if let Some(def) = Self::parse_skill_md(&skill_md) {
                        // Check availability
                        if !def.available {
                            warn!("Skill {} loaded but marked UNAVAILABLE due to missing requirements.", def.name);
                        } else {
                            info!("Loaded Skill: {}", def.name);
                        }
                        self.skills.insert(def.name.clone(), def);
                    }
                }
            }
        }
    }

    /// Parses a SKILL.md file extracting YAML frontmatter and the Markdown body.
    fn parse_skill_md(path: &PathBuf) -> Option<SkillDefinition> {
        let content = fs::read_to_string(path).ok()?;

        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return None;
        }

        // Extremely basic frontmatter parsing: looks for --- ... ---
        if lines[0] != "---" {
            warn!(
                "Skipping {:?}: No YAML frontmatter found (must start with '---')",
                path
            );
            return None;
        }

        let mut end_idx = 0;
        for (i, &line) in lines.iter().enumerate().skip(1) {
            if line == "---" {
                end_idx = i;
                break;
            }
        }

        if end_idx == 0 {
            warn!("Skipping {:?}: Unclosed YAML frontmatter", path);
            return None;
        }

        let frontmatter_str = lines[1..end_idx].join("\n");
        let body_str = lines[end_idx + 1..].join("\n");

        match serde_yaml::from_str::<SkillFrontmatter>(&frontmatter_str) {
            Ok(metadata) => {
                let mut available = true;
                let mut missing_reasons = Vec::new();

                if let Some(reqs) = &metadata.requires {
                    if let Some(bins) = &reqs.bins {
                        for bin in bins {
                            if which::which(bin).is_err() {
                                available = false;
                                missing_reasons.push(format!("missing bin: {}", bin));
                            }
                        }
                    }
                    if let Some(envs) = &reqs.env {
                        for env in envs {
                            if std::env::var(env).is_err() {
                                available = false;
                                missing_reasons.push(format!("missing env var: {}", env));
                            }
                        }
                    }
                }

                let final_desc = if available {
                    metadata.description
                } else {
                    format!(
                        "{} [❌ UNAVAILABLE - {}]",
                        metadata.description,
                        missing_reasons.join(", ")
                    )
                };

                Some(SkillDefinition {
                    name: metadata.name,
                    description: final_desc,
                    instructions: body_str,
                    path: path.clone(),
                    always: metadata.always.unwrap_or(false),
                    available,
                })
            }
            Err(e) => {
                warn!("Failed to parse YAML frontmatter for {:?}: {}", path, e);
                None
            }
        }
    }

    /// Returns the metadata for progressive disclosure to the prompt
    pub fn get_capabilities_summary(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }

        let mut summary = String::from("\n\nAvailable Agent Skills:\n");
        let mut always_blocks = String::new();

        for skill in self.skills.values() {
            if skill.always && skill.available {
                always_blocks.push_str(&format!(
                    "\n--- SKILL AUTOMATICALLY LOADED: {} ---\n{}\n",
                    skill.name, skill.instructions
                ));
            } else {
                summary.push_str(&format!("- **{}**: {}\n", skill.name, skill.description));
            }
        }
        summary.push_str("\nTo execute a skill, use the 'load_skill_instructions' tool with the skill's name to learn how to use it contextually.\n");

        format!("{}{}", summary, always_blocks)
    }

    pub fn get_skill_instructions(&self, name: &str) -> Result<String, String> {
        match self.skills.get(name) {
            Some(skill) => {
                if !skill.available {
                    return Err(format!(
                        "Skill '{}' cannot be loaded because it is missing dependencies: {}",
                        name, skill.description
                    ));
                }
                Ok(skill.instructions.clone())
            }
            None => Err(format!("Skill '{}' not found", name)),
        }
    }

    pub fn get_skill_names(&self) -> Vec<String> {
        self.skills.keys().cloned().collect()
    }

    /// One line per skill for quick discovery (includes unavailable entries with their reason).
    pub fn format_skill_directory(&self) -> String {
        if self.skills.is_empty() {
            return "No skills discovered.".to_string();
        }
        let mut names: Vec<_> = self.skills.keys().cloned().collect();
        names.sort();
        let mut out = String::from("Available skills:\n\n");
        for n in names {
            if let Some(s) = self.skills.get(&n) {
                out.push_str(&format!("- **{}**: {}\n", s.name, s.description));
            }
        }
        out
    }

    /// Short metadata for a skill (no full instruction body).
    pub fn get_skill_metadata(&self, name: &str) -> Result<String, String> {
        let skill = self
            .skills
            .get(name)
            .ok_or_else(|| format!("Skill '{}' not found", name))?;
        Ok(format!(
            "Skill: {}\nAvailable: {}\nDescription: {}\nInstruction length: {} characters\nPath: {}",
            skill.name,
            skill.available,
            skill.description,
            skill.instructions.len(),
            skill.path.display()
        ))
    }

    /// Installs skills from a remote GitHub repository.
    /// If `specific_skill` is Some, only that skill will be installed.
    /// Returns a list of installed skill names.
    pub async fn install_skills_from_repo(
        &mut self,
        repo_url: &str,
        specific_skill: Option<&str>,
    ) -> Result<Vec<String>, String> {
        // 0. Pre-flight check: is git installed?
        if which::which("git").is_err() {
            return Err("The 'git' command was not found. Please install git to use remote skill installation.".to_string());
        }

        // Support shorthand owner/repo format
        let full_repo_url = if !repo_url.contains("://") && repo_url.contains('/') {
            format!("https://github.com/{}", repo_url)
        } else {
            repo_url.to_string()
        };

        info!(
            "Installing skills from repository: {} (specific: {:?})",
            full_repo_url, specific_skill
        );

        // 1. Create a temporary directory with a cleanup guard
        let temp_dir_path =
            std::env::temp_dir().join(format!("isanagent-skills-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir_path)
            .map_err(|e| format!("Failed to create temp dir: {}", e))?;
        let _guard = TempDirGuard::new(temp_dir_path.clone());

        // 2. Clone the repository (using git command)
        let status = std::process::Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg(&full_repo_url)
            .arg(&temp_dir_path)
            .status()
            .map_err(|e| {
                format!(
                    "Failed to execute git clone: {}. Make sure 'git' is installed and in your PATH.",
                    e
                )
            })?;

        if !status.success() {
            return Err(format!(
                "git clone failed with exit code: {:?}",
                status.code()
            ));
        }

        // 3. Scan the cloned repo for SKILL.md files
        let mut installed_skills = Vec::new();
        let walker = walkdir::WalkDir::new(&temp_dir_path).into_iter();

        for entry in walker.filter_map(|e| e.ok()) {
            if entry.file_name() == "SKILL.md" {
                let skill_md_path = entry.path();
                let skill_dir = skill_md_path.parent().ok_or("Invalid skill path")?;

                // Parse to get the skill name
                if let Some(def) = Self::parse_skill_md(&skill_md_path.to_path_buf()) {
                    // Filter if specific skill requested
                    if let Some(requested) = specific_skill {
                        if def.name != requested {
                            continue;
                        }
                    }

                    let dest_dir = self.skills_dir.join(&def.name);

                    // Atomic installation: copy to a .tmp sibling first
                    let tmp_dest_dir = self.skills_dir.join(format!("{}.tmp", def.name));
                    if tmp_dest_dir.exists() {
                        let _ = fs::remove_dir_all(&tmp_dest_dir);
                    }

                    fs::create_dir_all(&tmp_dest_dir)
                        .map_err(|e| format!("Failed to create skill dir: {}", e))?;

                    // Copy everything from skill_dir to tmp_dest_dir
                    copy_dir_recursive(skill_dir, &tmp_dest_dir)?;

                    // Swap: remove old and rename tmp to dest
                    if dest_dir.exists() {
                        fs::remove_dir_all(&dest_dir)
                            .map_err(|e| format!("Failed to remove existing skill dir: {}", e))?;
                    }
                    fs::rename(&tmp_dest_dir, &dest_dir)
                        .map_err(|e| format!("Failed to finalize skill installation: {}", e))?;

                    installed_skills.push(def.name);

                    // If we found the specific skill, we can stop
                    if specific_skill.is_some() {
                        break;
                    }
                }
            }
        }

        if let Some(skill) = specific_skill.filter(|_| installed_skills.is_empty()) {
            return Err(format!(
                "Skill '{}' not found in repository {}",
                skill, full_repo_url
            ));
        }

        // 5. Re-scan for skills
        self.scan_for_skills();

        Ok(installed_skills)
    }
}

/// Ensures a temporary directory is deleted when this guard is dropped.
struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    if !dst.exists() {
        fs::create_dir_all(dst)
            .map_err(|e| format!("Failed to create directory {:?}: {}", dst, e))?;
    }

    for entry in
        fs::read_dir(src).map_err(|e| format!("Failed to read directory {:?}: {}", src, e))?
    {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let file_type = entry
            .file_type()
            .map_err(|e| format!("Failed to get file type: {}", e))?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).map_err(|e| {
                format!(
                    "Failed to copy file {:?} to {:?}: {}",
                    src_path, dst_path, e
                )
            })?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod skill_metadata_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn metadata_and_directory_without_loading_full_body() {
        let dir = std::env::temp_dir().join(format!(
            "skill_md_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let skill_dir = dir.join("demo_skill");
        std::fs::create_dir_all(&skill_dir).unwrap();
        let md = skill_dir.join("SKILL.md");
        let mut f = std::fs::File::create(&md).unwrap();
        writeln!(
            f,
            "---\nname: demo_skill\ndescription: A demo\nrequires:\n  bins: [nonexistent_bin_xyz123]\n---\n\nBODY {}",
            "x".repeat(500)
        )
        .unwrap();

        let reg = SkillRegistry::new(dir.clone());
        let meta = reg.get_skill_metadata("demo_skill").unwrap();
        let n: usize = meta
            .lines()
            .find_map(|l| {
                l.strip_prefix("Instruction length:")
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|n| n.parse().ok())
            })
            .expect("instruction length line");
        assert!(n >= 500, "expected long body, got length {}", n);
        assert!(meta.contains("Available: false"));

        let dir_txt = reg.format_skill_directory();
        assert!(dir_txt.contains("demo_skill"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
