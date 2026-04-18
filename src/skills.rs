use log::{info, warn};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

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
                    .and_then(|s| s.trim().split_whitespace().next())
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
