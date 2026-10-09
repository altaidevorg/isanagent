use serde::Serialize;

use crate::skills::SharedSkillRegistry;

/// Skill metadata for the settings screen. Instruction bodies stay in the registry.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct InstalledSkill {
    pub name: String,
    pub description: String,
    pub available: bool,
    pub always: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SkillList {
    pub skills: Vec<InstalledSkill>,
}

pub async fn list_installed_skills(registry: &SharedSkillRegistry) -> SkillList {
    let registry = registry.read().await;
    SkillList {
        skills: registry
            .list_skills()
            .into_iter()
            .map(|skill| InstalledSkill {
                name: skill.name,
                description: skill.description,
                available: skill.available,
                always: skill.always,
            })
            .collect(),
    }
}

/// Installs into the live registry the agent already reads. Same function as `InstallSkill`.
pub async fn install_from_repo(
    registry: &SharedSkillRegistry,
    repo: &str,
    skill_name: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut registry = registry.write().await;
    registry.install_skills_from_repo(repo, skill_name).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::SkillRegistry;
    use std::io::Write;

    fn write_skill(dir: &std::path::Path, name: &str, body: &str) {
        let skill_dir = dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let mut file = std::fs::File::create(skill_dir.join("SKILL.md")).unwrap();
        writeln!(
            file,
            "---\nname: {name}\ndescription: {name} description\n---\n\n{body}"
        )
        .unwrap();
    }

    #[tokio::test]
    async fn list_sorts_names_and_omits_instruction_bodies() {
        let dir = std::env::temp_dir().join(format!(
            "skill_settings_list_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0)
        ));
        write_skill(&dir, "zeta", "ZETA_BODY_SHOULD_NOT_LEAK");
        write_skill(&dir, "alpha", "ALPHA_BODY_SHOULD_NOT_LEAK");
        let registry =
            std::sync::Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(dir.clone())));

        let listed = list_installed_skills(&registry).await;
        let json = serde_json::to_string(&listed).unwrap();

        assert_eq!(
            listed
                .skills
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
        assert!(!json.contains("instructions"));
        assert!(!json.contains("SHOULD_NOT_LEAK"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
