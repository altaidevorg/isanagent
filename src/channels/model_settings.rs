//! Web settings surface for choosing a configured model.
//!
//! The catalog is the same expanded provider map the terminal `/model` picker uses.
//! API keys are never serialized. A submitted key is stored in the OS keychain and
//! applied to later reasoning runs through `BusMessage::SwitchModel`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::Serialize;

use crate::config::ProviderConfig;

const LAST_MODEL_FILE: &str = ".system_generated/last_model";

/// Providers offered by the settings screen, plus the key of the model currently
/// selected for runs admitted after the next switch.
#[derive(Clone)]
pub struct ModelCatalog {
    pub workspace_dir: PathBuf,
    pub providers: HashMap<String, ProviderConfig>,
    pub active_key: Arc<RwLock<Option<String>>>,
    /// Providers given a key during this process. Keychain reads can fail in the
    /// same session that just stored the secret, so the settings list remembers
    /// the provider name without keeping the secret.
    keyed_providers: Arc<RwLock<HashSet<String>>>,
}

impl ModelCatalog {
    pub fn new(
        workspace_dir: PathBuf,
        providers: HashMap<String, ProviderConfig>,
        active_key: Option<String>,
    ) -> Self {
        Self {
            workspace_dir,
            providers,
            active_key: Arc::new(RwLock::new(active_key)),
            keyed_providers: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub fn note_provider_key(&self, provider_name: &str) {
        self.keyed_providers
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(provider_name.to_string());
    }

    fn provider_has_noted_key(&self, provider_name: &str) -> bool {
        self.keyed_providers
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(provider_name)
    }

    pub fn active_key(&self) -> Option<String> {
        self.active_key
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn set_active_key(&self, key: String) {
        *self
            .active_key
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(key);
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelEntry {
    pub key: String,
    pub provider_name: String,
    pub model_name: String,
    pub has_api_key: bool,
    pub api_key_env: String,
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelList {
    pub active_key: Option<String>,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedModelSwitch {
    pub key: String,
    pub provider_name: String,
    pub model_name: String,
    pub base_url: String,
    pub api_key: String,
    pub store_in_keychain: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSwitchError {
    UnknownModel,
    MissingApiKey { api_key_env: String },
    PlaceholderApiKey,
    UnresolvedBaseUrl(String),
}

pub fn list_models(catalog: &ModelCatalog) -> ModelList {
    let mut models: Vec<ModelEntry> = catalog
        .providers
        .iter()
        .map(|(key, cfg)| ModelEntry {
            key: key.clone(),
            provider_name: cfg.provider_name.clone(),
            model_name: cfg.model_name.clone(),
            has_api_key: cfg.resolve_api_key().is_ok()
                || catalog.provider_has_noted_key(&cfg.provider_name),
            api_key_env: cfg.resolved_api_key_env(),
            base_url: cfg.resolved_base_url().unwrap_or_default(),
        })
        .collect();
    models.sort_by(|a, b| {
        a.provider_name
            .cmp(&b.provider_name)
            .then_with(|| a.model_name.cmp(&b.model_name))
            .then_with(|| a.key.cmp(&b.key))
    });
    let active_key = catalog
        .active_key()
        .filter(|key| models.iter().any(|model| &model.key == key));
    ModelList { active_key, models }
}

pub fn prepare_model_switch(
    catalog: &ModelCatalog,
    key: &str,
    provided_api_key: Option<&str>,
) -> Result<PreparedModelSwitch, ModelSwitchError> {
    let key = key.trim();
    let cfg = catalog
        .providers
        .get(key)
        .ok_or(ModelSwitchError::UnknownModel)?;
    let provided = provided_api_key
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let (api_key, store_in_keychain) = if let Some(provided) = provided {
        if api_key_looks_like_placeholder(provided) {
            return Err(ModelSwitchError::PlaceholderApiKey);
        }
        (provided.to_string(), true)
    } else {
        match cfg.resolve_api_key() {
            Ok(existing) => (existing, false),
            Err(_) => {
                return Err(ModelSwitchError::MissingApiKey {
                    api_key_env: cfg.resolved_api_key_env(),
                })
            }
        }
    };
    let base_url = cfg
        .resolved_base_url()
        .map_err(ModelSwitchError::UnresolvedBaseUrl)?;
    Ok(PreparedModelSwitch {
        key: key.to_string(),
        provider_name: cfg.provider_name.clone(),
        model_name: cfg.model_name.clone(),
        base_url,
        api_key,
        store_in_keychain,
    })
}

pub fn persist_active_model_key(workspace_dir: &Path, key: &str) -> Result<(), String> {
    let dir = workspace_dir.join(".system_generated");
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("Failed to create {}: {error}", dir.display()))?;
    let path = workspace_dir.join(LAST_MODEL_FILE);
    std::fs::write(&path, key)
        .map_err(|error| format!("Failed to write {}: {error}", path.display()))
}

fn api_key_looks_like_placeholder(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.starts_with('<') || !trimmed.is_ascii() {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    lower == "changethis"
        || [
            "optional",
            "placeholder",
            "replace_me",
            "replaceme",
            "your_api_key",
        ]
        .iter()
        .any(|pattern| lower.contains(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str, model: &str, env: &str) -> ProviderConfig {
        ProviderConfig {
            provider_name: name.to_string(),
            model_name: model.to_string(),
            models: None,
            api_key_env: env.to_string(),
            api_key: None,
            base_url: None,
        }
    }

    fn catalog(active: Option<&str>) -> ModelCatalog {
        let mut providers = HashMap::new();
        providers.insert(
            "gemini-2.5-flash".to_string(),
            provider(
                "gemini",
                "gemini-2.5-flash",
                "ISANAGENT_SETTINGS_TEST_GEMINI",
            ),
        );
        providers.insert(
            "gpt-4o".to_string(),
            provider("openai", "gpt-4o", "ISANAGENT_SETTINGS_TEST_OPENAI"),
        );
        ModelCatalog::new(
            PathBuf::from("/tmp/isanagent-settings-test"),
            providers,
            active.map(str::to_string),
        )
    }

    #[test]
    fn list_omits_secrets_and_sorts_by_provider() {
        let listed = list_models(&catalog(Some("gpt-4o")));
        assert_eq!(listed.active_key.as_deref(), Some("gpt-4o"));
        assert_eq!(listed.models.len(), 2);
        assert_eq!(listed.models[0].provider_name, "gemini");
        assert_eq!(listed.models[1].key, "gpt-4o");
        assert!(!listed.models.iter().any(|model| model.has_api_key));
        let encoded = serde_json::to_value(&listed).expect("json");
        assert!(encoded["models"][0].get("api_key").is_none());
    }

    #[test]
    fn unknown_active_key_is_cleared() {
        let listed = list_models(&catalog(Some("missing")));
        assert_eq!(listed.active_key, None);
    }

    #[test]
    fn switch_without_a_key_asks_for_one() {
        let error = prepare_model_switch(&catalog(None), "gpt-4o", None).unwrap_err();
        assert_eq!(
            error,
            ModelSwitchError::MissingApiKey {
                api_key_env: "ISANAGENT_SETTINGS_TEST_OPENAI".to_string(),
            }
        );
    }

    #[test]
    fn switch_rejects_placeholder_keys() {
        let error =
            prepare_model_switch(&catalog(None), "gpt-4o", Some("<changethis>")).unwrap_err();
        assert_eq!(error, ModelSwitchError::PlaceholderApiKey);
    }

    #[test]
    fn switch_uses_a_supplied_key_without_echoing_it_into_the_list_type() {
        let prepared =
            prepare_model_switch(&catalog(None), "gemini-2.5-flash", Some("test-key-value"))
                .expect("prepared");
        assert!(prepared.store_in_keychain);
        assert_eq!(prepared.api_key, "test-key-value");
        assert_eq!(prepared.provider_name, "gemini");
        assert!(prepared
            .base_url
            .contains("generativelanguage.googleapis.com"));
    }
}
