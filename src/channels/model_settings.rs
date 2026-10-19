//! Web settings surface for choosing a configured model.
//!
//! The catalog is the same expanded provider map the terminal `/model` picker uses.
//! API keys are never serialized. A submitted key is stored in the OS keychain (kept in
//! memory for this process when the keychain cannot hold it) and applied to later
//! reasoning runs through `BusMessage::SwitchModel`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::Serialize;

use crate::config::ProviderConfig;

const LAST_MODEL_FILE: &str = ".system_generated/last_model";

/// Where provider API keys are written when a user enters one in the settings screen.
///
/// The production store is the OS keychain; tests swap in [`MemoryKeyStore`] so they never
/// touch a real keychain.
pub trait ProviderKeyStore: Send + Sync {
    fn get(&self, provider_name: &str) -> Option<String>;
    /// Persist the key. `Err` means it will not survive a restart.
    fn set(&self, provider_name: &str, api_key: &str) -> Result<(), String>;
    fn delete(&self, provider_name: &str) -> Result<(), String>;
}

/// OS keychain under [`crate::credentials::KEYRING_SERVICE`].
pub struct OsKeychainStore;

impl ProviderKeyStore for OsKeychainStore {
    fn get(&self, provider_name: &str) -> Option<String> {
        crate::credentials::get_provider_key(provider_name)
    }

    fn set(&self, provider_name: &str, api_key: &str) -> Result<(), String> {
        if !crate::credentials::keychain_is_persistent() {
            return Err("This build has no OS keychain backend.".to_string());
        }
        crate::credentials::set_provider_key(provider_name, api_key)
    }

    fn delete(&self, provider_name: &str) -> Result<(), String> {
        crate::credentials::delete_provider_key(provider_name)
    }
}

/// In-memory key store for tests. `persistent: false` behaves like a missing keychain.
#[cfg(test)]
pub(crate) struct MemoryKeyStore {
    pub persistent: bool,
    pub keys: RwLock<HashMap<String, String>>,
}

#[cfg(test)]
impl MemoryKeyStore {
    pub(crate) fn new(persistent: bool) -> Self {
        Self {
            persistent,
            keys: RwLock::new(HashMap::new()),
        }
    }
}

#[cfg(test)]
impl ProviderKeyStore for MemoryKeyStore {
    fn get(&self, provider_name: &str) -> Option<String> {
        self.keys
            .read()
            .unwrap()
            .get(&provider_slot(provider_name))
            .cloned()
    }

    fn set(&self, provider_name: &str, api_key: &str) -> Result<(), String> {
        if !self.persistent {
            return Err("no keychain".to_string());
        }
        self.keys
            .write()
            .unwrap()
            .insert(provider_slot(provider_name), api_key.to_string());
        Ok(())
    }

    fn delete(&self, provider_name: &str) -> Result<(), String> {
        self.keys
            .write()
            .unwrap()
            .remove(&provider_slot(provider_name));
        Ok(())
    }
}

/// Keychain entries are keyed by the lowercased provider name.
fn provider_slot(provider_name: &str) -> String {
    provider_name.trim().to_lowercase()
}

/// A key entered during this process. Kept so the provider stays usable when the keychain
/// cannot store it (or cannot read it back in the same session). Never serialized.
#[derive(Clone)]
struct SessionKey {
    secret: String,
    in_keychain: bool,
}

/// Where the key a model would use comes from, in the order `resolve_key` checks.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum KeySource {
    /// The environment variable named by `api_key_env` (wins over everything else).
    Env,
    /// The OS keychain.
    Keychain,
    /// Entered in this process but not persisted; lost on restart.
    Session,
    None,
}

/// Providers offered by the settings screen, plus the key of the model currently
/// selected for runs admitted after the next switch.
#[derive(Clone)]
pub struct ModelCatalog {
    pub workspace_dir: PathBuf,
    pub providers: HashMap<String, ProviderConfig>,
    /// The runtime default the agent uses (startup provider or the last switch).
    pub active_key: Arc<RwLock<Option<String>>>,
    /// The model picked in the web settings during this process. Starts empty on every
    /// start, so the web UI shows no current model until the user chooses one.
    selected_key: Arc<RwLock<Option<String>>>,
    key_store: Arc<dyn ProviderKeyStore>,
    session_keys: Arc<RwLock<HashMap<String, SessionKey>>>,
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
            selected_key: Arc::new(RwLock::new(None)),
            key_store: Arc::new(OsKeychainStore),
            session_keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Replace the key store (tests use an in-memory one).
    pub fn with_key_store(mut self, store: Arc<dyn ProviderKeyStore>) -> Self {
        self.key_store = store;
        self
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

    /// The model chosen in this process through the settings screen, if any.
    pub fn selected_key(&self) -> Option<String> {
        self.selected_key
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Record a successful selection: it becomes both the runtime default and the
    /// session's selected model.
    pub fn select(&self, key: String) {
        *self
            .selected_key
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(key.clone());
        self.set_active_key(key);
    }

    /// Provider name of the active model, when there is one.
    pub fn active_provider_name(&self) -> Option<String> {
        let key = self.active_key()?;
        self.providers
            .get(&key)
            .map(|cfg| cfg.provider_name.clone())
    }

    /// The configured provider name matching `name` (case-insensitive), if any.
    pub fn find_provider_name(&self, name: &str) -> Option<String> {
        let slot = provider_slot(name);
        if slot.is_empty() {
            return None;
        }
        let mut names: Vec<&String> = self
            .providers
            .values()
            .map(|cfg| &cfg.provider_name)
            .filter(|candidate| provider_slot(candidate) == slot)
            .collect();
        names.sort();
        names.first().map(|name| (*name).clone())
    }

    /// Resolve the key a model would use, mirroring `ProviderConfig::resolve_api_key`
    /// (env first, then keychain) plus keys entered during this session.
    pub fn resolve_key(&self, cfg: &ProviderConfig) -> Option<(String, KeySource)> {
        if let Some(key) = env_key(cfg) {
            return Some((key, KeySource::Env));
        }
        if let Some(session) = self.session_key(&cfg.provider_name) {
            let source = if session.in_keychain {
                KeySource::Keychain
            } else {
                KeySource::Session
            };
            return Some((session.secret, source));
        }
        self.key_store
            .get(&cfg.provider_name)
            .map(|key| key.trim().to_string())
            .filter(|key| !api_key_looks_like_placeholder(key))
            .map(|key| (key, KeySource::Keychain))
    }

    fn session_key(&self, provider_name: &str) -> Option<SessionKey> {
        self.session_keys
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&provider_slot(provider_name))
            .cloned()
    }

    /// Store a key for every model of `provider_name`. Returns whether the keychain kept it;
    /// either way it stays usable for the rest of this process.
    pub fn remember_provider_key(&self, provider_name: &str, api_key: &str) -> bool {
        let secret = api_key.trim().to_string();
        let in_keychain = self.key_store.set(provider_name, &secret).is_ok();
        self.session_keys
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                provider_slot(provider_name),
                SessionKey {
                    secret,
                    in_keychain,
                },
            );
        in_keychain
    }

    /// Drop the stored key (keychain and session copy) for `provider_name`.
    pub fn forget_provider_key(&self, provider_name: &str) -> Result<(), String> {
        self.session_keys
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&provider_slot(provider_name));
        self.key_store.delete(provider_name)
    }
}

fn env_key(cfg: &ProviderConfig) -> Option<String> {
    let env_var = cfg.resolved_api_key_env();
    if env_var.is_empty() {
        return None;
    }
    std::env::var(&env_var)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !api_key_looks_like_placeholder(value))
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelEntry {
    pub key: String,
    pub provider_name: String,
    pub model_name: String,
    pub has_api_key: bool,
    pub key_source: KeySource,
    pub api_key_env: String,
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelList {
    /// Runtime default the agent would use for the next run.
    pub active_key: Option<String>,
    /// Model chosen in the web settings during this process; `None` after every start.
    pub selected_key: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderKeyError {
    UnknownProvider,
    PlaceholderApiKey,
    /// Removing the key would leave the active model without one.
    KeyInUse {
        model_name: String,
    },
}

pub fn list_models(catalog: &ModelCatalog) -> ModelList {
    let mut models: Vec<ModelEntry> = catalog
        .providers
        .iter()
        .map(|(key, cfg)| {
            let key_source = catalog
                .resolve_key(cfg)
                .map_or(KeySource::None, |(_, source)| source);
            ModelEntry {
                key: key.clone(),
                provider_name: cfg.provider_name.clone(),
                model_name: cfg.model_name.clone(),
                has_api_key: key_source != KeySource::None,
                key_source,
                api_key_env: cfg.resolved_api_key_env(),
                base_url: cfg.resolved_base_url().unwrap_or_default(),
            }
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
    let selected_key = catalog
        .selected_key()
        .filter(|key| models.iter().any(|model| &model.key == key));
    ModelList {
        active_key,
        selected_key,
        models,
    }
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
        match catalog.resolve_key(cfg) {
            Some((existing, _)) => (existing, false),
            None => {
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

/// Validate a key submitted for `provider_name`. Returns the configured provider name.
pub fn validate_provider_key(
    catalog: &ModelCatalog,
    provider_name: &str,
    api_key: &str,
) -> Result<String, ProviderKeyError> {
    let provider = catalog
        .find_provider_name(provider_name)
        .ok_or(ProviderKeyError::UnknownProvider)?;
    if api_key_looks_like_placeholder(api_key) {
        return Err(ProviderKeyError::PlaceholderApiKey);
    }
    Ok(provider)
}

/// The switch that re-applies the active model's key after `provider_name`'s stored key
/// changed, or `None` when the active model belongs to another provider.
pub fn active_switch_for_provider(
    catalog: &ModelCatalog,
    provider_name: &str,
) -> Option<PreparedModelSwitch> {
    let active = catalog.active_key()?;
    let cfg = catalog.providers.get(&active)?;
    if provider_slot(&cfg.provider_name) != provider_slot(provider_name) {
        return None;
    }
    prepare_model_switch(catalog, &active, None).ok()
}

/// Decide whether the stored key for `provider_name` may be removed. Refuses when the
/// active model uses that provider and no environment key would take over.
pub fn check_provider_key_removal(
    catalog: &ModelCatalog,
    provider_name: &str,
) -> Result<String, ProviderKeyError> {
    let provider = catalog
        .find_provider_name(provider_name)
        .ok_or(ProviderKeyError::UnknownProvider)?;
    if let Some(active) = catalog.active_key() {
        if let Some(cfg) = catalog.providers.get(&active) {
            if provider_slot(&cfg.provider_name) == provider_slot(&provider)
                && env_key(cfg).is_none()
            {
                return Err(ProviderKeyError::KeyInUse {
                    model_name: cfg.model_name.clone(),
                });
            }
        }
    }
    Ok(provider)
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
        .with_key_store(Arc::new(MemoryKeyStore::new(true)))
    }

    fn catalog_with_store(active: Option<&str>, store: MemoryKeyStore) -> ModelCatalog {
        catalog(active).with_key_store(Arc::new(store))
    }

    fn source_of(list: &ModelList, key: &str) -> KeySource {
        list.models
            .iter()
            .find(|model| model.key == key)
            .expect("model listed")
            .key_source
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
    fn selected_key_starts_empty_and_is_set_by_select() {
        let catalog = catalog(Some("gpt-4o"));
        let listed = list_models(&catalog);
        assert_eq!(listed.active_key.as_deref(), Some("gpt-4o"));
        assert_eq!(listed.selected_key, None);
        let encoded = serde_json::to_value(&listed).expect("json");
        assert!(encoded["selected_key"].is_null());

        catalog.select("gemini-2.5-flash".to_string());
        let listed = list_models(&catalog);
        assert_eq!(listed.selected_key.as_deref(), Some("gemini-2.5-flash"));
        assert_eq!(listed.active_key.as_deref(), Some("gemini-2.5-flash"));

        // A fresh catalog (a restart) forgets the selection again.
        assert_eq!(
            list_models(&catalog_with_store(
                Some("gemini-2.5-flash"),
                MemoryKeyStore::new(true)
            ))
            .selected_key,
            None
        );
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

    #[test]
    fn key_source_reports_none_keychain_session_and_env() {
        let keyed = catalog(None);
        assert_eq!(source_of(&list_models(&keyed), "gpt-4o"), KeySource::None);

        assert!(keyed.remember_provider_key("openai", "sk-real-value"));
        let listed = list_models(&keyed);
        assert_eq!(source_of(&listed, "gpt-4o"), KeySource::Keychain);
        assert!(listed
            .models
            .iter()
            .any(|m| m.key == "gpt-4o" && m.has_api_key));
        let encoded = serde_json::to_string(&listed).expect("json");
        assert!(!encoded.contains("sk-real-value"));
        assert!(encoded.contains(r#""key_source":"keychain""#));

        let no_keychain = catalog_with_store(None, MemoryKeyStore::new(false));
        assert!(!no_keychain.remember_provider_key("openai", "sk-real-value"));
        assert_eq!(
            source_of(&list_models(&no_keychain), "gpt-4o"),
            KeySource::Session
        );

        let mut env_catalog = catalog(None);
        let env = "ISANAGENT_SETTINGS_TEST_ENV_SOURCE";
        std::env::set_var(env, "env-real-value");
        env_catalog
            .providers
            .get_mut("gpt-4o")
            .expect("gpt-4o")
            .api_key_env = env.to_string();
        env_catalog.remember_provider_key("openai", "sk-real-value");
        let (resolved, source) = env_catalog
            .resolve_key(&env_catalog.providers["gpt-4o"])
            .expect("resolved");
        std::env::remove_var(env);
        assert_eq!(source, KeySource::Env);
        assert_eq!(resolved, "env-real-value");
    }

    #[test]
    fn keychain_entries_from_earlier_runs_are_listed() {
        let store = MemoryKeyStore::new(true);
        store.set("Gemini", "AIza-from-keychain").expect("set");
        let catalog = catalog_with_store(None, store);
        assert_eq!(
            source_of(&list_models(&catalog), "gemini-2.5-flash"),
            KeySource::Keychain
        );
    }

    #[test]
    fn a_saved_key_makes_the_provider_switchable() {
        let catalog = catalog_with_store(None, MemoryKeyStore::new(false));
        catalog.remember_provider_key("openai", "sk-session-only");
        let prepared = prepare_model_switch(&catalog, "gpt-4o", None).expect("prepared");
        assert_eq!(prepared.api_key, "sk-session-only");
        assert!(!prepared.store_in_keychain);
    }

    #[test]
    fn provider_key_validation_rejects_placeholders_and_unknown_providers() {
        let catalog = catalog(None);
        assert_eq!(
            validate_provider_key(&catalog, "openai", "<changethis>"),
            Err(ProviderKeyError::PlaceholderApiKey)
        );
        assert_eq!(
            validate_provider_key(&catalog, "openai", "your_api_key_here"),
            Err(ProviderKeyError::PlaceholderApiKey)
        );
        assert_eq!(
            validate_provider_key(&catalog, "openai", "   "),
            Err(ProviderKeyError::PlaceholderApiKey)
        );
        assert_eq!(
            validate_provider_key(&catalog, "mistral", "sk-real"),
            Err(ProviderKeyError::UnknownProvider)
        );
        assert_eq!(
            validate_provider_key(&catalog, "OpenAI", "sk-real").as_deref(),
            Ok("openai")
        );
    }

    #[test]
    fn active_switch_only_for_the_active_provider() {
        let catalog = catalog(Some("gpt-4o"));
        catalog.remember_provider_key("openai", "sk-new");
        let switch = active_switch_for_provider(&catalog, "openai").expect("switch");
        assert_eq!(switch.api_key, "sk-new");
        assert_eq!(switch.key, "gpt-4o");
        catalog.remember_provider_key("gemini", "AIza-new");
        assert!(active_switch_for_provider(&catalog, "gemini").is_none());
    }

    #[test]
    fn removing_the_active_providers_only_key_is_refused() {
        let catalog = catalog(Some("gpt-4o"));
        catalog.remember_provider_key("openai", "sk-new");
        catalog.remember_provider_key("gemini", "AIza-new");
        assert_eq!(
            check_provider_key_removal(&catalog, "openai"),
            Err(ProviderKeyError::KeyInUse {
                model_name: "gpt-4o".to_string()
            })
        );
        assert_eq!(
            check_provider_key_removal(&catalog, "gemini").as_deref(),
            Ok("gemini")
        );
        catalog.forget_provider_key("gemini").expect("forget");
        assert_eq!(
            source_of(&list_models(&catalog), "gemini-2.5-flash"),
            KeySource::None
        );
        assert_eq!(
            check_provider_key_removal(&catalog, "nope"),
            Err(ProviderKeyError::UnknownProvider)
        );
    }

    #[test]
    fn removing_the_active_providers_key_is_allowed_when_env_remains() {
        let mut catalog = catalog(Some("gpt-4o"));
        let env = "ISANAGENT_SETTINGS_TEST_ENV_REMOVAL";
        std::env::set_var(env, "env-real-value");
        catalog
            .providers
            .get_mut("gpt-4o")
            .expect("gpt-4o")
            .api_key_env = env.to_string();
        let allowed = check_provider_key_removal(&catalog, "openai");
        std::env::remove_var(env);
        assert_eq!(allowed.as_deref(), Ok("openai"));
    }
}
