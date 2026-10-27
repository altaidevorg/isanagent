//! OS Keychain and secure credential storage for `isanagent` and `altai-app`.
//!
//! Stores and resolves provider API keys securely in the platform keychain
//! (Windows Credential Manager, macOS Keychain, Linux Secret Service / DBus)
//! under the canonical reverse-domain service namespace `dev.altai.isanagent`.
//!
//! ## One keychain read per provider per process
//!
//! Every keychain read can raise an OS consent prompt (macOS asks whenever the calling
//! binary is not on the item's access list, which includes every fresh `cargo build`).
//! Callers resolve keys per *model* entry: start-up builds the failover list from all
//! expanded `[providers.*]` models, and the settings screens probe every provider. Reading
//! the keychain on each of those calls produced one prompt per model, so the dialog looked
//! like it rejected a correct password.
//!
//! Lookups therefore go through a process-wide cache keyed by the normalised provider
//! name: the keychain is read at most once per provider, concurrent callers wait for that
//! single read, and the outcome (key, no entry, or access denied/failed) is remembered.
//! Writes and deletes made through this module update the cache. A key changed outside
//! this process (Keychain Access, `altai-app`) is picked up on the next start.
//!
//! If saving a key to the keychain fails (access denied, locked keychain, no backend),
//! the key is still kept in the cache for the rest of the process and the caller gets an
//! `Err` describing why it was not persisted. A key the user just typed is therefore
//! always usable for the session; only persistence depends on the keychain.
//!
//! ## Keychain is opt-in
//!
//! By default isanagent does **not** touch the OS keychain: a key typed in `/settings`,
//! `/model` or `/key` is kept in memory for the current run only, and the user enters it
//! again after a restart (or sets the provider's environment variable, which always wins).
//! This avoids macOS keychain password dialogs, which every rebuilt binary triggers.
//! Set `ISANAGENT_KEYCHAIN=1` to read and save keys in the OS keychain instead.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use log::{debug, warn};

/// Service name used for OS Keychain entries. Matches `altai-app` and the plugin extension namespace.
pub const KEYRING_SERVICE: &str = "dev.altai.isanagent";

/// Environment variable that turns OS keychain storage on (`1`, `true`, `yes`, `on`).
/// Off by default: keys typed at runtime last for the current process only.
pub const KEYCHAIN_ENV: &str = "ISANAGENT_KEYCHAIN";

/// Whether API keys are read from and saved to the OS keychain in this process.
pub fn keychain_enabled() -> bool {
    os_key_cache().enabled
}

/// Whether keys written through this module survive a process restart.
///
/// `keyring` falls back to its in-memory mock store when no platform backend feature is
/// compiled in. Writes to that store succeed but are gone as soon as the entry is
/// dropped, so callers that promise "saved in keychain" must check this first.
pub fn keychain_is_persistent() -> bool {
    use keyring::credential::CredentialPersistence;
    if !keychain_enabled() {
        return false;
    }
    matches!(
        keyring::default::default_credential_builder().persistence(),
        CredentialPersistence::UntilDelete | CredentialPersistence::UntilReboot
    )
}

fn keychain_enabled_by_env() -> bool {
    std::env::var(KEYCHAIN_ENV)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn normalize_provider(provider_name: &str) -> Option<String> {
    let provider = provider_name.trim().to_lowercase();
    (!provider.is_empty()).then_some(provider)
}

/// Raw keychain operations. Production uses [`OsKeychain`]; tests use a counting fake.
trait KeychainBackend: Send + Sync {
    /// `Ok(None)` = no entry. `Err` = the read failed or the user denied access.
    fn get(&self, provider: &str) -> Result<Option<String>, String>;
    fn set(&self, provider: &str, secret: &str) -> Result<(), String>;
    fn delete(&self, provider: &str) -> Result<(), String>;
}

struct OsKeychain;

impl KeychainBackend for OsKeychain {
    fn get(&self, provider: &str) -> Result<Option<String>, String> {
        let entry = keyring::Entry::new(KEYRING_SERVICE, provider).map_err(|e| e.to_string())?;
        match entry.get_password() {
            Ok(secret) => Ok(Some(secret)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    fn set(&self, provider: &str, secret: &str) -> Result<(), String> {
        let entry = keyring::Entry::new(KEYRING_SERVICE, provider)
            .map_err(|e| format!("Failed to access OS Keychain: {e}"))?;
        entry
            .set_password(secret)
            .map_err(|e| format!("Failed to save key in OS Keychain: {e}"))
    }

    fn delete(&self, provider: &str) -> Result<(), String> {
        let entry = keyring::Entry::new(KEYRING_SERVICE, provider)
            .map_err(|e| format!("Failed to access OS Keychain: {e}"))?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(format!("Failed to delete key from OS Keychain: {e}")),
        }
    }
}

/// Process-wide memo of keychain lookups (see the module docs).
struct KeyCache<B: KeychainBackend> {
    backend: B,
    enabled: bool,
    /// provider -> resolved key (`None` = no entry, or access denied / failed).
    /// The lock is held across the backend read on purpose: concurrent callers for any
    /// provider wait for the in-flight read instead of raising a second OS prompt.
    entries: Mutex<HashMap<String, Option<String>>>,
}

impl<B: KeychainBackend> KeyCache<B> {
    fn new(backend: B, enabled: bool) -> Self {
        Self {
            backend,
            enabled,
            entries: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Option<String>>> {
        // A panic while holding the lock cannot leave the map inconsistent (single
        // insert), so recover from poisoning instead of failing every later lookup.
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn get(&self, provider: &str) -> Option<String> {
        let mut entries = self.lock();
        if !self.enabled {
            // Keychain off: only keys entered during this session.
            return entries.get(provider).cloned().flatten();
        }
        if let Some(cached) = entries.get(provider) {
            return cached.clone();
        }
        let resolved = match self.backend.get(provider) {
            Ok(Some(secret)) => {
                let trimmed = secret.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    debug!("Resolved API key for provider '{provider}' from OS Keychain ({KEYRING_SERVICE})");
                    Some(trimmed)
                }
            }
            Ok(None) => None,
            Err(e) => {
                warn!(
                    "OS Keychain read for provider '{provider}' failed or was denied ({e}). \
                     isanagent will not ask again until restart; set the provider's API key \
                     environment variable or allow access next time."
                );
                None
            }
        };
        entries.insert(provider.to_string(), resolved.clone());
        resolved
    }

    fn set(&self, provider: &str, secret: &str) -> Result<(), String> {
        let mut entries = self.lock();
        // Usable for this process whatever happens to the keychain write below.
        entries.insert(provider.to_string(), Some(secret.to_string()));
        if !self.enabled {
            // Session-only storage is the intended mode, not an error.
            return Ok(());
        }
        match self.backend.set(provider, secret) {
            Ok(()) => {
                debug!(
                    "Saved API key for provider '{provider}' in OS Keychain ({KEYRING_SERVICE})"
                );
                Ok(())
            }
            Err(e) => {
                warn!("API key for '{provider}' kept for this session only: {e}");
                Err(format!("{e} (the key is used for this session only)"))
            }
        }
    }

    fn delete(&self, provider: &str) -> Result<(), String> {
        let mut entries = self.lock();
        if !self.enabled {
            entries.remove(provider);
            return Ok(());
        }
        match self.backend.delete(provider) {
            Ok(()) => {
                entries.insert(provider.to_string(), None);
                debug!("Deleted API key for provider '{provider}' from OS Keychain ({KEYRING_SERVICE})");
                Ok(())
            }
            Err(e) => {
                // State unknown after a failed delete: forget it so the next lookup re-reads.
                entries.remove(provider);
                warn!("Failed to delete key from OS Keychain for '{provider}': {e}");
                Err(e)
            }
        }
    }
}

fn os_key_cache() -> &'static KeyCache<OsKeychain> {
    static CACHE: OnceLock<KeyCache<OsKeychain>> = OnceLock::new();
    CACHE.get_or_init(|| KeyCache::new(OsKeychain, keychain_enabled_by_env()))
}

/// Retrieve an API key for the given provider from the OS keychain.
/// Returns `None` if not found, if access was denied, or if the platform keyring is
/// unavailable. The keychain is read at most once per provider per process.
pub fn get_provider_key(provider_name: &str) -> Option<String> {
    let provider = normalize_provider(provider_name)?;
    os_key_cache().get(&provider)
}

/// Store an API key for the given provider into the OS keychain.
pub fn set_provider_key(provider_name: &str, api_key: &str) -> Result<(), String> {
    let provider = normalize_provider(provider_name)
        .ok_or_else(|| "Provider name cannot be empty".to_string())?;
    let secret = api_key.trim();
    if secret.is_empty() {
        return Err("API key cannot be empty".to_string());
    }
    os_key_cache().set(&provider, secret)
}

/// Delete an API key for the given provider from the OS keychain.
pub fn delete_provider_key(provider_name: &str) -> Result<(), String> {
    let provider = normalize_provider(provider_name)
        .ok_or_else(|| "Provider name cannot be empty".to_string())?;
    os_key_cache().delete(&provider)
}

/// Why an interactively entered API key was refused by [`normalize_api_key_input`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKeyInputError {
    /// Nothing left after removing whitespace and invisible characters.
    Empty,
    /// A character that cannot be sent in an HTTP header. `position` is 1-based and counts
    /// characters of the cleaned key (whitespace and invisible characters already removed).
    NonAscii { ch: char, position: usize },
    /// A template value such as `<changethis>` or `your_api_key_here`.
    Placeholder,
}

impl std::fmt::Display for ApiKeyInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "The API key is empty."),
            Self::NonAscii { ch, position } => write!(
                f,
                "The key contains a non-ASCII character (U+{:04X} '{}' at position {position}). \
                 API keys are plain ASCII \u{2014} check for a stray character from copy-paste.",
                u32::from(*ch),
                ch.escape_debug()
            ),
            Self::Placeholder => write!(f, "That doesn't look like a real API key."),
        }
    }
}

impl std::error::Error for ApiKeyInputError {}

/// Invisible format characters that copy-paste commonly drags along with a key.
fn is_invisible_paste_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{200B}'..='\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{00AD}'
    )
}

/// True for template values that are not real keys. Expects an already-cleaned key.
fn api_key_is_placeholder(key: &str) -> bool {
    if key.starts_with('<') {
        return true;
    }
    let lower = key.to_ascii_lowercase();
    lower == "changethis"
        || [
            "optional",
            "placeholder",
            "replace_me",
            "replaceme",
            "your_api_key",
            "changeme",
        ]
        .iter()
        .any(|pattern| lower.contains(pattern))
}

/// Clean an API key typed or pasted by a user and check that it can be used.
///
/// Removes all whitespace (including no-break spaces and newlines) and invisible format
/// characters (zero-width space/joiners, word joiner, BOM, soft hyphen). The result must be
/// non-empty plain ASCII (keys travel in HTTP headers) and must not be a template value.
/// Returns the cleaned key, which is what callers should store and use.
pub fn normalize_api_key_input(raw: &str) -> Result<String, ApiKeyInputError> {
    let cleaned: String = raw
        .chars()
        .filter(|ch| !ch.is_whitespace() && !is_invisible_paste_char(*ch))
        .collect();
    if cleaned.is_empty() {
        return Err(ApiKeyInputError::Empty);
    }
    if let Some((index, ch)) = cleaned.chars().enumerate().find(|(_, ch)| !ch.is_ascii()) {
        return Err(ApiKeyInputError::NonAscii {
            ch,
            position: index + 1,
        });
    }
    if api_key_is_placeholder(&cleaned) {
        return Err(ApiKeyInputError::Placeholder);
    }
    Ok(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_provider_key_empty_is_none() {
        assert_eq!(get_provider_key(""), None);
        assert_eq!(get_provider_key("   "), None);
    }

    #[test]
    fn normalize_strips_invisible_and_whitespace_characters() {
        for raw in [
            "\u{200B}sk-abc123",
            "\u{FEFF}sk-abc123",
            "sk-abc\u{00A0}123",
            "sk-abc123\n",
            " sk-abc123\r\n",
            "sk-\u{200C}abc\u{200D}123\u{2060}\u{00AD}",
        ] {
            assert_eq!(
                normalize_api_key_input(raw).as_deref(),
                Ok("sk-abc123"),
                "{raw:?}"
            );
        }
    }

    #[test]
    fn normalize_reports_the_first_non_ascii_character() {
        let error = normalize_api_key_input("sk-ab\u{e7}d").unwrap_err();
        assert_eq!(
            error,
            ApiKeyInputError::NonAscii {
                ch: '\u{e7}',
                position: 6
            }
        );
        let message = error.to_string();
        assert!(message.contains("U+00E7"), "{message}");
        assert!(message.contains("position 6"), "{message}");
        // Position counts the cleaned key: the leading zero-width space is not counted.
        assert_eq!(
            normalize_api_key_input("\u{200B}sk-ab\u{e7}d"),
            Err(ApiKeyInputError::NonAscii {
                ch: '\u{e7}',
                position: 6
            })
        );
    }

    #[test]
    fn normalize_rejects_empty_and_placeholders() {
        assert_eq!(normalize_api_key_input(""), Err(ApiKeyInputError::Empty));
        assert_eq!(
            normalize_api_key_input(" \u{200B}\n"),
            Err(ApiKeyInputError::Empty)
        );
        for bad in [
            "<changethis>",
            "changethis",
            "CHANGEME",
            "YOUR_API_KEY",
            "your_api_key_here",
            "replace_me",
            "placeholder",
        ] {
            assert_eq!(
                normalize_api_key_input(bad),
                Err(ApiKeyInputError::Placeholder),
                "{bad:?}"
            );
        }
    }

    // --- Keychain read cache (one OS prompt per provider per process) ---

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct FakeKeychain {
        reads: AtomicUsize,
        writes: AtomicUsize,
        store: Mutex<HashMap<String, String>>,
        deny_reads: bool,
        read_delay_ms: u64,
    }

    impl KeychainBackend for Arc<FakeKeychain> {
        fn get(&self, provider: &str) -> Result<Option<String>, String> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if self.read_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.read_delay_ms));
            }
            if self.deny_reads {
                return Err("User canceled the operation.".into());
            }
            Ok(self.store.lock().unwrap().get(provider).cloned())
        }
        fn set(&self, provider: &str, secret: &str) -> Result<(), String> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            self.store
                .lock()
                .unwrap()
                .insert(provider.into(), secret.into());
            Ok(())
        }
        fn delete(&self, provider: &str) -> Result<(), String> {
            self.store.lock().unwrap().remove(provider);
            Ok(())
        }
    }

    fn fake_with(keys: &[(&str, &str)]) -> Arc<FakeKeychain> {
        let fake = FakeKeychain::default();
        for (p, k) in keys {
            fake.store.lock().unwrap().insert((*p).into(), (*k).into());
        }
        Arc::new(fake)
    }

    #[test]
    fn many_lookups_read_the_keychain_once_per_provider() {
        // Start-up resolves the key once per expanded model (8 Gemini models, 6 OpenAI…).
        let fake = fake_with(&[("gemini", "AIza-test")]);
        let cache = KeyCache::new(fake.clone(), true);
        for _ in 0..8 {
            assert_eq!(cache.get("gemini").as_deref(), Some("AIza-test"));
        }
        for _ in 0..6 {
            assert_eq!(cache.get("openai"), None); // no entry: cached as well
        }
        assert_eq!(fake.reads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn provider_names_are_normalized_before_caching() {
        let fake = fake_with(&[("gemini", "k")]);
        let cache = KeyCache::new(fake.clone(), true);
        for name in ["gemini", "Gemini", "  GEMINI "] {
            let provider = normalize_provider(name).unwrap();
            assert_eq!(cache.get(&provider).as_deref(), Some("k"));
        }
        assert_eq!(fake.reads.load(Ordering::SeqCst), 1);
        assert_eq!(normalize_provider("   "), None);
    }

    #[test]
    fn denied_access_is_remembered_instead_of_prompting_again() {
        let fake = Arc::new(FakeKeychain {
            deny_reads: true,
            ..Default::default()
        });
        let cache = KeyCache::new(fake.clone(), true);
        for _ in 0..10 {
            assert_eq!(cache.get("gemini"), None);
        }
        assert_eq!(fake.reads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn concurrent_lookups_share_one_read() {
        let fake = Arc::new(FakeKeychain {
            read_delay_ms: 50,
            ..Default::default()
        });
        fake.store
            .lock()
            .unwrap()
            .insert("anthropic".into(), "sk-ant".into());
        let cache = Arc::new(KeyCache::new(fake.clone(), true));
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let cache = cache.clone();
                std::thread::spawn(move || cache.get("anthropic"))
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap().as_deref(), Some("sk-ant"));
        }
        assert_eq!(fake.reads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_keychain_write_still_keeps_the_key_for_this_session() {
        struct BrokenKeychain;
        impl KeychainBackend for BrokenKeychain {
            fn get(&self, _: &str) -> Result<Option<String>, String> {
                Err("User interaction is not allowed.".into())
            }
            fn set(&self, _: &str, _: &str) -> Result<(), String> {
                Err("Failed to save key in OS Keychain: The user name or passphrase you entered is not correct.".into())
            }
            fn delete(&self, _: &str) -> Result<(), String> {
                Ok(())
            }
        }
        let cache = KeyCache::new(BrokenKeychain, true);
        assert_eq!(cache.get("gemini"), None); // denied read, cached
        let err = cache.set("gemini", "AIza-typed").unwrap_err();
        assert!(err.contains("session only"), "{err}");
        assert_eq!(cache.get("gemini").as_deref(), Some("AIza-typed"));
    }

    #[test]
    fn set_and_delete_update_the_cache_without_rereading() {
        let fake = fake_with(&[]);
        let cache = KeyCache::new(fake.clone(), true);
        assert_eq!(cache.get("openai"), None);
        cache.set("openai", "sk-new").unwrap();
        assert_eq!(cache.get("openai").as_deref(), Some("sk-new"));
        cache.delete("openai").unwrap();
        assert_eq!(cache.get("openai"), None);
        assert_eq!(fake.reads.load(Ordering::SeqCst), 1);
        assert_eq!(fake.writes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn disabled_cache_never_touches_the_keychain() {
        let fake = fake_with(&[("gemini", "k")]);
        let cache = KeyCache::new(fake.clone(), false);
        assert_eq!(cache.get("gemini"), None);
        // Typed keys work for the session; they are just not persisted.
        cache.set("gemini", "x").unwrap();
        assert_eq!(cache.get("gemini").as_deref(), Some("x"));
        assert!(cache.delete("gemini").is_ok());
        assert_eq!(cache.get("gemini"), None);
        assert_eq!(fake.reads.load(Ordering::SeqCst), 0);
        assert_eq!(fake.writes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn whitespace_only_secret_counts_as_missing() {
        let fake = fake_with(&[("deepseek", "   ")]);
        let cache = KeyCache::new(fake.clone(), true);
        assert_eq!(cache.get("deepseek"), None);
        assert_eq!(cache.get("deepseek"), None);
        assert_eq!(fake.reads.load(Ordering::SeqCst), 1);
    }
}
