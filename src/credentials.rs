//! OS Keychain and secure credential storage for `isanagent` and `altai-app`.
//!
//! Stores and resolves provider API keys securely in the platform keychain
//! (Windows Credential Manager, macOS Keychain, Linux Secret Service / DBus)
//! under the canonical reverse-domain service namespace `dev.altai.isanagent`.

use log::{debug, warn};

/// Service name used for OS Keychain entries. Matches `altai-app` and the plugin extension namespace.
pub const KEYRING_SERVICE: &str = "dev.altai.isanagent";

/// Retrieve an API key for the given provider from the OS keychain.
/// Returns `None` if not found or if the platform keyring is unavailable.
pub fn get_provider_key(provider_name: &str) -> Option<String> {
    let provider = provider_name.trim().to_lowercase();
    if provider.is_empty() {
        return None;
    }
    match keyring::Entry::new(KEYRING_SERVICE, &provider) {
        Ok(entry) => match entry.get_password() {
            Ok(secret) => {
                let trimmed = secret.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    debug!("Resolved API key for provider '{provider}' from OS Keychain ({KEYRING_SERVICE})");
                    Some(trimmed)
                }
            }
            Err(keyring::Error::NoEntry) => None,
            Err(e) => {
                debug!("Keyring lookup for '{provider}' returned: {e}");
                None
            }
        },
        Err(e) => {
            debug!("Failed to initialize keyring entry for '{provider}': {e}");
            None
        }
    }
}

/// Store an API key for the given provider into the OS keychain.
pub fn set_provider_key(provider_name: &str, api_key: &str) -> Result<(), String> {
    let provider = provider_name.trim().to_lowercase();
    let secret = api_key.trim();
    if provider.is_empty() {
        return Err("Provider name cannot be empty".to_string());
    }
    if secret.is_empty() {
        return Err("API key cannot be empty".to_string());
    }
    let entry = keyring::Entry::new(KEYRING_SERVICE, &provider)
        .map_err(|e| format!("Failed to access OS Keychain: {e}"))?;
    entry
        .set_password(secret)
        .map_err(|e| format!("Failed to save key in OS Keychain: {e}"))?;
    debug!("Saved API key for provider '{provider}' in OS Keychain ({KEYRING_SERVICE})");
    Ok(())
}

/// Delete an API key for the given provider from the OS keychain.
pub fn delete_provider_key(provider_name: &str) -> Result<(), String> {
    let provider = provider_name.trim().to_lowercase();
    if provider.is_empty() {
        return Err("Provider name cannot be empty".to_string());
    }
    let entry = keyring::Entry::new(KEYRING_SERVICE, &provider)
        .map_err(|e| format!("Failed to access OS Keychain: {e}"))?;
    match entry.delete_credential() {
        Ok(()) => {
            debug!(
                "Deleted API key for provider '{provider}' from OS Keychain ({KEYRING_SERVICE})"
            );
            Ok(())
        }
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => {
            warn!("Failed to delete key from OS Keychain for '{provider}': {e}");
            Err(format!("Failed to delete key from OS Keychain: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_provider_key_empty_is_none() {
        assert_eq!(get_provider_key(""), None);
        assert_eq!(get_provider_key("   "), None);
    }
}
