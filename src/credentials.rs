//! OS Keychain and secure credential storage for `isanagent` and `altai-app`.
//!
//! Stores and resolves provider API keys securely in the platform keychain
//! (Windows Credential Manager, macOS Keychain, Linux Secret Service / DBus)
//! under the canonical reverse-domain service namespace `dev.altai.isanagent`.

use log::{debug, warn};

/// Service name used for OS Keychain entries. Matches `altai-app` and the plugin extension namespace.
pub const KEYRING_SERVICE: &str = "dev.altai.isanagent";

/// Whether keys written through this module survive a process restart.
///
/// `keyring` falls back to its in-memory mock store when no platform backend feature is
/// compiled in. Writes to that store succeed but are gone as soon as the entry is
/// dropped, so callers that promise "saved in keychain" must check this first.
pub fn keychain_is_persistent() -> bool {
    use keyring::credential::CredentialPersistence;
    matches!(
        keyring::default::default_credential_builder().persistence(),
        CredentialPersistence::UntilDelete | CredentialPersistence::UntilReboot
    )
}

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
}
