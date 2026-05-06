//! Registry of well-known LLM provider names and their OpenAI-compatible chat-completions URLs.
//!
//! The agent's `[provider]` config block uses a `provider_name` key. When that name appears in
//! [`KNOWN_PROVIDERS`] the base URL is implied and no `base_url` field is required in
//! `config.toml`. The reserved name [`OPENAI_COMPATIBLE`] indicates a third-party endpoint that
//! speaks the OpenAI Chat Completions protocol; for that case `base_url` becomes mandatory.
//!
//! An explicit `base_url` in the config always wins, including for known providers (this lets
//! users point a known provider at a proxy / Azure-OpenAI relay / self-hosted gateway).

/// Reserved provider name for any OpenAI-compatible third-party endpoint that requires the user
/// to supply a base URL explicitly (no built-in default).
pub const OPENAI_COMPATIBLE: &str = "openai_compatible";

/// Known LLM providers and their OpenAI-compatible chat-completions URLs.
///
/// Kept alphabetical so `known_names()` returns a deterministic order suitable for help text
/// and error messages.
pub const KNOWN_PROVIDERS: &[(&str, &str)] = &[
    ("anthropic", "https://api.anthropic.com/v1/messages"),
    ("deepseek", "https://api.deepseek.com/v1/chat/completions"),
    (
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    ),
    ("openai", "https://api.openai.com/v1/chat/completions"),
    (
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
    ),
];

/// Look up the chat-completions URL for a known provider name. Case-sensitive on the canonical
/// lower-case keys in [`KNOWN_PROVIDERS`]; callers should normalize user input to lower-case
/// before invoking this.
pub fn lookup(name: &str) -> Option<&'static str> {
    KNOWN_PROVIDERS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, u)| *u)
}

/// All known provider names (alphabetical, excluding the [`OPENAI_COMPATIBLE`] sentinel). Useful
/// for building human-readable error messages like
/// `"unknown provider_name 'foo'; expected one of [.., openai_compatible]"`.
pub fn known_names() -> Vec<&'static str> {
    KNOWN_PROVIDERS.iter().map(|(n, _)| *n).collect()
}

/// Returns true when the given name is either a built-in known provider or the
/// `openai_compatible` sentinel.
pub fn is_recognized(name: &str) -> bool {
    name == OPENAI_COMPATIBLE || lookup(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_are_alphabetical() {
        let names = known_names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "KNOWN_PROVIDERS should stay alphabetical");
    }

    #[test]
    fn lookup_returns_url_for_each_known_name() {
        for name in known_names() {
            let url = lookup(name).unwrap_or_else(|| panic!("missing url for {name}"));
            assert!(url.starts_with("https://"), "url for {name} must be https");
            // Anthropic uses /messages, all others use /chat/completions
            assert!(
                url.ends_with("/chat/completions") || url.ends_with("/messages"),
                "url for {name} must end in /chat/completions or /messages"
            );
        }
    }

    #[test]
    fn lookup_returns_none_for_unknown() {
        assert!(lookup("nope").is_none());
        assert!(
            lookup(OPENAI_COMPATIBLE).is_none(),
            "OPENAI_COMPATIBLE has no built-in URL on purpose"
        );
    }

    #[test]
    fn is_recognized_covers_known_and_sentinel() {
        for name in known_names() {
            assert!(is_recognized(name), "{name} should be recognized");
        }
        assert!(is_recognized(OPENAI_COMPATIBLE));
        assert!(!is_recognized("bogus"));
    }
}
