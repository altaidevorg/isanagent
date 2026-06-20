//! Secret redaction for **observability output** (currently the observation hook's telemetry JSONL
//! journal + webhook sink).
//!
//! The executed child still receives the real host environment — this module only sanitizes
//! monitoring output, so it never breaks the agent's own use of API keys/tokens. Its job is to keep
//! a secret that lands in tool output (an `env` / `printenv` dump, an echoed `$OPENAI_API_KEY`, a
//! key printed by a script) from being written to disk or POSTed to a third-party webhook.
//!
//! Two redaction strategies, applied in order:
//! 1. **By value** — for each env var whose *name* looks secret (`*_KEY`, `*TOKEN*`, `*SECRET*`,
//!    `*PASSWORD*`, `*_DSN`, …) and whose value is long enough to be a real credential, replace any
//!    exact occurrence of that value with `[REDACTED:<NAME>]`. Catches secrets regardless of format.
//! 2. **By format** — regexes for recognizable credential shapes (OpenAI `sk-…`, Stripe `sk_live_…`,
//!    AWS `AKIA…`, HuggingFace `hf_…`, GitHub `ghp_…`, GitLab `glpat-…`, Slack `xox…`, Google
//!    `AIza…`, npm `npm_…`, JWTs, `Bearer …`, PEM private-key blocks, and generic
//!    `secret/api_key/token/password = <value>` assignments). Catches secrets not present in *this*
//!    process's environment.
//!
//! Over-redaction in a monitoring sink is harmless (you lose some telemetry detail, never break the
//! agent), so the patterns are deliberately generous. The `regex` crate is linear-time, so the
//! generous patterns carry no ReDoS risk.
//!
//! Scope: this covers every persisted telemetry sink — the observation hook (JSONL + webhook), the
//! `conversation.jsonl` analytical log, and the execution `run.json` / `source.txt` journals (via
//! [`shared`]). The model's own context and provider-side logs are intentionally out of scope (the
//! agent needs real keys to do its work).

use regex::Regex;
use serde_json::Value;
use std::borrow::Cow;
use std::sync::LazyLock;

/// Process-wide redactor, built once from the environment. For sink writers that can't easily thread
/// an instance through (the conversation + execution journals). The observation hook builds its own
/// via [`SecretRedactor::from_env`] at startup. The process environment is stable, so a one-time
/// build is correct.
pub fn shared() -> &'static SecretRedactor {
    static SHARED: LazyLock<SecretRedactor> = LazyLock::new(SecretRedactor::from_env);
    &SHARED
}

/// Minimum length for an env value to be redacted by value. Short values (`DEBUG=1`, `TZ=UTC`) would
/// cause noisy false-positive substring matches in otherwise-normal output. Short secrets without a
/// recognizable format (e.g. a 6-char DB password) are an accepted gap.
const MIN_SECRET_VALUE_LEN: usize = 8;

/// Guards `redact_json` recursion against a pathologically deep JSON tree (well above serde_json's
/// own default parse depth of 128). Deeper subtrees are left unredacted rather than overflowing.
const MAX_JSON_DEPTH: usize = 256;

/// Matches env-var / JSON-key *names* that look like they hold a secret (`*_KEY`, `*TOKEN*`,
/// `*SECRET*`, `*PASSWORD*`, `*_DSN`, …). Compiled once and reused for both the env-value scan and
/// the JSON-key redaction pass.
static NAME_IS_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(secret|token|passwd|password|credential|api[_-]?key|access[_-]?key|private[_-]?key|auth[_-]?token|signing|connection[_-]?string|_pat$|_psk$|_dsn$|_key$|^key$)",
    )
    .expect("static secret-name regex compiles")
});

/// Redacts secret values/credentials from strings and JSON trees. Built once (env scan + regex
/// compile) and reused for every emitted telemetry envelope.
pub struct SecretRedactor {
    /// `(placeholder, value)` for secret-named env vars, sorted by value length **descending** so a
    /// secret that is a substring of another is masked first (no leftover tail).
    secret_values: Vec<(String, String)>,
    /// `(pattern, replacement)` for format-identifiable credentials. Replacements may use `${1}`.
    patterns: Vec<(Regex, &'static str)>,
}

impl SecretRedactor {
    /// Build from the current process environment.
    pub fn from_env() -> Self {
        Self::from_pairs(std::env::vars())
    }

    /// Build from an explicit set of `(name, value)` pairs (used by tests).
    pub fn from_pairs<I: IntoIterator<Item = (String, String)>>(vars: I) -> Self {
        let mut secret_values: Vec<(String, String)> = vars
            .into_iter()
            .filter(|(name, value)| {
                value.len() >= MIN_SECRET_VALUE_LEN && NAME_IS_SECRET.is_match(name)
            })
            .map(|(name, value)| (format!("[REDACTED:{name}]"), value))
            .collect();
        secret_values.sort_by_key(|(_, value)| std::cmp::Reverse(value.len()));

        Self {
            secret_values,
            patterns: Self::build_patterns(),
        }
    }

    fn build_patterns() -> Vec<(Regex, &'static str)> {
        // Static literals: a malformed pattern is a build-time bug, so `expect` fails loudly rather
        // than silently disabling a redaction class (which would leak a secret with no signal).
        let raw: &[(&str, &str)] = &[
            (r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED_OPENAI_KEY]"),
            (
                r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}",
                "[REDACTED_STRIPE_KEY]",
            ),
            // AKIA = permanent access keys, ASIA = temporary/session keys; both are 20 chars.
            (r"(?:AKIA|ASIA)[0-9A-Z]{16}", "[REDACTED_AWS_KEY]"),
            (r"hf_[A-Za-z0-9]{20,}", "[REDACTED_HF_TOKEN]"),
            (r"gh[pousr]_[A-Za-z0-9]{20,}", "[REDACTED_GITHUB_TOKEN]"),
            (r"glpat-[A-Za-z0-9_-]{16,}", "[REDACTED_GITLAB_TOKEN]"),
            (r"npm_[A-Za-z0-9]{20,}", "[REDACTED_NPM_TOKEN]"),
            (r"xox[baprs]-[A-Za-z0-9-]{10,}", "[REDACTED_SLACK_TOKEN]"),
            (r"AIza[0-9A-Za-z_\-]{30,}", "[REDACTED_GOOGLE_KEY]"),
            (
                r"eyJ[A-Za-z0-9_-]{6,}\.eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}",
                "[REDACTED_JWT]",
            ),
            (r"(?i)bearer[\s:=]+[A-Za-z0-9._\-]{8,}", "Bearer [REDACTED]"),
            (
                r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                "[REDACTED_PRIVATE_KEY]",
            ),
            // Generic `name = value` assignments; keep the key name, drop the value. Capture the
            // delimiter+opening-quote (group 2) and the closing quote (group 3) so `api_key: "x"`
            // stays well-formed as `api_key: "[REDACTED]"` rather than the mangled `api_key=[REDACTED]"`.
            (
                r#"(?i)(api[_-]?key|secret|token|passwd|password|access[_-]?key)(["']?\s*[:=]\s*["']?)[A-Za-z0-9._\-/+]{8,}(["']?)"#,
                "${1}${2}[REDACTED]${3}",
            ),
        ];
        raw.iter()
            .map(|(p, r)| {
                (
                    Regex::new(p).expect("static credential pattern compiles"),
                    *r,
                )
            })
            .collect()
    }

    /// Redact one string: env secret values first (exact substrings), then format patterns.
    /// Returns `Cow::Borrowed` unchanged when nothing matched (no allocation on the common path).
    pub fn redact<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let mut out: Cow<'a, str> = Cow::Borrowed(input);
        for (placeholder, value) in &self.secret_values {
            if out.contains(value.as_str()) {
                out = Cow::Owned(out.replace(value.as_str(), placeholder));
            }
        }
        for (re, repl) in &self.patterns {
            // `replace_all` already returns `Cow::Owned` only when it actually replaced something,
            // so match on that instead of a separate `is_match` pass (which would search twice).
            if let Cow::Owned(replaced) = re.replace_all(&out, *repl) {
                out = Cow::Owned(replaced);
            }
        }
        out
    }

    /// Recursively redact every string leaf in a JSON value, in place.
    pub fn redact_json(&self, value: &mut Value) {
        self.redact_json_at(value, 0);
    }

    fn redact_json_at(&self, value: &mut Value, depth: usize) {
        if depth > MAX_JSON_DEPTH {
            return;
        }
        match value {
            Value::String(s) => {
                if let Cow::Owned(redacted) = self.redact(s) {
                    *s = redacted;
                }
            }
            Value::Array(arr) => arr
                .iter_mut()
                .for_each(|v| self.redact_json_at(v, depth + 1)),
            Value::Object(map) => {
                for (k, v) in map.iter_mut() {
                    // A string value under a secret-named key (e.g. {"api_key": "..."}) is redacted
                    // by key even when the value matches no credential format and isn't in this
                    // process's env — structured telemetry is the common place such bare secrets
                    // appear. Over-redaction in a monitoring sink is harmless (see module docs).
                    if let Value::String(s) = v {
                        if NAME_IS_SECRET.is_match(k) {
                            *s = format!("[REDACTED:{}]", k.to_uppercase());
                            continue;
                        }
                    }
                    self.redact_json_at(v, depth + 1);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn redactor() -> SecretRedactor {
        SecretRedactor::from_pairs([
            (
                "OPENAI_API_KEY".to_string(),
                "sk-abcdef0123456789ABCDEF".to_string(),
            ),
            (
                "HF_TOKEN".to_string(),
                "hf_supersecretvalue0123456789".to_string(),
            ),
            ("HOME".to_string(), "/home/appuser".to_string()), // not a secret name
            ("DEBUG".to_string(), "1".to_string()),            // too short + not secret
            (
                "DB_PASSWORD".to_string(),
                "correct-horse-battery".to_string(),
            ),
            (
                "SENTRY_DSN".to_string(),
                "https://abc123def456@sentry.io/42".to_string(),
            ),
        ])
    }

    #[test]
    fn redacts_env_secret_values_by_name() {
        let r = redactor();
        assert_eq!(
            r.redact("key is sk-abcdef0123456789ABCDEF here"),
            "key is [REDACTED:OPENAI_API_KEY] here"
        );
        assert_eq!(
            r.redact("pw=correct-horse-battery"),
            "pw=[REDACTED:DB_PASSWORD]"
        );
        // `*_DSN` names are treated as secret (DSNs embed passwords).
        assert!(r
            .redact("conn https://abc123def456@sentry.io/42")
            .contains("[REDACTED:SENTRY_DSN]"));
    }

    #[test]
    fn does_not_redact_nonsecret_env_or_short_values() {
        let r = redactor();
        assert_eq!(r.redact("cwd is /home/appuser"), "cwd is /home/appuser");
        assert_eq!(r.redact("DEBUG=1"), "DEBUG=1");
    }

    #[test]
    fn clean_input_is_returned_borrowed_unchanged() {
        let r = redactor();
        let input = "plain prose with no secrets at all";
        assert!(matches!(r.redact(input), Cow::Borrowed(_)));
        assert_eq!(r.redact(input), input);
    }

    #[test]
    fn longer_overlapping_secret_is_masked_without_tail() {
        // The length-descending sort must mask the longer value first so no fragment survives.
        let r = SecretRedactor::from_pairs([
            ("A_TOKEN".to_string(), "abcdef1234".to_string()),
            ("B_TOKEN".to_string(), "abcdef1234EXTRA".to_string()),
        ]);
        let out = r.redact("val=abcdef1234EXTRA end");
        assert!(out.contains("[REDACTED:B_TOKEN]"), "got: {out}");
        assert!(
            !out.contains("abcdef1234"),
            "no fragment should survive: {out}"
        );
    }

    #[test]
    fn redacts_format_identifiable_credentials_not_in_env() {
        let r = SecretRedactor::from_pairs(std::iter::empty());
        assert_eq!(
            r.redact("token AKIAIOSFODNN7EXAMPLE done"),
            "token [REDACTED_AWS_KEY] done"
        );
        // Temporary/session keys (ASIA prefix) are redacted too, not just permanent AKIA keys.
        assert_eq!(
            r.redact("token ASIAIOSFODNN7EXAMPLE done"),
            "token [REDACTED_AWS_KEY] done"
        );
        assert!(r
            .redact("Authorization: Bearer ya29.A0ARrdaM-longtoken")
            .contains("Bearer [REDACTED]"));
        assert!(r
            .redact("ghp_0123456789abcdefghijABCDEFGHIJ")
            .contains("[REDACTED_GITHUB_TOKEN]"));
        assert!(r
            .redact("jwt eyJhbGciOi.eyJzdWIiNDA.SflKxwRJSM done")
            .contains("[REDACTED_JWT]"));
        assert!(r
            .redact("stripe sk_live_0123456789abcdefghij")
            .contains("[REDACTED_STRIPE_KEY]"));
        assert!(r
            .redact("plain prose with no secrets")
            .contains("plain prose"));
    }

    #[test]
    fn redacts_generic_assignment_and_full_pem_block() {
        let r = SecretRedactor::from_pairs(std::iter::empty());
        // Generic assignment keeps the key name, drops the value.
        let out = r.redact("password = hunter2longenough");
        assert!(out.contains("[REDACTED]"), "got: {out}");
        assert!(
            !out.contains("hunter2longenough"),
            "value must be gone: {out}"
        );
        // Quoted assignment keeps its delimiter and BOTH quotes intact (no mangling to
        // `api_key=[REDACTED]"`).
        let quoted = r.redact(r#"{"api_key": "abcdef123456"}"#);
        assert!(
            quoted.contains(r#""api_key": "[REDACTED]""#),
            "quotes/delimiter must be preserved: {quoted}"
        );
        assert!(
            !quoted.contains("abcdef123456"),
            "value must be gone: {quoted}"
        );
        // Full PEM block (body + footer) is masked, not just the header.
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK...\nabcd1234\n-----END RSA PRIVATE KEY-----";
        let red = r.redact(pem);
        assert_eq!(red, "[REDACTED_PRIVATE_KEY]");
        assert!(!red.contains("MIIBOgIBAAJBAK"));
    }

    #[test]
    fn redacts_string_leaves_in_nested_json() {
        let r = redactor();
        let mut v = json!({
            "telemetry": {
                "tool_name": "exec",
                "result": "OPENAI_API_KEY=sk-abcdef0123456789ABCDEF\nHF_TOKEN=hf_supersecretvalue0123456789",
                "exit": 0
            },
            "list": ["safe", "pw=correct-horse-battery"]
        });
        r.redact_json(&mut v);
        let result = v["telemetry"]["result"].as_str().unwrap();
        assert!(result.contains("[REDACTED:OPENAI_API_KEY]"));
        assert!(result.contains("[REDACTED:HF_TOKEN]"));
        assert!(!result.contains("sk-abcdef"));
        assert_eq!(v["telemetry"]["exit"], 0); // non-strings untouched
        assert_eq!(v["telemetry"]["tool_name"], "exec");
        assert!(v["list"][1]
            .as_str()
            .unwrap()
            .contains("[REDACTED:DB_PASSWORD]"));
    }

    #[test]
    fn redacts_json_string_values_under_secret_keys() {
        // A bare secret under a secret-named key — no recognizable credential format, not in the
        // env — is still redacted by key. Non-secret keys and non-string values are untouched.
        let r = SecretRedactor::from_pairs(std::iter::empty());
        let mut v = json!({
            "api_key": "plainvalue_no_format",
            "password": "short",
            "user": "alice",
            "count": 7,
            "nested": { "access_key": "anotherplainsecret" }
        });
        r.redact_json(&mut v);
        assert_eq!(v["api_key"], "[REDACTED:API_KEY]");
        assert_eq!(v["password"], "[REDACTED:PASSWORD]"); // redacted by key even though short
        assert_eq!(v["user"], "alice");
        assert_eq!(v["count"], 7);
        assert_eq!(v["nested"]["access_key"], "[REDACTED:ACCESS_KEY]");
    }
}
