//! Secret-safe subprocess environment policy and sanitization engine.
//!
//! Prevents agent-controlled child processes (exec, python_run, execution sessions)
//! from inheriting master host credentials (e.g. `OPENAI_API_KEY`, `GITHUB_TOKEN`, `ALTAI_*`)
//! by default while allowing declared, explicit credential grants.

use std::collections::{HashMap, HashSet};

/// Default system-safe environment variable names permitted across platforms.
pub const DEFAULT_SAFE_INHERITED_KEYS: &[&str] = &[
    // Common / POSIX
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "PWD",
    // Windows
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)",
    "ALLUSERSPROFILE",
    "PUBLIC",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "SYSTEMDRIVE",
];

/// Default case-insensitive substrings / suffixes indicating sensitive credentials.
pub const DEFAULT_SECRET_PATTERNS: &[&str] = &[
    "_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_AUTH",
    "_CREDENTIAL",
    "OPENAI_",
    "ANTHROPIC_",
    "GEMINI_",
    "DEEPSEEK_",
    "GROQ_",
    "MISTRAL_",
    "COHERE_",
    "HUGGINGFACE_",
    "HF_TOKEN",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "SLACK_",
    "ALTAI_",
    "ISANAGENT_",
    "SSH_PASSWORD",
];

/// Execution environment policy configuring subprocess variable inheritance and sanitization.
#[derive(Clone, Debug)]
pub struct ExecutionEnvironmentPolicy {
    /// Variables allowed to be inherited from the host process if present.
    pub inherited_allowlist: HashSet<String>,
    /// Explicit environment variables to inject.
    pub explicit_env: HashMap<String, String>,
    /// Sensitive patterns that are blocked from inheritance unless explicitly granted.
    pub secret_patterns: Vec<String>,
    /// Whether to strip all non-allowlisted variables (default: true).
    pub scrub_secrets: bool,
}

impl Default for ExecutionEnvironmentPolicy {
    fn default() -> Self {
        Self::default_safe()
    }
}

impl ExecutionEnvironmentPolicy {
    /// Creates a safe default policy: inherits standard OS vars, scrubs credentials.
    pub fn default_safe() -> Self {
        let mut allowlist = HashSet::new();
        for key in DEFAULT_SAFE_INHERITED_KEYS {
            allowlist.insert(key.to_string());
            // On Windows env vars are case-insensitive, store uppercase for normalized matching
            #[cfg(windows)]
            allowlist.insert(key.to_uppercase());
        }

        Self {
            inherited_allowlist: allowlist,
            explicit_env: HashMap::new(),
            secret_patterns: DEFAULT_SECRET_PATTERNS
                .iter()
                .map(|&s| s.to_string())
                .collect(),
            scrub_secrets: true,
        }
    }

    /// Extends the allowlist with custom permitted keys.
    pub fn with_allowlist<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for k in keys {
            self.inherited_allowlist.insert(k.as_ref().to_string());
            #[cfg(windows)]
            self.inherited_allowlist.insert(k.as_ref().to_uppercase());
        }
        self
    }

    /// Sets explicit environment variables to be set in the child process.
    pub fn with_explicit_env(mut self, env: HashMap<String, String>) -> Self {
        self.explicit_env = env;
        self
    }

    /// Checks if an environment variable key looks like a sensitive credential.
    pub fn is_secret_var(&self, key: &str) -> bool {
        let upper = key.to_uppercase();
        for pattern in &self.secret_patterns {
            let pat_upper = pattern.to_uppercase();
            if pat_upper.starts_with('_') && upper.ends_with(&pat_upper) {
                return true;
            }
            if pat_upper.ends_with('_') && upper.starts_with(&pat_upper) {
                return true;
            }
            if upper.contains(&pat_upper) {
                return true;
            }
        }
        false
    }

    /// Computes the clean, sanitized map of environment variables.
    pub fn build_clean_env(&self) -> HashMap<String, String> {
        let mut clean = HashMap::new();

        if self.scrub_secrets {
            // Only pull allowed keys from the host environment
            for (k, v) in std::env::vars() {
                let key_norm = if cfg!(windows) {
                    k.to_uppercase()
                } else {
                    k.clone()
                };

                let is_allowed = self.inherited_allowlist.contains(&k)
                    || self.inherited_allowlist.contains(&key_norm);

                if is_allowed && !self.is_secret_var(&k) {
                    clean.insert(k, v);
                }
            }
        } else {
            // Inherit full environment except direct secret patterns
            for (k, v) in std::env::vars() {
                if !self.is_secret_var(&k) {
                    clean.insert(k, v);
                }
            }
        }

        // Explicit variables always take precedence (e.g. intentional credential grants)
        for (k, v) in &self.explicit_env {
            clean.insert(k.clone(), v.clone());
        }

        clean
    }

    /// Applies the sanitized environment to an asynchronous tokio Command.
    pub fn apply_to_tokio_command(&self, cmd: &mut tokio::process::Command) {
        cmd.env_clear();
        for (k, v) in self.build_clean_env() {
            cmd.env(k, v);
        }
    }

    /// Applies the sanitized environment to a synchronous std Command.
    pub fn apply_to_std_command(&self, cmd: &mut std::process::Command) {
        cmd.env_clear();
        for (k, v) in self.build_clean_env() {
            cmd.env(k, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_secret_variables() {
        let policy = ExecutionEnvironmentPolicy::default_safe();
        assert!(policy.is_secret_var("OPENAI_API_KEY"));
        assert!(policy.is_secret_var("ANTHROPIC_API_KEY"));
        assert!(policy.is_secret_var("GITHUB_TOKEN"));
        assert!(policy.is_secret_var("MY_SERVICE_PASSWORD"));
        assert!(policy.is_secret_var("ALTAI_APP_DISPATCH_TOKEN"));
        assert!(!policy.is_secret_var("PATH"));
        assert!(!policy.is_secret_var("USER"));
        assert!(!policy.is_secret_var("PYTHONPATH"));
    }

    #[test]
    fn builds_sanitized_environment() {
        let policy = ExecutionEnvironmentPolicy::default_safe();
        let clean = policy.build_clean_env();

        // Host API keys must never appear in clean environment
        assert!(!clean.contains_key("OPENAI_API_KEY"));
        assert!(!clean.contains_key("ANTHROPIC_API_KEY"));
        assert!(!clean.contains_key("ALTAI_APP_DISPATCH_TOKEN"));

        // Essential PATH should be preserved if set in host
        if std::env::var("PATH").is_ok() || std::env::var("Path").is_ok() {
            let has_path = clean.keys().any(|k| k.eq_ignore_ascii_case("PATH"));
            assert!(has_path, "PATH must be preserved in clean environment");
        }
    }

    #[test]
    fn explicit_grants_override_sanitization() {
        let mut grants = HashMap::new();
        grants.insert(
            "CUSTOM_API_KEY".to_string(),
            "granted-secret-123".to_string(),
        );

        let policy = ExecutionEnvironmentPolicy::default_safe().with_explicit_env(grants);
        let clean = policy.build_clean_env();

        assert_eq!(
            clean.get("CUSTOM_API_KEY"),
            Some(&"granted-secret-123".to_string())
        );
    }
}
