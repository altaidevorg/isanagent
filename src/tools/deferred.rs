//! Deferred tool schemas.
//!
//! A deferred tool is registered (callable, searchable) but its JSON schema is left out of the
//! provider `tools` array until a session activates it — through `search_tools` or by loading a
//! skill whose `allowed-tools` names it. Large MCP servers (a chemistry server with ~100 tools is
//! ~27k tokens of schema) then cost a short name list instead of their full schemas on every turn.
//!
//! Activation is per session key (see [`crate::bus::clarification_session_key`]), so one chat
//! loading a server's tools does not grow every other chat's requests.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::tools::mcp::tool_pattern_matches;

#[derive(Default)]
struct DeferredInner {
    /// Deferred tool name → group label (MCP server name).
    deferred: BTreeMap<String, String>,
    /// Session key → deferred tools activated in that session.
    active: HashMap<String, HashSet<String>>,
}

/// Shared handle; clones see the same state.
#[derive(Clone, Default)]
pub struct DeferredTools {
    inner: Arc<RwLock<DeferredInner>>,
}

impl DeferredTools {
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks `name` deferred under `group` (usually the MCP server name).
    pub fn defer(&self, name: impl Into<String>, group: impl Into<String>) {
        if let Ok(mut inner) = self.inner.write() {
            inner.deferred.insert(name.into(), group.into());
        }
    }

    pub fn is_deferred(&self, name: &str) -> bool {
        self.inner
            .read()
            .map(|inner| inner.deferred.contains_key(name))
            .unwrap_or(false)
    }

    pub fn is_empty(&self) -> bool {
        self.inner
            .read()
            .map(|inner| inner.deferred.is_empty())
            .unwrap_or(true)
    }

    /// True when `name` is not deferred, or is deferred and active in `session_key`.
    pub fn is_visible(&self, session_key: Option<&str>, name: &str) -> bool {
        let Ok(inner) = self.inner.read() else {
            return true;
        };
        if !inner.deferred.contains_key(name) {
            return true;
        }
        session_key
            .and_then(|key| inner.active.get(key))
            .is_some_and(|set| set.contains(name))
    }

    /// Activates the deferred tools among `names` for `session_key`; returns the ones that were
    /// not active before (non-deferred names are ignored).
    pub fn activate<I, S>(&self, session_key: &str, names: I) -> Vec<String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let Ok(mut inner) = self.inner.write() else {
            return Vec::new();
        };
        let wanted: Vec<String> = names
            .into_iter()
            .map(|n| n.as_ref().to_string())
            .filter(|n| inner.deferred.contains_key(n))
            .collect();
        let set = inner.active.entry(session_key.to_string()).or_default();
        wanted
            .into_iter()
            .filter(|n| set.insert(n.clone()))
            .collect()
    }

    /// Activates every deferred tool matching one of `patterns` (exact name, or a prefix with
    /// a trailing `*`, e.g. `chem__rdkit_*`); returns the newly active ones.
    pub fn activate_matching(&self, session_key: &str, patterns: &[String]) -> Vec<String> {
        let matching: Vec<String> = match self.inner.read() {
            Ok(inner) => inner
                .deferred
                .keys()
                .filter(|name| patterns.iter().any(|p| tool_pattern_matches(p, name)))
                .cloned()
                .collect(),
            Err(_) => return Vec::new(),
        };
        self.activate(session_key, matching)
    }

    /// Stops deferring every tool matching `patterns` (always-on skills name tools that should
    /// always be in the request). Returns the tools affected.
    pub fn undefer_matching(&self, patterns: &[String]) -> Vec<String> {
        let Ok(mut inner) = self.inner.write() else {
            return Vec::new();
        };
        let names: Vec<String> = inner
            .deferred
            .keys()
            .filter(|name| patterns.iter().any(|p| tool_pattern_matches(p, name)))
            .cloned()
            .collect();
        for name in &names {
            inner.deferred.remove(name);
        }
        names
    }

    /// Deferred tool names grouped by label, both sorted.
    pub fn groups(&self) -> BTreeMap<String, Vec<String>> {
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Ok(inner) = self.inner.read() {
            for (name, group) in &inner.deferred {
                out.entry(group.clone()).or_default().push(name.clone());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::DeferredTools;

    #[test]
    fn activation_is_per_session_and_only_for_deferred_tools() {
        let d = DeferredTools::new();
        d.defer("chem__mol_weight", "chem");
        d.defer("chem__logp", "chem");
        assert!(d.is_visible(None, "read_file"));
        assert!(!d.is_visible(Some("a"), "chem__logp"));

        let newly = d.activate("a", ["chem__logp", "read_file"]);
        assert_eq!(newly, vec!["chem__logp".to_string()]);
        assert!(d.is_visible(Some("a"), "chem__logp"));
        assert!(!d.is_visible(Some("b"), "chem__logp"));
        assert!(d.activate("a", ["chem__logp"]).is_empty());
    }

    #[test]
    fn activate_matching_supports_prefix_patterns() {
        let d = DeferredTools::new();
        d.defer("chem__rdkit_a", "chem");
        d.defer("chem__rdkit_b", "chem");
        d.defer("chem__stats_c", "chem");
        let mut got = d.activate_matching("s", &["chem__rdkit_*".to_string()]);
        got.sort();
        assert_eq!(got, vec!["chem__rdkit_a", "chem__rdkit_b"]);
        assert_eq!(d.groups()["chem"].len(), 3);
    }
}
