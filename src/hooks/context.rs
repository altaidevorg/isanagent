use std::path::Path;
use std::sync::Arc;

use crate::config::HarnessConfig;

use super::observation::{observation_params_from_config, start_observation_hooks};
use super::steering::build_steering_engine;

/// Shared observation + steering runtimes for one process (optional each).
#[derive(Clone)]
pub struct ToolCallHookContext {
    pub observation: Option<Arc<super::ObservationHooksHandle>>,
    pub steering: Option<Arc<super::SteeringHooksEngine>>,
}

impl ToolCallHookContext {
    /// Returns `Some` when at least one hook subsystem is active.
    pub fn from_harness_config(
        workspace_dir: &Path,
        sandbox_dir: &Path,
        harness: &HarnessConfig,
    ) -> Option<Arc<Self>> {
        let hooks = harness.hooks.as_ref()?;

        let obs_cfg = hooks.observation.as_ref();
        let obs_on = obs_cfg.is_some_and(|o| o.enabled.unwrap_or(false));
        let observation = if obs_on {
            let o = obs_cfg?;
            let params = observation_params_from_config(
                workspace_dir,
                o.jsonl_path.as_deref(),
                o.webhook_url.as_deref(),
                o.webhook_hmac_secret.as_deref(),
                o.metadata_keys.clone().unwrap_or_default(),
                o.queue_capacity,
            );
            start_observation_hooks(params)
        } else {
            None
        };

        let steer_cfg = hooks.steering.as_ref();
        let steer_on = steer_cfg.is_some_and(|s| s.enabled.unwrap_or(false));
        let steering = if steer_on {
            let s = steer_cfg?;
            match build_steering_engine(s, workspace_dir.to_path_buf(), sandbox_dir.to_path_buf()) {
                Ok(e) => Some(e),
                Err(e) => {
                    log::error!("hooks steering init failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        if observation.is_none() && steering.is_none() {
            return None;
        }

        Some(Arc::new(Self {
            observation: observation.map(Arc::new),
            steering,
        }))
    }
}
