use async_trait::async_trait;
use std::str::FromStr;
use log::{info, error};
use chrono::Utc;
use cron::Schedule;
use serde::{Deserialize, Serialize};

use crate::{ActorLogic, ActorError};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "kind")]
pub enum ScheduleKind {
    At { at_ms: i64 },
    Every { every_ms: i64 },
    Cron { cron_expr: String },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum CronCommand {
    Add {
        id: String,
        schedule: ScheduleKind,
        message: String,
    },
    Remove {
        id: String,
    }
}

pub struct ActiveJob {
    pub id: String,
    pub schedule: ScheduleKind,
    pub message: String,
    pub last_run_at_ms: Option<i64>,
}

/// An Actor that wakes up at scheduled intervals using a cron expression,
/// and outputs a specific trigger message to its downstream listeners.
pub struct CronActor {
    name: String,
    jobs: Vec<ActiveJob>,
}

impl CronActor {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            jobs: Vec::new(),
        }
    }
}

// For CronActor, the incoming packet type could be anything (like an empty trigger or control signals).
// We'll use `String` to match the Agent network. It ignores incoming packets mostly,
// and relies on `tick_interval` and `on_tick` to do the actual cron work.
#[async_trait]
impl ActorLogic<String> for CronActor {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn process(&mut self, packet: String) -> Result<Option<(String, String)>, ActorError> {
        if let Ok(cmd) = serde_json::from_str::<CronCommand>(&packet) {
            match cmd {
                CronCommand::Add { id, schedule, message } => {
                    // Validate Cron if needed
                    if let ScheduleKind::Cron { ref cron_expr } = schedule {
                        if let Err(e) = Schedule::from_str(cron_expr) {
                            error!("[CronActor {}] Invalid cron expression '{}': {}", self.name, cron_expr, e);
                            return Err(ActorError::from(format!("Invalid cron expression: {}", e)));
                        }
                    }

                    info!("[CronActor {}] Added job '{}' with schedule {:?}", self.name, id, schedule);
                    self.jobs.push(ActiveJob {
                        id,
                        schedule,
                        message,
                        last_run_at_ms: None,
                    });
                }
                CronCommand::Remove { id } => {
                    self.jobs.retain(|j| j.id != id);
                    info!("[CronActor {}] Removed job '{}'", self.name, id);
                }
            }
        }
        Ok(None)
    }

    fn tick_interval(&self) -> Option<tokio::time::Duration> {
        Some(tokio::time::Duration::from_secs(1))
    }

    async fn on_tick(&mut self) -> Result<Option<(String, String)>, ActorError> {
        let now = Utc::now();
        let now_ms = now.timestamp_millis();
        
        let mut triggered_messages = Vec::new();
        let mut jobs_to_remove = Vec::new();

        for job in &mut self.jobs {
            let mut should_trigger = false;

            match &job.schedule {
                ScheduleKind::At { at_ms } => {
                    if now_ms >= *at_ms {
                        should_trigger = true;
                        jobs_to_remove.push(job.id.clone());
                    }
                }
                ScheduleKind::Every { every_ms } => {
                    let last = job.last_run_at_ms.unwrap_or(now_ms); // if never run, anchor to now
                    
                    if job.last_run_at_ms.is_none() {
                        job.last_run_at_ms = Some(now_ms); // just set anchor
                    } else if (now_ms - last) >= *every_ms {
                        should_trigger = true;
                        job.last_run_at_ms = Some(now_ms);
                    }
                }
                ScheduleKind::Cron { cron_expr } => {
                    // Re-parse the cron. (In production, cache this)
                    if let Ok(sched) = Schedule::from_str(cron_expr) {
                        let a_second_ago = now - chrono::Duration::seconds(1);
                        if let Some(next) = sched.after(&a_second_ago).next() {
                            if next <= now {
                                should_trigger = true;
                            }
                        }
                    }
                }
            }

            if should_trigger {
                info!("[CronActor {}] Triggering scheduled event: {}", self.name, job.id);
                triggered_messages.push(job.message.clone());
            }
        }

        // Cleanup `At` jobs
        self.jobs.retain(|j| !jobs_to_remove.contains(&j.id));
        
        // Return only the first triggered message this tick, if multiple they'll queue or 
        // we can aggregate them. For now, just emit the first one.
        if !triggered_messages.is_empty() {
            return Ok(Some(("trigger".to_string(), triggered_messages[0].clone())));
        }
        
        Ok(None)
    }
}
