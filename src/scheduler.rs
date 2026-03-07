use async_trait::async_trait;
use std::str::FromStr;
use log::{info, error};
use chrono::Utc;
use cron::Schedule;
use serde::{Deserialize, Serialize};
use rusqlite::{Connection, params};

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
        chat_id: String,
        channel: String,
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
    pub chat_id: String,
    pub channel: String,
}

/// An Actor that wakes up at scheduled intervals using a cron expression,
/// and outputs a specific trigger message to its downstream listeners.
pub struct CronActor {
    name: String,
    jobs: Vec<ActiveJob>,
    conn: Connection,
}

impl CronActor {
    pub fn new(name: &str, db_path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;

        // Create the cron_jobs table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cron_jobs (
                id TEXT PRIMARY KEY,
                schedule TEXT NOT NULL,
                message TEXT NOT NULL,
                last_run_at_ms INTEGER,
                chat_id TEXT NOT NULL DEFAULT 'unknown',
                channel TEXT NOT NULL DEFAULT 'unknown'
            )",
            [],
        )?;

        // Provide backwards compatibility
        let _ = conn.execute("ALTER TABLE cron_jobs ADD COLUMN chat_id TEXT DEFAULT 'unknown'", []);
        let _ = conn.execute("ALTER TABLE cron_jobs ADD COLUMN channel TEXT DEFAULT 'unknown'", []);

        let mut jobs = Vec::new();

        {
            let mut stmt = conn.prepare("SELECT id, schedule, message, last_run_at_ms, chat_id, channel FROM cron_jobs")?;
            
            let job_iter = stmt.query_map([], |row| {
                let schedule_str: String = row.get(1)?;
                let schedule: ScheduleKind = serde_json::from_str(&schedule_str)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                
                Ok(ActiveJob {
                    id: row.get(0)?,
                    schedule,
                    message: row.get(2)?,
                    last_run_at_ms: row.get(3)?,
                    chat_id: row.get(4)?,
                    channel: row.get(5)?,
                })
            })?;

            for job_result in job_iter {
                match job_result {
                    Ok(job) => jobs.push(job),
                    Err(e) => error!("Failed to load a cron job from DB: {}", e),
                }
            }
        }
        
        info!("Loaded {} cron jobs from database.", jobs.len());

        Ok(Self {
            name: name.to_string(),
            jobs,
            conn,
        })
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
                CronCommand::Add { id, schedule, message, chat_id, channel } => {
                    // Validate Cron if needed
                    if let ScheduleKind::Cron { ref cron_expr } = schedule {
                        if let Err(e) = Schedule::from_str(cron_expr) {
                            error!("[CronActor {}] Invalid cron expression '{}': {}", self.name, cron_expr, e);
                            return Err(ActorError::from(format!("Invalid cron expression: {}", e)));
                        }
                    }

                    info!("[CronActor {}] Added job '{}' with schedule {:?}", self.name, id, schedule);
                    
                    let schedule_json = match serde_json::to_string(&schedule) {
                        Ok(json) => json,
                        Err(e) => {
                            error!("[CronActor {}] Failed to serialize schedule: {}", self.name, e);
                            return Err(ActorError::from(format!("Failed to serialize schedule: {}", e)));
                        }
                    };

                    if let Err(e) = self.conn.execute(
                        "INSERT INTO cron_jobs (id, schedule, message, last_run_at_ms, chat_id, channel) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![id, schedule_json, message, None::<i64>, chat_id, channel],
                    ) {
                        error!("[CronActor {}] Failed to save cron job to DB: {}", self.name, e);
                    }

                    self.jobs.push(ActiveJob {
                        id,
                        schedule,
                        message,
                        last_run_at_ms: None,
                        chat_id,
                        channel,
                    });
                }
                CronCommand::Remove { id } => {
                    self.jobs.retain(|j| j.id != id);
                    if let Err(e) = self.conn.execute(
                        "DELETE FROM cron_jobs WHERE id = ?1",
                        params![id],
                    ) {
                        error!("[CronActor {}] Failed to remove cron job from DB: {}", self.name, e);
                    }
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
                    
                    let mut updated = false;
                    if job.last_run_at_ms.is_none() {
                        job.last_run_at_ms = Some(now_ms); // just set anchor
                        updated = true;
                    } else if (now_ms - last) >= *every_ms {
                        should_trigger = true;
                        job.last_run_at_ms = Some(now_ms);
                        updated = true;
                    }
                    
                    if updated {
                        if let Err(e) = self.conn.execute(
                            "UPDATE cron_jobs SET last_run_at_ms = ?1 WHERE id = ?2",
                            params![job.last_run_at_ms, job.id],
                        ) {
                            error!("[CronActor {}] Failed to update job last_run_at_ms in DB: {}", self.name, e);
                        }
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
                if let Ok(json_trigger) = serde_json::to_string(&serde_json::json!({
                    "chat_id": job.chat_id,
                    "channel": job.channel,
                    "message": job.message
                })) {
                    triggered_messages.push(json_trigger);
                }
            }
        }

        // Cleanup `At` jobs
        self.jobs.retain(|j| !jobs_to_remove.contains(&j.id));
        for id in jobs_to_remove {
            if let Err(e) = self.conn.execute(
                "DELETE FROM cron_jobs WHERE id = ?1",
                params![id],
            ) {
                error!("[CronActor {}] Failed to remove expired AT job from DB: {}", self.name, e);
            }
        }
        
        // Return only the first triggered message this tick, if multiple they'll queue or 
        // we can aggregate them. For now, just emit the first one.
        if !triggered_messages.is_empty() {
            return Ok(Some(("trigger".to_string(), triggered_messages[0].clone())));
        }
        
        Ok(None)
    }
}
