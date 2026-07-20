use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Datelike, Timelike, Utc};
use cron::Schedule;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::logging::LoggerHandle;
use crate::multi_tenant_edge::{CronRegistrationClient, CronRule};
use crate::{ActorError, ActorLogic};

const WEBHOOK_PATH_PREFIX: &str = "/_mte/cron";
const ONE_SHOT_CLAIM_TTL_MS: i64 = 60_000;
const ONE_SHOT_RETRY_DELAY_MS: i64 = 30_000;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind")]
pub enum ScheduleKind {
    At { at_ms: i64 },
    Every { every_ms: i64 },
    Cron { cron_expr: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
    },
    /// Reload durable jobs after a trusted embedding host has updated the
    /// store directly and needs this local scheduler to observe the change.
    Reload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveJob {
    pub id: String,
    pub schedule: ScheduleKind,
    pub message: String,
    pub last_run_at_ms: Option<i64>,
    pub chat_id: String,
    pub channel: String,
    pub webhook_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CronTriggerPayload {
    pub job_id: String,
    pub message: String,
    pub chat_id: String,
    pub channel: String,
}

#[derive(Debug)]
pub enum CronWebhookError {
    NotFound,
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CronSchedulingMode {
    Local,
    MultiTenantEdge,
}

#[derive(Clone)]
pub struct CronStore {
    conn: Arc<Mutex<Connection>>,
}

#[derive(Clone)]
pub struct MultiTenantEdgeCronScheduler {
    store: CronStore,
    sync_client: CronRegistrationClient,
    claim_ttl_ms: i64,
}

pub struct PendingCronTrigger {
    payload: CronTriggerPayload,
    completion: PendingCronTriggerCompletion,
}

pub enum PendingCronTriggerFinalize {
    Completed,
    CompletedWithWarning(String),
}

enum PendingCronTriggerCompletion {
    Immediate,
    OneShot {
        scheduler: MultiTenantEdgeCronScheduler,
        job_id: String,
        claim_token: String,
        original_at_ms: i64,
    },
}

struct ClaimedTrigger {
    job: ActiveJob,
    claim_token: Option<String>,
}

/// An Actor that wakes up at scheduled intervals using a cron expression,
/// and outputs a specific trigger message to its downstream listeners.
pub struct CronActor {
    name: String,
    jobs: Vec<ActiveJob>,
    cron_schedule_cache: HashMap<String, Schedule>,
    store: CronStore,
    logger_tx: LoggerHandle,
    scheduling_mode: CronSchedulingMode,
    bus_tx: tokio::sync::mpsc::Sender<crate::bus::BusMessage>,
}

impl CronStore {
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
        crate::memory::ensure_cron_jobs_schema(&conn).map_err(|e| e.to_string())?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn load_jobs(&self) -> Result<Vec<ActiveJob>, String> {
        let conn = self.lock_conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, schedule, message, last_run_at_ms, chat_id, channel, webhook_token
                 FROM cron_jobs
                 WHERE completed_at_ms IS NULL",
            )
            .map_err(|e| e.to_string())?;

        let rows = stmt
            .query_map([], |row| {
                let schedule_json: String = row.get(1)?;
                let schedule = decode_schedule(&schedule_json)?;
                Ok(ActiveJob {
                    id: row.get(0)?,
                    schedule,
                    message: row.get(2)?,
                    last_run_at_ms: row.get(3)?,
                    chat_id: row.get(4)?,
                    channel: row.get(5)?,
                    webhook_token: row.get(6)?,
                })
            })
            .map_err(|e| e.to_string())?;

        let mut jobs = Vec::new();
        for row in rows {
            let job = row.map_err(|e| e.to_string())?;
            if job.webhook_token.is_empty() {
                return Err(format!(
                    "Cron job {} is missing webhook_token; recreate the generated workspace to rebuild cron state",
                    job.id
                ));
            }
            jobs.push(job);
        }

        Ok(jobs)
    }

    pub fn insert_job(&self, job: &ActiveJob) -> Result<(), String> {
        let schedule_json = serde_json::to_string(&job.schedule)
            .map_err(|e| format!("Failed to serialize schedule: {}", e))?;
        let conn = self.lock_conn()?;
        conn.execute(
            "INSERT INTO cron_jobs (
                id, schedule, message, last_run_at_ms, chat_id, channel, webhook_token,
                trigger_claim_token, trigger_claimed_at_ms, completed_at_ms
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, '', NULL, NULL)",
            params![
                job.id,
                schedule_json,
                job.message,
                job.last_run_at_ms,
                job.chat_id,
                job.channel,
                job.webhook_token
            ],
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }

    pub fn remove_job(&self, id: &str) -> Result<(), String> {
        let conn = self.lock_conn()?;
        conn.execute("DELETE FROM cron_jobs WHERE id = ?1", params![id])
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    pub fn find_job(&self, id: &str) -> Result<Option<ActiveJob>, String> {
        let conn = self.lock_conn()?;
        conn.query_row(
            "SELECT id, schedule, message, last_run_at_ms, chat_id, channel, webhook_token
             FROM cron_jobs WHERE id = ?1 AND completed_at_ms IS NULL",
            params![id],
            |row| {
                let schedule_json: String = row.get(1)?;
                let schedule = decode_schedule(&schedule_json)?;
                Ok(ActiveJob {
                    id: row.get(0)?,
                    schedule,
                    message: row.get(2)?,
                    last_run_at_ms: row.get(3)?,
                    chat_id: row.get(4)?,
                    channel: row.get(5)?,
                    webhook_token: row.get(6)?,
                })
            },
        )
        .optional()
        .map_err(|e| e.to_string())
    }

    pub fn update_last_run_at_ms(
        &self,
        id: &str,
        last_run_at_ms: Option<i64>,
    ) -> Result<(), String> {
        let conn = self.lock_conn()?;
        conn.execute(
            "UPDATE cron_jobs SET last_run_at_ms = ?1 WHERE id = ?2",
            params![last_run_at_ms, id],
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }

    fn begin_pending_trigger(
        &self,
        job_id: &str,
        token: &str,
        now_ms: i64,
        claim_ttl_ms: i64,
    ) -> Result<Option<ClaimedTrigger>, String> {
        let mut conn = self.lock_conn()?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;

        let row = tx
            .query_row(
                "SELECT id, schedule, message, last_run_at_ms, chat_id, channel, webhook_token,
                        trigger_claim_token, trigger_claimed_at_ms
                 FROM cron_jobs
                 WHERE id = ?1 AND webhook_token = ?2 AND completed_at_ms IS NULL",
                params![job_id, token],
                |row| {
                    let schedule_json: String = row.get(1)?;
                    let schedule = decode_schedule(&schedule_json)?;
                    Ok((
                        ActiveJob {
                            id: row.get(0)?,
                            schedule,
                            message: row.get(2)?,
                            last_run_at_ms: row.get(3)?,
                            chat_id: row.get(4)?,
                            channel: row.get(5)?,
                            webhook_token: row.get(6)?,
                        },
                        row.get::<_, String>(7)?,
                        row.get::<_, Option<i64>>(8)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| e.to_string())?;

        let Some((job, existing_claim_token, existing_claimed_at_ms)) = row else {
            return Ok(None);
        };

        if !matches!(job.schedule, ScheduleKind::At { .. }) {
            tx.commit().map_err(|e| e.to_string())?;
            return Ok(Some(ClaimedTrigger {
                job,
                claim_token: None,
            }));
        }

        let stale_cutoff_ms = now_ms - claim_ttl_ms;
        let new_claim_token = generate_webhook_token();
        let updated_rows = tx
            .execute(
                "UPDATE cron_jobs
                 SET trigger_claim_token = ?3, trigger_claimed_at_ms = ?4
                 WHERE id = ?1
                   AND webhook_token = ?2
                   AND (
                        trigger_claim_token = ''
                        OR trigger_claimed_at_ms IS NULL
                        OR trigger_claimed_at_ms <= ?5
                   )",
                params![job_id, token, new_claim_token, now_ms, stale_cutoff_ms],
            )
            .map_err(|e| e.to_string())?;

        if updated_rows == 0
            && !existing_claim_token.is_empty()
            && existing_claimed_at_ms.unwrap_or(i64::MAX) > stale_cutoff_ms
        {
            return Ok(None);
        }

        if updated_rows == 0 {
            return Ok(None);
        }

        tx.commit().map_err(|e| e.to_string())?;
        Ok(Some(ClaimedTrigger {
            job,
            claim_token: Some(new_claim_token),
        }))
    }

    fn complete_pending_trigger(&self, job_id: &str, claim_token: &str) -> Result<(), String> {
        let conn = self.lock_conn()?;
        let deleted = conn
            .execute(
                "DELETE FROM cron_jobs WHERE id = ?1 AND trigger_claim_token = ?2",
                params![job_id, claim_token],
            )
            .map_err(|e| e.to_string())?;
        if deleted == 0 {
            return Err(format!(
                "Failed to finalize multi-tenant-edge one-shot cron job {} because its claim was no longer valid",
                job_id
            ));
        }
        Ok(())
    }

    fn mark_pending_trigger_delivered(
        &self,
        job_id: &str,
        claim_token: &str,
        delivered_at_ms: i64,
    ) -> Result<(), String> {
        let conn = self.lock_conn()?;
        let updated = conn
            .execute(
                "UPDATE cron_jobs
                 SET completed_at_ms = ?3
                 WHERE id = ?1
                   AND trigger_claim_token = ?2
                   AND completed_at_ms IS NULL",
                params![job_id, claim_token, delivered_at_ms],
            )
            .map_err(|e| e.to_string())?;
        if updated == 0 {
            return Err(format!(
                "Failed to mark multi-tenant-edge one-shot cron job {} as delivered because its claim was no longer valid",
                job_id
            ));
        }
        Ok(())
    }

    fn reschedule_one_shot_after_claim(
        &self,
        job_id: &str,
        claim_token: &str,
        retry_at_ms: i64,
    ) -> Result<(), String> {
        let schedule_json = serde_json::to_string(&ScheduleKind::At { at_ms: retry_at_ms })
            .map_err(|e| format!("Failed to serialize one-shot retry schedule: {}", e))?;
        let conn = self.lock_conn()?;
        let updated = conn
            .execute(
                "UPDATE cron_jobs
                 SET schedule = ?3, trigger_claim_token = '', trigger_claimed_at_ms = NULL
                 WHERE id = ?1 AND trigger_claim_token = ?2",
                params![job_id, claim_token, schedule_json],
            )
            .map_err(|e| e.to_string())?;
        if updated == 0 {
            return Err(format!(
                "Failed to reschedule multi-tenant-edge one-shot cron job {} because its claim was no longer valid",
                job_id
            ));
        }
        Ok(())
    }

    fn lock_conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>, String> {
        self.conn
            .lock()
            .map_err(|_| "Failed to lock cron SQLite connection".to_string())
    }

    fn record_cron_background_job(
        &self,
        job_id: &str,
        channel: &str,
        chat_id: &str,
        message: &str,
        now_ms: i64,
    ) -> Result<bool, String> {
        let conn = self.lock_conn()?;

        let full_job_id = format!("cron:{}", job_id);
        let existing_state: Option<String> = conn
            .query_row(
                "SELECT state FROM background_jobs WHERE job_id = ?1",
                params![full_job_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;

        if let Some(state) = existing_state {
            if state == "running" || state == "waiting" {
                return Ok(false); // Already active, skip.
            }
        }

        let payload = serde_json::json!({
            "trigger": "cron",
            "message": message,
            "cron_job_id": job_id,
        })
        .to_string();
        conn.execute(
            "INSERT INTO background_jobs (
                job_id, kind, chat_id, channel, thread_id, state, payload_json,
                resume_after_restart, detached, last_error, created_at_ms, updated_at_ms
            ) VALUES (?1, 'cron', ?2, ?3, NULL, 'running', ?4, 1, 1, NULL, ?5, ?5)
            ON CONFLICT(job_id) DO UPDATE SET
                state = 'running', payload_json = excluded.payload_json, updated_at_ms = excluded.updated_at_ms",
            params![full_job_id, chat_id, channel, payload, now_ms],
        ).map_err(|e| format!("insert background_jobs from cron: {}", e))?;
        conn.execute(
            "INSERT INTO notifications (
                notification_id, chat_id, channel, thread_id, kind, title, body, action_kind, action_payload,
                seen_at_ms, resolved_at_ms, created_at_ms
            ) VALUES (?1, ?2, ?3, NULL, 'cron_triggered', ?4, ?5, 'open_job', ?6, NULL, NULL, ?7)",
            params![
                format!("notif:cron:{}:{}", job_id, now_ms),
                chat_id,
                channel,
                "Cron Triggered",
                format!("Background task: {}", message),
                serde_json::json!({"job_id": full_job_id}).to_string(),
                now_ms
            ],
        ).map_err(|e| format!("insert notifications from cron: {}", e))?;
        Ok(true)
    }
}

impl MultiTenantEdgeCronScheduler {
    pub fn new(db_path: &str, sync_client: CronRegistrationClient) -> Result<Self, String> {
        Ok(Self {
            store: CronStore::new(db_path)?,
            sync_client,
            claim_ttl_ms: ONE_SHOT_CLAIM_TTL_MS,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_with_store(store: CronStore, sync_client: CronRegistrationClient) -> Self {
        Self {
            store,
            sync_client,
            claim_ttl_ms: ONE_SHOT_CLAIM_TTL_MS,
        }
    }

    pub async fn sync_all(&self, now: DateTime<Utc>) -> Result<(), String> {
        sync_multi_tenant_edge_cron_jobs(&self.store, &self.sync_client, now).await
    }

    pub async fn add_job(&self, job: ActiveJob, now: DateTime<Utc>) -> Result<(), String> {
        validate_multi_tenant_edge_schedule(&job.schedule, now)?;
        self.store.insert_job(&job)?;
        if let Err(error) = self.sync_all(now).await {
            let _ = self.store.remove_job(&job.id);
            return Err(error);
        }
        Ok(())
    }

    /// Load one active job so embedding hosts can authorize an operation
    /// against its persisted destination before removing it.
    pub fn find_job(&self, id: &str) -> Result<Option<ActiveJob>, String> {
        self.store.find_job(id)
    }

    pub async fn remove_job(&self, id: &str, now: DateTime<Utc>) -> Result<bool, String> {
        let existing_job = self.store.find_job(id)?;
        let Some(existing_job) = existing_job else {
            return Ok(false);
        };

        self.store.remove_job(id)?;
        if let Err(error) = self.sync_all(now).await {
            let _ = self.store.insert_job(&existing_job);
            return Err(error);
        }
        Ok(true)
    }

    pub async fn begin_trigger(
        &self,
        job_id: &str,
        token: &str,
        now: DateTime<Utc>,
    ) -> Result<PendingCronTrigger, CronWebhookError> {
        let claimed = self
            .store
            .begin_pending_trigger(job_id, token, now.timestamp_millis(), self.claim_ttl_ms)
            .map_err(CronWebhookError::Internal)?;
        let Some(claimed) = claimed else {
            return Err(CronWebhookError::NotFound);
        };

        let payload = CronTriggerPayload {
            job_id: claimed.job.id.clone(),
            message: claimed.job.message.clone(),
            chat_id: claimed.job.chat_id.clone(),
            channel: claimed.job.channel.clone(),
        };
        let original_at_ms = match claimed.job.schedule {
            ScheduleKind::At { at_ms } => Some(at_ms),
            _ => None,
        };
        let completion = match claimed.claim_token {
            Some(claim_token) => PendingCronTriggerCompletion::OneShot {
                scheduler: self.clone(),
                job_id: claimed.job.id,
                claim_token,
                original_at_ms: original_at_ms
                    .expect("one-shot claims must come from At schedules"),
            },
            None => PendingCronTriggerCompletion::Immediate,
        };

        Ok(PendingCronTrigger {
            payload,
            completion,
        })
    }
}

impl PendingCronTrigger {
    pub fn payload(&self) -> &CronTriggerPayload {
        &self.payload
    }

    pub fn mark_delivered(&self, delivered_at_ms: i64) -> Result<(), String> {
        match &self.completion {
            PendingCronTriggerCompletion::Immediate => Ok(()),
            PendingCronTriggerCompletion::OneShot {
                scheduler,
                job_id,
                claim_token,
                ..
            } => {
                scheduler
                    .store
                    .mark_pending_trigger_delivered(job_id, claim_token, delivered_at_ms)
            }
        }
    }

    pub async fn rollback(self) -> Result<(), String> {
        match self.completion {
            PendingCronTriggerCompletion::Immediate => Ok(()),
            PendingCronTriggerCompletion::OneShot {
                scheduler,
                job_id,
                claim_token,
                original_at_ms,
            } => {
                let now = Utc::now();
                let retry_at_ms = std::cmp::max(
                    now.timestamp_millis() + ONE_SHOT_RETRY_DELAY_MS,
                    original_at_ms,
                );
                scheduler.store.reschedule_one_shot_after_claim(
                    &job_id,
                    &claim_token,
                    retry_at_ms,
                )?;
                scheduler.sync_all(now).await
            }
        }
    }

    pub async fn complete(self) -> Result<PendingCronTriggerFinalize, String> {
        match self.completion {
            PendingCronTriggerCompletion::Immediate => Ok(PendingCronTriggerFinalize::Completed),
            PendingCronTriggerCompletion::OneShot {
                scheduler,
                job_id,
                claim_token,
                original_at_ms: _,
            } => {
                let cleanup_warning = scheduler
                    .store
                    .complete_pending_trigger(&job_id, &claim_token)
                    .err();
                let sync_warning = scheduler.sync_all(Utc::now()).await.err();

                match (cleanup_warning, sync_warning) {
                    (None, None) => Ok(PendingCronTriggerFinalize::Completed),
                    (Some(cleanup), None) => Ok(PendingCronTriggerFinalize::CompletedWithWarning(
                        format!(
                            "Failed to clean up completed one-shot cron job {} after delivery: {}",
                            job_id, cleanup
                        ),
                    )),
                    (None, Some(sync)) => Ok(PendingCronTriggerFinalize::CompletedWithWarning(
                        format!("Failed to resync edge rules afterward: {}", sync),
                    )),
                    (Some(cleanup), Some(sync)) => Ok(
                        PendingCronTriggerFinalize::CompletedWithWarning(format!(
                            "Failed to clean up completed one-shot cron job {} after delivery: {}. Failed to resync edge rules afterward: {}",
                            job_id, cleanup, sync
                        )),
                    ),
                }
            }
        }
    }
}

impl CronActor {
    pub fn new(
        name: &str,
        db_path: &str,
        logger_tx: LoggerHandle,
        scheduling_mode: CronSchedulingMode,
        bus_tx: tokio::sync::mpsc::Sender<crate::bus::BusMessage>,
    ) -> Result<Self, String> {
        let store = CronStore::new(db_path)?;
        let jobs = store.load_jobs()?;
        let _ = logger_tx.send(crate::bus::BusMessage::Log(crate::bus::LogEvent::info(
            name,
            &format!("Loaded {} cron jobs from database.", jobs.len()),
        )));

        Ok(Self {
            name: name.to_string(),
            jobs,
            cron_schedule_cache: HashMap::new(),
            store,
            logger_tx,
            scheduling_mode,
            bus_tx,
        })
    }

    fn log_error(&self, message: impl AsRef<str>) {
        let _ = self
            .logger_tx
            .send(crate::bus::BusMessage::Log(crate::bus::LogEvent::error(
                &self.name,
                message.as_ref(),
            )));
    }
}

#[async_trait]
impl ActorLogic<String> for CronActor {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn process(&mut self, packet: String) -> Result<Option<(String, String)>, ActorError> {
        let Ok(cmd) = serde_json::from_str::<CronCommand>(&packet) else {
            return Ok(None);
        };

        if self.scheduling_mode == CronSchedulingMode::MultiTenantEdge {
            return Err(ActorError::from(
                "CronActor command processing is disabled when multi-tenant-edge cron scheduling is enabled",
            ));
        }

        match cmd {
            CronCommand::Add {
                id,
                schedule,
                message,
                chat_id,
                channel,
            } => {
                if let ScheduleKind::Cron { ref cron_expr } = schedule {
                    validate_cron_expression(cron_expr)
                        .map_err(|error| ActorError::from(error.to_string()))?;
                }

                let job = ActiveJob {
                    id: id.clone(),
                    schedule: schedule.clone(),
                    message: message.clone(),
                    last_run_at_ms: None,
                    chat_id: chat_id.clone(),
                    channel: channel.clone(),
                    webhook_token: generate_webhook_token(),
                };
                self.store.insert_job(&job).map_err(ActorError::from)?;
                if let ScheduleKind::Cron { ref cron_expr } = schedule {
                    let parsed_schedule = Schedule::from_str(cron_expr).map_err(|error| {
                        ActorError::from(format!("Invalid cron expression: {}", error))
                    })?;
                    self.cron_schedule_cache
                        .insert(cron_expr.clone(), parsed_schedule);
                }
                self.jobs.push(job);

                let _ =
                    self.logger_tx
                        .send(crate::bus::BusMessage::Log(crate::bus::LogEvent::info(
                            &self.name,
                            &format!("Added job '{}' with schedule {:?}", id, schedule),
                        )));
            }
            CronCommand::Remove { id } => {
                if let Some(cron_expr) =
                    self.jobs
                        .iter()
                        .find(|job| job.id == id)
                        .and_then(|job| match &job.schedule {
                            ScheduleKind::Cron { cron_expr } => Some(cron_expr.clone()),
                            _ => None,
                        })
                {
                    self.cron_schedule_cache.remove(&cron_expr);
                }
                self.jobs.retain(|job| job.id != id);
                self.store.remove_job(&id).map_err(ActorError::from)?;
                let _ =
                    self.logger_tx
                        .send(crate::bus::BusMessage::Log(crate::bus::LogEvent::info(
                            &self.name,
                            &format!("Removed job '{}'", id),
                        )));
            }
            CronCommand::Reload => {
                self.jobs = self.store.load_jobs().map_err(ActorError::from)?;
                // Cache entries correspond to persisted expressions. Rebuild
                // lazily on the next tick, matching startup behavior and
                // preserving the existing invalid-expression diagnostics.
                self.cron_schedule_cache.clear();
                let _ =
                    self.logger_tx
                        .send(crate::bus::BusMessage::Log(crate::bus::LogEvent::info(
                            &self.name,
                            &format!("Reloaded {} cron jobs from database.", self.jobs.len()),
                        )));
            }
        }

        Ok(None)
    }

    fn tick_interval(&self) -> Option<tokio::time::Duration> {
        match self.scheduling_mode {
            CronSchedulingMode::Local => Some(tokio::time::Duration::from_secs(1)),
            CronSchedulingMode::MultiTenantEdge => None,
        }
    }

    async fn on_tick(&mut self) -> Result<Option<(String, String)>, ActorError> {
        if self.scheduling_mode != CronSchedulingMode::Local {
            return Ok(None);
        }

        let now = Utc::now();
        let now_ms = now.timestamp_millis();
        let mut triggered = Vec::new();
        let mut jobs_to_remove = Vec::new();
        let logger_tx = self.logger_tx.clone();
        let actor_name = self.name.clone();
        let store = self.store.clone();

        {
            let jobs = &mut self.jobs;
            let cron_schedule_cache = &mut self.cron_schedule_cache;

            for job in jobs {
                let mut should_trigger = false;

                match &job.schedule {
                    ScheduleKind::At { at_ms } => {
                        if now_ms >= *at_ms {
                            should_trigger = true;
                            jobs_to_remove.push(job.id.clone());
                        }
                    }
                    ScheduleKind::Every { every_ms } => {
                        let last = job.last_run_at_ms.unwrap_or(now_ms);
                        let mut updated = false;

                        if job.last_run_at_ms.is_none() {
                            job.last_run_at_ms = Some(now_ms);
                            updated = true;
                        } else if (now_ms - last) >= *every_ms {
                            should_trigger = true;
                            job.last_run_at_ms = Some(now_ms);
                            updated = true;
                        }

                        if updated {
                            store
                                .update_last_run_at_ms(&job.id, job.last_run_at_ms)
                                .map_err(ActorError::from)?;
                        }
                    }
                    ScheduleKind::Cron { cron_expr } => {
                        if !cron_schedule_cache.contains_key(cron_expr) {
                            match Schedule::from_str(cron_expr) {
                                Ok(schedule) => {
                                    cron_schedule_cache.insert(cron_expr.clone(), schedule);
                                }
                                Err(error) => {
                                    let _ = logger_tx.send(crate::bus::BusMessage::Log(
                                        crate::bus::LogEvent::error(
                                            &actor_name,
                                            &format!(
                                                "Invalid persisted cron expression for {}: {}",
                                                job.id, error
                                            ),
                                        ),
                                    ));
                                    continue;
                                }
                            }
                        }

                        if let Some(schedule) = cron_schedule_cache.get(cron_expr) {
                            // Use last_run_at_ms as the lookback point so we catch
                            // triggers that were missed while the app was shut down.
                            // Falls back to a 1-second window when last_run_at_ms is
                            // not yet set (brand-new job that hasn't fired yet).
                            let lookback = match job.last_run_at_ms {
                                Some(ms) => DateTime::from_timestamp_millis(ms)
                                    .unwrap_or(now - chrono::Duration::seconds(1)),
                                None => now - chrono::Duration::seconds(1),
                            };
                            if let Some(next) = schedule.after(&lookback).next() {
                                if next <= now {
                                    should_trigger = true;
                                    job.last_run_at_ms = Some(now_ms);
                                    let _ = store.update_last_run_at_ms(&job.id, Some(now_ms));
                                }
                            }
                        }
                    }
                }

                if should_trigger {
                    triggered.push((
                        job.id.clone(),
                        job.channel.clone(),
                        job.chat_id.clone(),
                        job.message.clone(),
                    ));
                }
            }
        }

        self.jobs.retain(|job| !jobs_to_remove.contains(&job.id));
        for job_id in jobs_to_remove {
            if let Err(error) = self.store.remove_job(&job_id) {
                self.log_error(format!(
                    "Failed to remove expired one-shot cron job {}: {}",
                    job_id, error
                ));
            }
        }

        for (job_id, channel, chat_id, message) in triggered {
            match self
                .store
                .record_cron_background_job(&job_id, &channel, &chat_id, &message, now_ms)
            {
                Ok(true) => {
                    let mut metadata = HashMap::new();
                    metadata.insert(
                        crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
                        serde_json::json!(format!("cron:{}", job_id)),
                    );
                    metadata.insert("cron_job_id".to_string(), serde_json::json!(job_id.clone()));
                    metadata.insert(
                        crate::bus::METADATA_SYNTHETIC_CRON_TRIGGER.to_string(),
                        serde_json::json!(true),
                    );
                    metadata.insert(
                        crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS.to_string(),
                        serde_json::json!(true),
                    );

                    let inbound = crate::bus::InboundMessage {
                        channel,
                        sender_id: "cron_scheduler".to_string(),
                        chat_id,
                        thread_id: None,
                        content: message.clone(),
                        attachments: Vec::new(),
                        metadata,
                    };

                    let _ = self
                        .bus_tx
                        .send(crate::bus::BusMessage::Inbound(inbound))
                        .await;

                    let _ = self.logger_tx.send(crate::bus::BusMessage::Telemetry(
                        crate::bus::TelemetryEvent::CronTrigger {
                            job_id: job_id.clone(),
                            message: message.clone(),
                        },
                    ));
                    let _ = self.logger_tx.send(crate::bus::BusMessage::Log(
                        crate::bus::LogEvent::info(
                            &self.name,
                            &format!("Fired local cron job {}", job_id),
                        ),
                    ));
                }
                Ok(false) => {
                    // Skip trigger, already active
                }
                Err(error) => {
                    self.log_error(format!(
                        "Failed to record cron background job for {}: {}",
                        job_id, error
                    ));
                }
            }
        }

        Ok(None)
    }
}

pub fn validate_multi_tenant_edge_runtime(api_enabled: bool) -> Result<(), String> {
    if !api_enabled {
        return Err(
            "multi-tenant-edge cron scheduling requires [api].enabled = true because the edge wakes jobs through GET /_mte/cron/{job_id}/{token}"
                .to_string(),
        );
    }
    Ok(())
}

pub async fn sync_multi_tenant_edge_cron_jobs(
    store: &CronStore,
    client: &CronRegistrationClient,
    now: DateTime<Utc>,
) -> Result<(), String> {
    let jobs = store.load_jobs()?;
    let cron_rules = build_multi_tenant_edge_cron_rules(&jobs, now)?;
    client.sync_cron_rules(&cron_rules).await
}

pub fn validate_multi_tenant_edge_schedule(
    schedule: &ScheduleKind,
    now: DateTime<Utc>,
) -> Result<(), String> {
    match schedule {
        ScheduleKind::Every { .. } => Err(
            "every_seconds is not supported when [multi_tenant_edge].cron_scheduling_enabled = true"
                .to_string(),
        ),
        ScheduleKind::At { at_ms } => {
            if *at_ms <= now.timestamp_millis() {
                return Err(
                    "one-shot 'at' cron jobs must be scheduled in the future when [multi_tenant_edge].cron_scheduling_enabled = true"
                        .to_string(),
                );
            }
            Ok(())
        }
        ScheduleKind::Cron { cron_expr } => {
            validate_cron_expression(cron_expr)?;
            if !is_six_field_cron_expr(cron_expr) {
                return Err(
                    "cron_expr must be a 6-field UTC cron expression when [multi_tenant_edge].cron_scheduling_enabled = true"
                        .to_string(),
                );
            }
            Ok(())
        }
    }
}

pub fn validate_cron_expression(expr: &str) -> Result<(), String> {
    Schedule::from_str(expr)
        .map(|_| ())
        .map_err(|error| format!("Invalid cron expression: {}", error))
}

pub fn is_six_field_cron_expr(expr: &str) -> bool {
    expr.split_whitespace().count() == 6
}

pub fn generate_webhook_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn build_multi_tenant_edge_cron_rules(
    jobs: &[ActiveJob],
    now: DateTime<Utc>,
) -> Result<Vec<CronRule>, String> {
    jobs.iter()
        .map(|job| build_multi_tenant_edge_cron_rule(job, now))
        .collect()
}

fn build_multi_tenant_edge_cron_rule(
    job: &ActiveJob,
    now: DateTime<Utc>,
) -> Result<CronRule, String> {
    validate_multi_tenant_edge_schedule(&job.schedule, now)?;

    let schedule = match &job.schedule {
        ScheduleKind::At { at_ms } => at_schedule_to_utc_cron(*at_ms)?,
        ScheduleKind::Cron { cron_expr } => cron_expr.clone(),
        ScheduleKind::Every { .. } => unreachable!("validated above"),
    };

    Ok(CronRule {
        schedule,
        path: webhook_path(&job.id, &job.webhook_token),
    })
}

fn at_schedule_to_utc_cron(at_ms: i64) -> Result<String, String> {
    let at = DateTime::<Utc>::from_timestamp_millis(at_ms)
        .ok_or_else(|| format!("Invalid one-shot 'at' timestamp in cron job: {}", at_ms))?;
    Ok(format!(
        "{} {} {} {} {} *",
        at.second(),
        at.minute(),
        at.hour(),
        at.day(),
        at.month()
    ))
}

fn webhook_path(job_id: &str, token: &str) -> String {
    format!("{}/{}/{}", WEBHOOK_PATH_PREFIX, job_id, token)
}

fn decode_schedule(schedule_json: &str) -> Result<ScheduleKind, rusqlite::Error> {
    serde_json::from_str(schedule_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            schedule_json.len(),
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_multi_tenant_edge_cron_rules, generate_webhook_token,
        sync_multi_tenant_edge_cron_jobs, validate_multi_tenant_edge_runtime, ActiveJob, CronActor,
        CronCommand, CronSchedulingMode, CronStore, MultiTenantEdgeCronScheduler,
        PendingCronTriggerFinalize, ScheduleKind,
    };
    use crate::logging::create_logger_channel;
    use crate::multi_tenant_edge::{CronRegistrationClient, CronRule, CronTransport};
    use crate::ActorLogic;
    use async_trait::async_trait;
    use chrono::{Datelike, Timelike, Utc};
    use reqwest::StatusCode;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct LocalTempDir {
        path: std::path::PathBuf,
    }

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    impl LocalTempDir {
        fn new() -> Self {
            let unique = format!(
                "isanagent-scheduler-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos(),
                NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("tempdir");
            Self { path }
        }

        fn db_path(&self) -> std::path::PathBuf {
            self.path.join("agent.db")
        }
    }

    impl Drop for LocalTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[tokio::test]
    async fn reload_command_observes_host_persisted_jobs() {
        let temp = LocalTempDir::new();
        let db_path = temp.db_path();
        let db_path_str = db_path.to_string_lossy();
        let store = CronStore::new(&db_path_str).expect("cron store");
        let (logger, _logger_rx) = create_logger_channel(8);
        let (bus_tx, _bus_rx) = tokio::sync::mpsc::channel(1);
        let mut actor = CronActor::new(
            "test-cron",
            &db_path_str,
            logger,
            CronSchedulingMode::Local,
            bus_tx,
        )
        .expect("cron actor");
        assert!(actor.jobs.is_empty());

        store
            .insert_job(&ActiveJob {
                id: "host-job".to_string(),
                schedule: ScheduleKind::Every { every_ms: 60_000 },
                message: "wake up".to_string(),
                last_run_at_ms: None,
                chat_id: "chat-1".to_string(),
                channel: "tauri".to_string(),
                webhook_token: generate_webhook_token(),
            })
            .expect("persist host job");

        actor
            .process(serde_json::to_string(&CronCommand::Reload).expect("serialize reload"))
            .await
            .expect("reload command");
        assert_eq!(actor.jobs.len(), 1);
        assert_eq!(actor.jobs[0].id, "host-job");
    }

    #[derive(Clone)]
    struct RecordingCronTransport {
        records: Arc<Mutex<Vec<Vec<CronRule>>>>,
        statuses: Arc<Mutex<Vec<StatusCode>>>,
    }

    #[async_trait]
    impl CronTransport for RecordingCronTransport {
        async fn put_crons(
            &self,
            _url: &str,
            _token: &str,
            cron_rules: &[CronRule],
        ) -> Result<StatusCode, String> {
            self.records.lock().unwrap().push(cron_rules.to_vec());
            Ok(self
                .statuses
                .lock()
                .unwrap()
                .pop()
                .unwrap_or(StatusCode::NO_CONTENT))
        }
    }

    fn sample_job(id: &str, schedule: ScheduleKind) -> ActiveJob {
        ActiveJob {
            id: id.to_string(),
            schedule,
            message: format!("message for {}", id),
            last_run_at_ms: None,
            chat_id: "chat-123".to_string(),
            channel: "terminal".to_string(),
            webhook_token: generate_webhook_token(),
        }
    }

    #[test]
    fn mte_runtime_requires_api_listener() {
        let error = validate_multi_tenant_edge_runtime(false).expect_err("api disabled");
        assert!(error.contains("[api].enabled = true"));
    }

    #[test]
    fn mte_rule_derivation_keeps_cron_expr_and_path() {
        let job = sample_job(
            "job-1",
            ScheduleKind::Cron {
                cron_expr: "0 15 9 * * *".to_string(),
            },
        );
        let rules = build_multi_tenant_edge_cron_rules(std::slice::from_ref(&job), Utc::now())
            .expect("rules");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].schedule, "0 15 9 * * *");
        assert_eq!(
            rules[0].path,
            format!("/_mte/cron/{}/{}", job.id, job.webhook_token)
        );
    }

    #[test]
    fn mte_rule_derivation_converts_at_to_utc_cron() {
        let at = Utc::now() + chrono::Duration::minutes(5);
        let job = sample_job(
            "job-1",
            ScheduleKind::At {
                at_ms: at.timestamp_millis(),
            },
        );
        let rules = build_multi_tenant_edge_cron_rules(&[job], Utc::now()).expect("rules");
        assert_eq!(
            rules[0].schedule,
            format!(
                "{} {} {} {} {} *",
                at.second(),
                at.minute(),
                at.hour(),
                at.day(),
                at.month()
            )
        );
    }

    #[tokio::test]
    async fn sync_multi_tenant_edge_cron_jobs_pushes_full_rule_set() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("store");
        let job = sample_job(
            "job-1",
            ScheduleKind::Cron {
                cron_expr: "0 15 9 * * *".to_string(),
            },
        );
        store.insert_job(&job).expect("insert");

        let records = Arc::new(Mutex::new(Vec::new()));
        let client = CronRegistrationClient::new_with_transport(
            "https://edge.example.com/_internal/crons".to_string(),
            "cron-token".to_string(),
            Arc::new(RecordingCronTransport {
                records: records.clone(),
                statuses: Arc::new(Mutex::new(Vec::new())),
            }),
        );

        sync_multi_tenant_edge_cron_jobs(&store, &client, Utc::now())
            .await
            .expect("sync");

        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].len(), 1);
        assert_eq!(
            records[0][0].path,
            format!("/_mte/cron/{}/{}", job.id, job.webhook_token)
        );
    }

    #[tokio::test]
    async fn add_job_rolls_back_when_edge_sync_fails() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("store");
        let scheduler = MultiTenantEdgeCronScheduler::new_with_store(
            store.clone(),
            CronRegistrationClient::new_with_transport(
                "https://edge.example.com/_internal/crons".to_string(),
                "cron-token".to_string(),
                Arc::new(RecordingCronTransport {
                    records: Arc::new(Mutex::new(Vec::new())),
                    statuses: Arc::new(Mutex::new(vec![StatusCode::INTERNAL_SERVER_ERROR])),
                }),
            ),
        );
        let job = sample_job(
            "job-1",
            ScheduleKind::Cron {
                cron_expr: "0 15 9 * * *".to_string(),
            },
        );

        let error = scheduler
            .add_job(job.clone(), Utc::now())
            .await
            .expect_err("sync should fail");
        assert!(error.contains("cron sync failed"));
        assert!(store.find_job(&job.id).expect("job lookup").is_none());
    }

    #[tokio::test]
    async fn remove_job_rolls_back_when_edge_sync_fails() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("store");
        let job = sample_job(
            "job-1",
            ScheduleKind::Cron {
                cron_expr: "0 15 9 * * *".to_string(),
            },
        );
        store.insert_job(&job).expect("insert");
        let scheduler = MultiTenantEdgeCronScheduler::new_with_store(
            store.clone(),
            CronRegistrationClient::new_with_transport(
                "https://edge.example.com/_internal/crons".to_string(),
                "cron-token".to_string(),
                Arc::new(RecordingCronTransport {
                    records: Arc::new(Mutex::new(Vec::new())),
                    statuses: Arc::new(Mutex::new(vec![StatusCode::INTERNAL_SERVER_ERROR])),
                }),
            ),
        );

        let error = scheduler
            .remove_job(&job.id, Utc::now())
            .await
            .expect_err("sync should fail");
        assert!(error.contains("cron sync failed"));
        assert!(store.find_job(&job.id).expect("job lookup").is_some());
    }

    #[tokio::test]
    async fn one_shot_trigger_claim_blocks_duplicate_until_completion() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("store");
        let job = sample_job(
            "job-1",
            ScheduleKind::At {
                at_ms: (Utc::now() + chrono::Duration::minutes(1)).timestamp_millis(),
            },
        );
        store.insert_job(&job).expect("insert");
        let records = Arc::new(Mutex::new(Vec::new()));
        let scheduler = MultiTenantEdgeCronScheduler::new_with_store(
            store.clone(),
            CronRegistrationClient::new_with_transport(
                "https://edge.example.com/_internal/crons".to_string(),
                "cron-token".to_string(),
                Arc::new(RecordingCronTransport {
                    records: records.clone(),
                    statuses: Arc::new(Mutex::new(Vec::new())),
                }),
            ),
        );

        let first = scheduler
            .begin_trigger(&job.id, &job.webhook_token, Utc::now())
            .await
            .expect("first trigger");
        let second = scheduler
            .begin_trigger(&job.id, &job.webhook_token, Utc::now())
            .await;
        assert!(matches!(second, Err(super::CronWebhookError::NotFound)));

        match first.complete().await.expect("complete") {
            PendingCronTriggerFinalize::Completed => {}
            PendingCronTriggerFinalize::CompletedWithWarning(error) => {
                panic!("unexpected sync warning: {}", error)
            }
        }

        assert!(store.find_job(&job.id).expect("job lookup").is_none());
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].is_empty());
    }

    #[tokio::test]
    async fn one_shot_trigger_rollback_keeps_job_available() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("store");
        let now = Utc::now();
        let original_at_ms = (now - chrono::Duration::seconds(1)).timestamp_millis();
        let job = sample_job(
            "job-1",
            ScheduleKind::At {
                at_ms: original_at_ms,
            },
        );
        store.insert_job(&job).expect("insert");
        let records = Arc::new(Mutex::new(Vec::new()));
        let scheduler = MultiTenantEdgeCronScheduler::new_with_store(
            store.clone(),
            CronRegistrationClient::new_with_transport(
                "https://edge.example.com/_internal/crons".to_string(),
                "cron-token".to_string(),
                Arc::new(RecordingCronTransport {
                    records: records.clone(),
                    statuses: Arc::new(Mutex::new(Vec::new())),
                }),
            ),
        );

        let first = scheduler
            .begin_trigger(&job.id, &job.webhook_token, Utc::now())
            .await
            .expect("first trigger");
        first.rollback().await.expect("rollback");

        let rescheduled = store
            .find_job(&job.id)
            .expect("job lookup")
            .expect("job should still exist");
        match rescheduled.schedule {
            ScheduleKind::At { at_ms } => {
                assert!(at_ms > now.timestamp_millis());
                assert!(at_ms > original_at_ms);
            }
            other => panic!("expected rescheduled one-shot job, got {:?}", other),
        }
        assert_eq!(records.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn one_shot_trigger_mark_delivered_blocks_duplicate_before_completion() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("store");
        let job = sample_job(
            "job-1",
            ScheduleKind::At {
                at_ms: (Utc::now() + chrono::Duration::minutes(1)).timestamp_millis(),
            },
        );
        store.insert_job(&job).expect("insert");
        let records = Arc::new(Mutex::new(Vec::new()));
        let scheduler = MultiTenantEdgeCronScheduler::new_with_store(
            store.clone(),
            CronRegistrationClient::new_with_transport(
                "https://edge.example.com/_internal/crons".to_string(),
                "cron-token".to_string(),
                Arc::new(RecordingCronTransport {
                    records: records.clone(),
                    statuses: Arc::new(Mutex::new(Vec::new())),
                }),
            ),
        );

        let first = scheduler
            .begin_trigger(&job.id, &job.webhook_token, Utc::now())
            .await
            .expect("first trigger");
        first
            .mark_delivered(Utc::now().timestamp_millis())
            .expect("mark delivered");

        let second = scheduler
            .begin_trigger(
                &job.id,
                &job.webhook_token,
                Utc::now() + chrono::Duration::minutes(2),
            )
            .await;
        assert!(matches!(second, Err(super::CronWebhookError::NotFound)));
        assert!(store.find_job(&job.id).expect("job lookup").is_none());

        sync_multi_tenant_edge_cron_jobs(
            &store,
            &CronRegistrationClient::new_with_transport(
                "https://edge.example.com/_internal/crons".to_string(),
                "cron-token".to_string(),
                Arc::new(RecordingCronTransport {
                    records: records.clone(),
                    statuses: Arc::new(Mutex::new(Vec::new())),
                }),
            ),
            Utc::now(),
        )
        .await
        .expect("sync");
        assert!(records.lock().unwrap().iter().any(|rules| rules.is_empty()));
    }
}
