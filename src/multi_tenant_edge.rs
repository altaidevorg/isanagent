use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

use crate::bus::{BusMessage, LogEvent, LogLevel};
use crate::logging::LoggerHandle;
use crate::tool_activity::{
    ToolExecutionActivity, ToolExecutionActivityHandle, ToolExecutionActivityHandleFuture,
};

const ACTIVITY_PATH: &str = "/agent-dorm/activity";
const CRONS_PATH: &str = "/agent-dorm/crons";
const DEFAULT_HEARTBEAT_TTL_MS: u64 = 30_000;
const MAX_HEARTBEAT_INTERVAL_MS: u64 = 30_000;
const MIN_HEARTBEAT_INTERVAL_MS: u64 = 1_000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub struct ActivityHeartbeatClient {
    transport: Arc<dyn HeartbeatTransport>,
    url: String,
    token: String,
    interval: Duration,
    logger_tx: LoggerHandle,
}

#[derive(Clone)]
pub struct CronRegistrationClient {
    transport: Arc<dyn CronTransport>,
    url: String,
    token: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CronRule {
    pub schedule: String,
    pub path: String,
}

#[derive(Debug, Serialize)]
struct PutCronsBody {
    cron_rules: Vec<CronRule>,
}

#[async_trait]
pub(crate) trait HeartbeatTransport: Send + Sync {
    async fn post_activity(&self, url: &str, token: &str) -> Result<StatusCode, String>;
}

#[async_trait]
pub(crate) trait CronTransport: Send + Sync {
    async fn put_crons(
        &self,
        url: &str,
        token: &str,
        cron_rules: &[CronRule],
    ) -> Result<StatusCode, String>;
}

struct ReqwestHeartbeatTransport {
    client: reqwest::Client,
}

struct ReqwestCronTransport {
    client: reqwest::Client,
}

impl ReqwestHeartbeatTransport {
    fn new() -> Result<Self, String> {
        Ok(Self {
            client: build_reqwest_client()?,
        })
    }
}

impl ReqwestCronTransport {
    fn new() -> Result<Self, String> {
        Ok(Self {
            client: build_reqwest_client()?,
        })
    }
}

#[async_trait]
impl HeartbeatTransport for ReqwestHeartbeatTransport {
    async fn post_activity(&self, url: &str, token: &str) -> Result<StatusCode, String> {
        self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map(|response| response.status())
            .map_err(|error| error.to_string())
    }
}

#[async_trait]
impl CronTransport for ReqwestCronTransport {
    async fn put_crons(
        &self,
        url: &str,
        token: &str,
        cron_rules: &[CronRule],
    ) -> Result<StatusCode, String> {
        self.client
            .put(url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&PutCronsBody {
                cron_rules: cron_rules.to_vec(),
            })
            .send()
            .await
            .map(|response| response.status())
            .map_err(|error| error.to_string())
    }
}

enum HeartbeatLoopControl {
    Continue,
    Break,
}

impl ActivityHeartbeatClient {
    pub fn from_env(logger_tx: LoggerHandle) -> Result<Self, String> {
        let base_url = std::env::var("MTE_PROXY_BASE_URL").map_err(|_| {
            "multi-tenant-edge heartbeat enabled but MTE_PROXY_BASE_URL is not set".to_string()
        })?;
        let token = std::env::var("MTE_ACTIVITY_SECRET").map_err(|_| {
            "multi-tenant-edge heartbeat enabled but MTE_ACTIVITY_SECRET is not set".to_string()
        })?;
        let ttl_ms = parse_heartbeat_ttl_ms(std::env::var("MTE_HEARTBEAT_TTL_MS").ok(), &logger_tx);

        Self::new(
            normalize_activity_url(&base_url)?,
            token,
            heartbeat_interval_from_ttl_ms(ttl_ms),
            logger_tx,
        )
    }

    pub(crate) fn new(
        url: String,
        token: String,
        interval: Duration,
        logger_tx: LoggerHandle,
    ) -> Result<Self, String> {
        Ok(Self::new_with_transport(
            url,
            token,
            interval,
            logger_tx,
            Arc::new(ReqwestHeartbeatTransport::new()?),
        ))
    }

    pub(crate) fn new_with_transport(
        url: String,
        token: String,
        interval: Duration,
        logger_tx: LoggerHandle,
        transport: Arc<dyn HeartbeatTransport>,
    ) -> Self {
        Self {
            transport,
            url,
            token,
            interval,
            logger_tx,
        }
    }

    fn handle_heartbeat_result(
        &self,
        chat_id: &str,
        tool_name: &str,
        result: Result<StatusCode, String>,
        transient_failure_active: &mut bool,
    ) -> HeartbeatLoopControl {
        match result {
            Ok(StatusCode::NO_CONTENT) => {
                if *transient_failure_active {
                    self.log_recovered(chat_id, tool_name);
                    *transient_failure_active = false;
                }
                HeartbeatLoopControl::Continue
            }
            Ok(status) if is_permanent_heartbeat_failure(status) => {
                self.log_permanent_failure(
                    chat_id,
                    tool_name,
                    &format!(
                        "Background tool heartbeat stopped after {} from {}",
                        status, self.url
                    ),
                );
                HeartbeatLoopControl::Break
            }
            Ok(status) => {
                if !*transient_failure_active {
                    self.log_transient_failure(
                        chat_id,
                        tool_name,
                        &format!(
                            "Background tool heartbeat got transient status {} from {}; retrying",
                            status, self.url
                        ),
                    );
                    *transient_failure_active = true;
                }
                HeartbeatLoopControl::Continue
            }
            Err(error) => {
                if !*transient_failure_active {
                    self.log_transient_failure(
                        chat_id,
                        tool_name,
                        &format!(
                            "Background tool heartbeat request to {} failed: {}; retrying",
                            self.url, error
                        ),
                    );
                    *transient_failure_active = true;
                }
                HeartbeatLoopControl::Continue
            }
        }
    }

    fn log_permanent_failure(&self, chat_id: &str, tool_name: &str, message: &str) {
        self.log_with_level(chat_id, tool_name, message, LogLevel::Warn);
    }

    fn log_transient_failure(&self, chat_id: &str, tool_name: &str, message: &str) {
        self.log_with_level(chat_id, tool_name, message, LogLevel::Warn);
    }

    fn log_recovered(&self, chat_id: &str, tool_name: &str) {
        self.log_with_level(
            chat_id,
            tool_name,
            &format!("Background tool heartbeat recovered for {}", self.url),
            LogLevel::Info,
        );
    }

    fn log_with_level(&self, chat_id: &str, tool_name: &str, message: &str, level: LogLevel) {
        let log_event = LogEvent::new(level, "MultiTenantEdgeHeartbeat", message)
            .with_chat_id(chat_id)
            .with_metadata(serde_json::json!({
                "tool_name": tool_name,
                "activity_url": self.url,
            }));

        let _ = self.logger_tx.send(BusMessage::Log(log_event));
    }
}

impl CronRegistrationClient {
    pub fn from_env() -> Result<Self, String> {
        let base_url = std::env::var("MTE_PROXY_BASE_URL").map_err(|_| {
            "multi-tenant-edge cron scheduling enabled but MTE_PROXY_BASE_URL is not set"
                .to_string()
        })?;
        let token = std::env::var("MTE_CRON_SECRET").map_err(|_| {
            "multi-tenant-edge cron scheduling enabled but MTE_CRON_SECRET is not set".to_string()
        })?;

        Self::new(normalize_crons_url(&base_url)?, token)
    }

    pub(crate) fn new(url: String, token: String) -> Result<Self, String> {
        Ok(Self::new_with_transport(
            url,
            token,
            Arc::new(ReqwestCronTransport::new()?),
        ))
    }

    pub(crate) fn new_with_transport(
        url: String,
        token: String,
        transport: Arc<dyn CronTransport>,
    ) -> Self {
        Self {
            transport,
            url,
            token,
        }
    }

    pub async fn sync_cron_rules(&self, cron_rules: &[CronRule]) -> Result<(), String> {
        match self
            .transport
            .put_crons(&self.url, &self.token, cron_rules)
            .await
        {
            Ok(StatusCode::NO_CONTENT) => Ok(()),
            Ok(status) => Err(format!(
                "multi-tenant-edge cron sync failed with {} from {}",
                status, self.url
            )),
            Err(error) => Err(format!(
                "multi-tenant-edge cron sync request to {} failed: {}",
                self.url, error
            )),
        }
    }
}

impl ToolExecutionActivity for ActivityHeartbeatClient {
    fn start(&self, chat_id: &str, tool_name: &str) -> Box<dyn ToolExecutionActivityHandle> {
        Box::new(HeartbeatLoop::spawn(
            self.clone(),
            chat_id.to_string(),
            tool_name.to_string(),
        ))
    }
}

struct HeartbeatLoop {
    stop_tx: Option<oneshot::Sender<()>>,
    join_handle: JoinHandle<()>,
}

impl HeartbeatLoop {
    fn spawn(client: ActivityHeartbeatClient, chat_id: String, tool_name: String) -> Self {
        let (stop_tx, mut stop_rx) = oneshot::channel();
        let join_handle = tokio::spawn(async move {
            let mut transient_failure_active = false;
            if matches!(
                Self::await_heartbeat_attempt(
                    &client,
                    &chat_id,
                    &tool_name,
                    &mut stop_rx,
                    &mut transient_failure_active,
                )
                .await,
                HeartbeatLoopControl::Break
            ) {
                return;
            }

            let mut interval = tokio::time::interval(client.interval);
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
            // We already sent the immediate startup heartbeat above, so consume the
            // initial instant tick and wait a full interval before the next send.
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    _ = interval.tick() => {
                        if matches!(
                            Self::await_heartbeat_attempt(
                                &client,
                                &chat_id,
                                &tool_name,
                                &mut stop_rx,
                                &mut transient_failure_active,
                            ).await,
                            HeartbeatLoopControl::Break
                        ) {
                            break;
                        }
                    }
                }
            }
        });

        Self {
            stop_tx: Some(stop_tx),
            join_handle,
        }
    }

    async fn await_heartbeat_attempt(
        client: &ActivityHeartbeatClient,
        chat_id: &str,
        tool_name: &str,
        stop_rx: &mut oneshot::Receiver<()>,
        transient_failure_active: &mut bool,
    ) -> HeartbeatLoopControl {
        tokio::select! {
            _ = &mut *stop_rx => HeartbeatLoopControl::Break,
            result = client.transport.post_activity(&client.url, &client.token) => {
                client.handle_heartbeat_result(chat_id, tool_name, result, transient_failure_active)
            }
        }
    }

    async fn stop(mut self) {
        if let Some(stop_tx) = self.stop_tx.take() {
            let _ = stop_tx.send(());
        }
        let _ = self.join_handle.await;
    }
}

impl ToolExecutionActivityHandle for HeartbeatLoop {
    fn stop(self: Box<Self>) -> ToolExecutionActivityHandleFuture {
        Box::pin(async move { (*self).stop().await })
    }
}

fn build_reqwest_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(REQUEST_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|error| {
            format!(
                "failed to build multi-tenant-edge reqwest client: {}",
                error
            )
        })
}

fn is_permanent_heartbeat_failure(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::BAD_REQUEST
            | StatusCode::UNAUTHORIZED
            | StatusCode::FORBIDDEN
            | StatusCode::NOT_FOUND
            | StatusCode::NOT_IMPLEMENTED
    )
}

pub(crate) fn heartbeat_interval_from_ttl_ms(ttl_ms: Option<u64>) -> Duration {
    let ttl_ms = ttl_ms.unwrap_or(DEFAULT_HEARTBEAT_TTL_MS);
    let candidate = ttl_ms / 3;
    let bounded = candidate.clamp(MIN_HEARTBEAT_INTERVAL_MS, MAX_HEARTBEAT_INTERVAL_MS);
    Duration::from_millis(bounded)
}

fn parse_heartbeat_ttl_ms(ttl_ms: Option<String>, logger_tx: &LoggerHandle) -> Option<u64> {
    let ttl_ms = ttl_ms?;
    match ttl_ms.parse::<u64>() {
        Ok(ttl_ms) => Some(ttl_ms),
        Err(_) => {
            let _ = logger_tx.send(BusMessage::Log(LogEvent::warn(
                "MultiTenantEdgeHeartbeat",
                "Ignoring invalid MTE_HEARTBEAT_TTL_MS; expected an integer number of milliseconds",
            )));
            None
        }
    }
}

fn normalize_activity_url(base_url: &str) -> Result<String, String> {
    normalize_internal_url(base_url, ACTIVITY_PATH, "heartbeat")
}

fn normalize_crons_url(base_url: &str) -> Result<String, String> {
    normalize_internal_url(base_url, CRONS_PATH, "cron scheduling")
}

fn normalize_internal_url(
    base_url: &str,
    path: &str,
    feature_name: &str,
) -> Result<String, String> {
    let trimmed = base_url.trim_end_matches('/');
    let url = if trimmed.ends_with(path) {
        trimmed.to_string()
    } else {
        format!("{}{}", trimmed, path)
    };
    Url::parse(&url).map_err(|error| {
        format!(
            "multi-tenant-edge {} enabled but MTE_PROXY_BASE_URL '{}' is invalid: {}",
            feature_name, base_url, error
        )
    })?;
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::{
        heartbeat_interval_from_ttl_ms, normalize_activity_url, normalize_crons_url,
        CronRegistrationClient, CronRule, CronTransport,
    };
    use async_trait::async_trait;
    use reqwest::StatusCode;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn heartbeat_interval_defaults_to_one_third_of_default_ttl_when_ttl_missing() {
        assert_eq!(
            heartbeat_interval_from_ttl_ms(None),
            Duration::from_millis(10_000)
        );
    }

    #[test]
    fn heartbeat_interval_is_clamped_to_minimum() {
        assert_eq!(
            heartbeat_interval_from_ttl_ms(Some(1_500)),
            Duration::from_millis(1_000)
        );
    }

    #[test]
    fn heartbeat_interval_is_clamped_to_maximum() {
        assert_eq!(
            heartbeat_interval_from_ttl_ms(Some(180_000)),
            Duration::from_millis(30_000)
        );
    }

    #[test]
    fn heartbeat_interval_uses_one_third_of_ttl_when_in_bounds() {
        assert_eq!(
            heartbeat_interval_from_ttl_ms(Some(12_000)),
            Duration::from_millis(4_000)
        );
    }

    #[test]
    fn normalize_activity_url_appends_activity_path_once() {
        assert_eq!(
            normalize_activity_url("https://edge.example.com").expect("normalized url"),
            "https://edge.example.com/agent-dorm/activity"
        );
        assert_eq!(
            normalize_activity_url("https://edge.example.com/agent-dorm/activity")
                .expect("normalized url"),
            "https://edge.example.com/agent-dorm/activity"
        );
    }

    #[test]
    fn normalize_crons_url_appends_crons_path_once() {
        assert_eq!(
            normalize_crons_url("https://edge.example.com").expect("normalized url"),
            "https://edge.example.com/agent-dorm/crons"
        );
        assert_eq!(
            normalize_crons_url("https://edge.example.com/agent-dorm/crons")
                .expect("normalized url"),
            "https://edge.example.com/agent-dorm/crons"
        );
    }

    #[test]
    fn normalize_activity_url_rejects_invalid_urls() {
        let error = normalize_activity_url("://invalid").expect_err("invalid url");
        assert!(error.contains("MTE_PROXY_BASE_URL"));
    }

    #[test]
    fn normalize_crons_url_rejects_invalid_urls() {
        let error = normalize_crons_url("://invalid").expect_err("invalid url");
        assert!(error.contains("MTE_PROXY_BASE_URL"));
    }

    #[derive(Clone, Debug)]
    struct CronRequestRecord {
        url: String,
        authorization: String,
        cron_rules: Vec<CronRule>,
    }

    struct RecordingCronTransport {
        records: Arc<Mutex<Vec<CronRequestRecord>>>,
        status: StatusCode,
    }

    #[async_trait]
    impl CronTransport for RecordingCronTransport {
        async fn put_crons(
            &self,
            url: &str,
            token: &str,
            cron_rules: &[CronRule],
        ) -> Result<StatusCode, String> {
            self.records.lock().unwrap().push(CronRequestRecord {
                url: url.to_string(),
                authorization: format!("Bearer {}", token),
                cron_rules: cron_rules.to_vec(),
            });
            Ok(self.status)
        }
    }

    #[tokio::test]
    async fn cron_sync_uses_bearer_auth_and_expected_body() {
        let records = Arc::new(Mutex::new(Vec::new()));
        let client = CronRegistrationClient::new_with_transport(
            "https://edge.example.com/agent-dorm/crons".to_string(),
            "cron-token".to_string(),
            Arc::new(RecordingCronTransport {
                records: records.clone(),
                status: StatusCode::NO_CONTENT,
            }),
        );
        let cron_rules = vec![CronRule {
            schedule: "0 0 9 * * *".to_string(),
            path: "/_mte/cron/job-1/token-1".to_string(),
        }];

        client
            .sync_cron_rules(&cron_rules)
            .await
            .expect("cron sync succeeds");

        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].url, "https://edge.example.com/agent-dorm/crons");
        assert_eq!(records[0].authorization, "Bearer cron-token");
        assert_eq!(records[0].cron_rules, cron_rules);
    }
}
