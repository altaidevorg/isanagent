//! Ratatui-based interactive `isanagent onboard` flow (provider → optional URL → API key env name
//! → models → feature enable/disable toggles).

use std::io::{self, stdout};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use serde::Deserialize;
use tokio::runtime::Handle;

use crate::onboarding::OnboardOptions;
use crate::provider_registry;

/// Matches [`assets/onboarding/config.toml`] defaults for boolean-ish sections.
const FEATURE_TOGGLE_COUNT: usize = 15;
const FEATURE_TOGGLE_LABELS: [&str; FEATURE_TOGGLE_COUNT] = [
    "[terminal] stdin/stdout chat",
    "[slack]",
    "[email]",
    "[api] HTTP channel",
    "[api] serve_ui (browser UI on API port)",
    "[multi_tenant_edge] activity heartbeat",
    "[multi_tenant_edge] cron scheduling",
    "[jina] web_search / web_fetch fallback",
    "[memory]",
    "[harness.git_worktree]",
    "[harness.subagents]",
    "[harness.ml_engineer] ML policy overlay (see workspace/ML_ENGINEER_OVERLAY.md)",
    "[harness.execution]",
    "[harness.background_jobs] job tracking and auto-resume",
    "[harness.notifications] in-app background notifications",
];

fn default_feature_toggle_values() -> [bool; FEATURE_TOGGLE_COUNT] {
    [
        true,  // terminal
        false, // slack
        false, // email
        false, // api
        false, // serve_ui
        false, // mte activity
        false, // mte cron
        false, // jina
        true,  // memory
        true,  // harness git_worktree
        true,  // harness subagents
        true,  // harness ml_engineer
        true,  // harness execution
        true,  // harness background_jobs
        true,  // harness notifications
    ]
}

fn build_onboard_options_with_toggles(
    provider: ProviderChoice,
    model_id: String,
    chat_url: String,
    api_key_env: String,
    values: &[bool; FEATURE_TOGGLE_COUNT],
) -> OnboardOptions {
    // Known providers: write `provider_name` only and let the registry supply `base_url` at
    // runtime. `Custom` (openai_compatible) has no built-in URL, so we persist the URL the user
    // typed into `provider_base_url`.
    let (provider_name, provider_base_url) = match provider {
        ProviderChoice::Custom => (
            provider_registry::OPENAI_COMPATIBLE.to_string(),
            Some(chat_url),
        ),
        _ => (provider.provider_name().to_string(), None),
    };
    OnboardOptions {
        provider_name: Some(provider_name),
        provider_model: Some(model_id),
        provider_base_url,
        provider_api_key_env: Some(api_key_env),
        terminal_enable: Some(values[0]),
        slack_enabled: Some(values[1]),
        email_enabled: Some(values[2]),
        api_enabled: Some(values[3]),
        api_serve_ui: Some(values[4]),
        multi_tenant_activity_heartbeat: Some(values[5]),
        multi_tenant_cron_scheduling: Some(values[6]),
        jina_enabled: Some(values[7]),
        memory_enabled: Some(values[8]),
        harness_git_worktree_enabled: Some(values[9]),
        harness_subagents_enabled: Some(values[10]),
        harness_ml_engineer_enabled: Some(values[11]),
        harness_execution_enabled: Some(values[12]),
        harness_background_jobs_enabled: Some(values[13]),
        harness_notifications_enabled: Some(values[14]),
        ..Default::default()
    }
}

/// Outcome of the interactive wizard: merged CLI overrides for `onboard_workspace`.
pub struct InteractiveOnboardOutcome {
    pub options: OnboardOptions,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProviderChoice {
    Gemini,
    OpenAI,
    DeepSeek,
    OpenRouter,
    Custom,
}

impl ProviderChoice {
    fn label(self) -> &'static str {
        match self {
            ProviderChoice::Gemini => "Gemini",
            ProviderChoice::OpenAI => "OpenAI",
            ProviderChoice::DeepSeek => "DeepSeek",
            ProviderChoice::OpenRouter => "OpenRouter",
            ProviderChoice::Custom => "Custom AI Compatible",
        }
    }

    /// Canonical name written into `[provider].provider_name`. The four built-in entries match
    /// keys in [`crate::provider_registry::KNOWN_PROVIDERS`]; `Custom` maps to the
    /// [`crate::provider_registry::OPENAI_COMPATIBLE`] sentinel.
    fn provider_name(self) -> &'static str {
        match self {
            ProviderChoice::Gemini => "gemini",
            ProviderChoice::OpenAI => "openai",
            ProviderChoice::DeepSeek => "deepseek",
            ProviderChoice::OpenRouter => "openrouter",
            ProviderChoice::Custom => provider_registry::OPENAI_COMPATIBLE,
        }
    }

    fn chat_completions_url(self) -> Option<&'static str> {
        match self {
            ProviderChoice::Custom => None,
            other => provider_registry::lookup(other.provider_name()),
        }
    }
}

const PROVIDERS: [ProviderChoice; 5] = [
    ProviderChoice::Gemini,
    ProviderChoice::OpenAI,
    ProviderChoice::DeepSeek,
    ProviderChoice::OpenRouter,
    ProviderChoice::Custom,
];

#[derive(Deserialize)]
struct ModelsListResponse {
    #[serde(default)]
    data: Vec<ModelId>,
    #[serde(default)]
    models: Vec<ModelId>,
}

#[derive(Deserialize)]
struct ModelId {
    id: String,
}

fn models_endpoint_url(chat_completions_url: &str) -> String {
    let u = chat_completions_url.trim();
    if let Some(prefix) = u.strip_suffix("/chat/completions") {
        format!("{}/models", prefix.trim_end_matches('/'))
    } else {
        format!("{}/models", u.trim_end_matches('/'))
    }
}

fn normalize_custom_base_url(input: &str) -> Result<String, String> {
    let t = input.trim();
    if t.is_empty() {
        return Err("Base URL cannot be empty.".to_string());
    }
    let base = t.trim_end_matches('/');
    if base.ends_with("/chat/completions") {
        return Ok(base.to_string());
    }
    Ok(format!("{}/chat/completions", base))
}

async fn fetch_model_ids(
    client: &reqwest::Client,
    chat_url: &str,
    api_key: &str,
) -> Result<Vec<String>, String> {
    let url = models_endpoint_url(chat_url);
    let res = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("HTTP error: {}", e))?;
    let status = res.status();
    if !status.is_success() {
        let body = res
            .text()
            .await
            .map_err(|e| format!("Failed to read response body: {}", e))?;
        return Err(format!(
            "List models failed ({}): {}",
            status,
            body.chars().take(500).collect::<String>()
        ));
    }
    let text = res.text().await.map_err(|e| format!("Read body: {}", e))?;
    let parsed: ModelsListResponse =
        serde_json::from_str(&text).map_err(|e| format!("Invalid JSON from /models: {}", e))?;
    let mut ids: Vec<String> = if !parsed.data.is_empty() {
        parsed.data.into_iter().map(|m| m.id).collect()
    } else {
        parsed.models.into_iter().map(|m| m.id).collect()
    };
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        return Err("No models returned by the API.".to_string());
    }
    Ok(ids)
}

fn centered_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let percent_x = percent_x.clamp(1, 100);
    let percent_y = percent_y.clamp(1, 100);
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    let vertical = popup_layout[1];
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical)[1]
}

#[derive(Clone)]
enum Step {
    PickProvider {
        selected: usize,
    },
    CustomUrl {
        input: String,
    },
    ApiKey {
        input: String,
    },
    FetchingModels,
    FetchError {
        message: String,
    },
    PickModel {
        models: Vec<String>,
        selected: usize,
        page: usize,
    },
    FeatureToggles {
        models: Vec<String>,
        model_selected: usize,
        model_page: usize,
        selected: usize,
        page: usize,
        values: [bool; FEATURE_TOGGLE_COUNT],
    },
}

struct UiState {
    step: Step,
    provider: Option<ProviderChoice>,
    chat_url: String,
    provider_list_selected: usize,
    /// Env var name submitted on the API key env step (used for /models and config).
    pending_api_key_env: String,
    custom_url_error: Option<String>,
    api_key_env_error: Option<String>,
    /// In-flight `/models` request; dropped on Esc to cancel (sender disconnects).
    fetch_rx: Option<mpsc::Receiver<Result<Vec<String>, String>>>,
    /// Last feature-toggle row state when leaving the toggles step (e.g. ← to change model); reused on re-entry.
    feature_toggle_values_cache: Option<[bool; FEATURE_TOGGLE_COUNT]>,
}

impl UiState {
    fn new() -> Self {
        Self {
            step: Step::PickProvider { selected: 0 },
            provider: None,
            chat_url: String::new(),
            provider_list_selected: 0,
            pending_api_key_env: String::new(),
            custom_url_error: None,
            api_key_env_error: None,
            fetch_rx: None,
            feature_toggle_values_cache: None,
        }
    }
}

/// Restores the terminal if `run_ui_loop` panics or returns (raw mode + alternate screen).
struct TerminalUiGuard;

impl TerminalUiGuard {
    fn enter() -> Result<Self, String> {
        enable_raw_mode().map_err(|e| format!("enable_raw_mode: {}", e))?;
        let mut stdout = stdout();
        if let Err(e) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(format!("EnterAlternateScreen: {}", e));
        }
        Ok(Self)
    }
}

impl Drop for TerminalUiGuard {
    fn drop(&mut self) {
        let mut stdout = stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

pub fn run_interactive_collect(handle: &Handle) -> Result<InteractiveOnboardOutcome, String> {
    let _guard = TerminalUiGuard::enter()?;
    run_ui_loop(handle)
}

fn run_ui_loop(handle: &Handle) -> Result<InteractiveOnboardOutcome, String> {
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))
        .map_err(|e| format!("terminal: {}", e))?;

    let client = crate::utils::build_reqwest_client();
    let mut state = UiState::new();

    loop {
        terminal
            .draw(|f| render(f, &state))
            .map_err(|e| format!("draw: {}", e))?;

        if matches!(state.step, Step::FetchingModels) {
            let rx = match state.fetch_rx.take() {
                Some(r) => r,
                None => {
                    state.api_key_env_error = None;
                    state.step = Step::ApiKey {
                        input: state.pending_api_key_env.clone(),
                    };
                    continue;
                }
            };
            loop {
                terminal
                    .draw(|f| render(f, &state))
                    .map_err(|e| format!("draw: {}", e))?;
                match rx.try_recv() {
                    Ok(Ok(models)) => {
                        state.step = Step::PickModel {
                            models,
                            selected: 0,
                            page: 0,
                        };
                        break;
                    }
                    Ok(Err(e)) => {
                        state.step = Step::FetchError { message: e };
                        break;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        state.step = Step::FetchError {
                            message: "Model list request ended unexpectedly.".to_string(),
                        };
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                if event::poll(Duration::from_millis(50))
                    .map_err(|e| format!("event poll: {}", e))?
                {
                    let evt = event::read().map_err(|e| format!("event: {}", e))?;
                    if let Event::Key(key) = evt {
                        if key.kind == KeyEventKind::Release {
                            continue;
                        }
                        if matches!(key.code, KeyCode::Esc) {
                            state.api_key_env_error = None;
                            state.step = Step::ApiKey {
                                input: state.pending_api_key_env.clone(),
                            };
                            break;
                        }
                    }
                }
            }
            continue;
        }

        let evt = event::read().map_err(|e| format!("event: {}", e))?;
        let Event::Key(key) = evt else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }

        match &mut state.step {
            Step::PickProvider { selected } => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    return Err("Onboarding cancelled.".to_string());
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if *selected > 0 {
                        *selected -= 1;
                    }
                    state.provider_list_selected = *selected;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if *selected + 1 < PROVIDERS.len() {
                        *selected += 1;
                    }
                    state.provider_list_selected = *selected;
                }
                KeyCode::Enter => {
                    let choice = PROVIDERS[*selected];
                    state.provider = Some(choice);
                    state.custom_url_error = None;
                    if choice == ProviderChoice::Custom {
                        state.step = Step::CustomUrl {
                            input: String::new(),
                        };
                    } else {
                        state.chat_url = choice
                            .chat_completions_url()
                            .expect("preset URL")
                            .to_string();
                        state.step = Step::ApiKey {
                            input: String::new(),
                        };
                    }
                }
                _ => {}
            },
            Step::CustomUrl { input } => match key.code {
                KeyCode::Esc => {
                    return Err("Onboarding cancelled.".to_string());
                }
                KeyCode::Left => {
                    state.custom_url_error = None;
                    state.step = Step::PickProvider {
                        selected: state.provider_list_selected,
                    };
                    state.provider = None;
                }
                KeyCode::Enter => {
                    state.custom_url_error = None;
                    match normalize_custom_base_url(input) {
                        Ok(url) => {
                            state.chat_url = url;
                            state.step = Step::ApiKey {
                                input: String::new(),
                            };
                        }
                        Err(e) => {
                            state.custom_url_error = Some(e);
                        }
                    }
                }
                KeyCode::Backspace => {
                    input.pop();
                    state.custom_url_error = None;
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    state.custom_url_error = None;
                }
                _ => {}
            },
            Step::ApiKey { input } => match key.code {
                KeyCode::Esc => {
                    return Err("Onboarding cancelled.".to_string());
                }
                KeyCode::Left => {
                    state.api_key_env_error = None;
                    state.pending_api_key_env.clear();
                    if state.provider == Some(ProviderChoice::Custom) {
                        let base = state
                            .chat_url
                            .trim_end_matches('/')
                            .strip_suffix("/chat/completions")
                            .unwrap_or(state.chat_url.trim_end_matches('/'));
                        state.step = Step::CustomUrl {
                            input: base.to_string(),
                        };
                    } else {
                        state.step = Step::PickProvider {
                            selected: state.provider_list_selected,
                        };
                        state.provider = None;
                    }
                }
                KeyCode::Enter => {
                    let trimmed = input.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match std::env::var(trimmed) {
                        Ok(_) => {
                            state.api_key_env_error = None;
                            state.pending_api_key_env = trimmed.to_string();
                            let (tx, rx) = mpsc::channel();
                            let client = client.clone();
                            let url = state.chat_url.clone();
                            let env_name = state.pending_api_key_env.clone();
                            let rt_handle = handle.clone();
                            thread::spawn(move || {
                                let key = match std::env::var(&env_name) {
                                    Ok(k) => k,
                                    Err(_) => {
                                        let _ = tx.send(Err(format!(
                                            "Environment variable `{}` is not set.",
                                            env_name
                                        )));
                                        return;
                                    }
                                };
                                let res =
                                    rt_handle.block_on(fetch_model_ids(&client, &url, key.trim()));
                                let _ = tx.send(res);
                            });
                            state.fetch_rx = Some(rx);
                            state.step = Step::FetchingModels;
                        }
                        Err(_) => {
                            state.api_key_env_error = Some(format!(
                                "`{}` is not set in this process. Export it first, then press Enter.",
                                trimmed
                            ));
                        }
                    }
                }
                KeyCode::Backspace => {
                    input.pop();
                    state.api_key_env_error = None;
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    state.api_key_env_error = None;
                }
                _ => {}
            },
            Step::FetchingModels => {
                // Handled at the top of the loop (polling + non-blocking UI).
            }
            Step::FetchError { .. } => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    return Err("Onboarding cancelled.".to_string());
                }
                KeyCode::Enter | KeyCode::Char('r') | KeyCode::Left => {
                    state.api_key_env_error = None;
                    state.step = Step::ApiKey {
                        input: state.pending_api_key_env.clone(),
                    };
                }
                _ => {}
            },
            Step::PickModel {
                models,
                selected,
                page,
            } => {
                let size = terminal.size().map_err(|e| format!("size: {}", e))?;
                let area = Rect::new(0, 0, size.width, size.height);
                let block = centered_rect(area, 85, 80);
                let inner_h = block.height.saturating_sub(4) as usize;
                let per_page = inner_h.max(3);
                let n = models.len();
                let total_pages = n.div_ceil(per_page);

                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        return Err("Onboarding cancelled.".to_string());
                    }
                    KeyCode::Left => {
                        state.api_key_env_error = None;
                        state.step = Step::ApiKey {
                            input: state.pending_api_key_env.clone(),
                        };
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if *selected > 0 {
                            *selected -= 1;
                        }
                        *page = *selected / per_page;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if *selected + 1 < n {
                            *selected += 1;
                        }
                        *page = *selected / per_page;
                    }
                    KeyCode::Char('n') | KeyCode::PageDown => {
                        let next_page = (*page + 1).min(total_pages.saturating_sub(1));
                        *page = next_page;
                        *selected = (*page * per_page).min(n.saturating_sub(1));
                    }
                    KeyCode::Char('p') | KeyCode::PageUp => {
                        let prev_page = page.saturating_sub(1);
                        *page = prev_page;
                        *selected = (*page * per_page).min(n.saturating_sub(1));
                    }
                    KeyCode::Enter => {
                        if state.provider.is_none() {
                            return Err("internal: missing provider".to_string());
                        }
                        if models.get(*selected).is_none() {
                            return Err("invalid selection".to_string());
                        }
                        let values = state
                            .feature_toggle_values_cache
                            .unwrap_or_else(default_feature_toggle_values);
                        state.step = Step::FeatureToggles {
                            models: models.clone(),
                            model_selected: *selected,
                            model_page: *page,
                            selected: 0,
                            page: 0,
                            values,
                        };
                    }
                    _ => {}
                }
            }
            Step::FeatureToggles {
                models,
                model_selected,
                model_page,
                selected,
                page,
                values,
            } => {
                let size = terminal.size().map_err(|e| format!("size: {}", e))?;
                let area = Rect::new(0, 0, size.width, size.height);
                let block = centered_rect(area, 85, 80);
                let inner_h = block.height.saturating_sub(4) as usize;
                let per_page = inner_h.max(2);
                let n = FEATURE_TOGGLE_COUNT;
                let total_pages = n.div_ceil(per_page);

                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        return Err("Onboarding cancelled.".to_string());
                    }
                    KeyCode::Left => {
                        state.feature_toggle_values_cache = Some(*values);
                        state.step = Step::PickModel {
                            models: models.clone(),
                            selected: *model_selected,
                            page: *model_page,
                        };
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if *selected > 0 {
                            *selected -= 1;
                        }
                        *page = *selected / per_page;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if *selected + 1 < n {
                            *selected += 1;
                        }
                        *page = *selected / per_page;
                    }
                    KeyCode::Char('n') | KeyCode::PageDown => {
                        let next_page = (*page + 1).min(total_pages.saturating_sub(1));
                        *page = next_page;
                        *selected = (*page * per_page).min(n.saturating_sub(1));
                    }
                    KeyCode::Char('p') | KeyCode::PageUp => {
                        let prev_page = page.saturating_sub(1);
                        *page = prev_page;
                        *selected = (*page * per_page).min(n.saturating_sub(1));
                    }
                    KeyCode::Char(' ') => {
                        values[*selected] = !values[*selected];
                        state.feature_toggle_values_cache = Some(*values);
                    }
                    KeyCode::Enter => {
                        let provider = state
                            .provider
                            .ok_or_else(|| "internal: missing provider".to_string())?;
                        let id = models
                            .get(*model_selected)
                            .ok_or_else(|| "invalid model selection".to_string())?
                            .clone();
                        let options = build_onboard_options_with_toggles(
                            provider,
                            id,
                            state.chat_url.clone(),
                            state.pending_api_key_env.clone(),
                            values,
                        );
                        return Ok(InteractiveOnboardOutcome { options });
                    }
                    _ => {}
                }
            }
        }
    }
}

fn render(frame: &mut Frame, state: &UiState) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        area,
    );

    let block = centered_rect(area, 85, 80);
    match &state.step {
        Step::PickProvider { selected } => {
            let title = "Choose your model provider";
            let items: Vec<ListItem> = PROVIDERS
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let style = if i == *selected {
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    ListItem::new(Line::from(p.label())).style(style)
                })
                .collect();
            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_alignment(Alignment::Center),
            );
            frame.render_widget(list, block);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(1)])
                .split(block);
            let hint = Paragraph::new("↑/↓ navigate · Enter select · Esc or q quit")
                .alignment(Alignment::Center);
            frame.render_widget(hint, chunks[1]);
        }
        Step::CustomUrl { input } => {
            let err_line = state
                .custom_url_error
                .as_ref()
                .map(|e| Line::from(format!("Error: {}", e)).style(Style::default().fg(Color::Red)))
                .unwrap_or_else(|| Line::from(""));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(1)])
                .split(block);
            let p = Paragraph::new(vec![
                Line::from("API base URL (OpenAI-compatible), e.g. https://api.example.com/v1"),
                Line::from(""),
                Line::from(input.as_str()).style(Style::default().fg(Color::Yellow)),
                err_line,
            ])
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Custom provider")
                    .title_alignment(Alignment::Center),
            );
            frame.render_widget(p, chunks[0]);
            let footer =
                Paragraph::new("Enter confirm · ← back · Esc quit").alignment(Alignment::Center);
            frame.render_widget(footer, chunks[1]);
        }
        Step::ApiKey { input } => {
            let err_line = state
                .api_key_env_error
                .as_ref()
                .map(|e| Line::from(format!("Error: {}", e)).style(Style::default().fg(Color::Red)))
                .unwrap_or_else(|| Line::from(""));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(1)])
                .split(block);
            let p = Paragraph::new(vec![
                Line::from("Name of the environment variable that holds your API key"),
                Line::from("(must already be set in this process, e.g. GEMINI_API_KEY)"),
                Line::from(""),
                Line::from(input.as_str()).style(Style::default().fg(Color::Yellow)),
                err_line,
            ])
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("API key environment variable")
                    .title_alignment(Alignment::Center),
            );
            frame.render_widget(p, chunks[0]);
            let footer =
                Paragraph::new("Enter submit · ← back · Esc quit").alignment(Alignment::Center);
            frame.render_widget(footer, chunks[1]);
        }
        Step::FetchingModels => {
            let p = Paragraph::new("Listing models from /models …\n\nEsc to cancel")
                .alignment(Alignment::Center)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Please wait")
                        .title_alignment(Alignment::Center),
                );
            frame.render_widget(p, block);
        }
        Step::FetchError { message } => {
            let p = Paragraph::new(format!(
                "{}\n\nPress Enter, r, or ← to edit the env var name again. Esc quit.",
                message
            ))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Could not list models")
                    .title_alignment(Alignment::Center),
            );
            frame.render_widget(p, block);
        }
        Step::PickModel {
            models,
            selected,
            page,
        } => {
            let inner_h = block.height.saturating_sub(4) as usize;
            let per_page = inner_h.max(3);
            let n = models.len();
            let total_pages = n.div_ceil(per_page);
            let page = (*page).min(total_pages.saturating_sub(1));
            let start = page * per_page;
            let end = (start + per_page).min(n);
            let page_models = &models[start..end];
            let local_sel = selected.saturating_sub(start);

            let items: Vec<ListItem> = page_models
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let style = if i == local_sel {
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Green)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    ListItem::new(m.as_str()).style(style)
                })
                .collect();

            let title = format!(
                "Choose a model (page {}/{}, {} total)",
                page + 1,
                total_pages.max(1),
                n
            );
            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_alignment(Alignment::Center),
            );
            frame.render_widget(list, block);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(1)])
                .split(block);
            let hint =
                Paragraph::new("↑/↓ j/k move · n/p page · Enter confirm · ← back · Esc quit")
                    .alignment(Alignment::Center);
            frame.render_widget(hint, chunks[1]);
        }
        Step::FeatureToggles {
            selected,
            page,
            values,
            ..
        } => {
            let inner_h = block.height.saturating_sub(4) as usize;
            let per_page = inner_h.max(2);
            let n = FEATURE_TOGGLE_COUNT;
            let total_pages = n.div_ceil(per_page);
            let page = (*page).min(total_pages.saturating_sub(1));
            let start = page * per_page;
            let end = (start + per_page).min(n);
            let local_sel = selected.saturating_sub(start);

            let items: Vec<ListItem> = (start..end)
                .map(|i| {
                    let on = values[i];
                    let mark = if on { "[on] " } else { "[off]" };
                    let text = format!("{} {}", mark, FEATURE_TOGGLE_LABELS[i]);
                    let style = if i - start == local_sel {
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Magenta)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    ListItem::new(Line::from(text)).style(style)
                })
                .collect();

            let title = format!(
                "Enable / disable features (page {}/{}, {} items)",
                page + 1,
                total_pages.max(1),
                n
            );
            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_alignment(Alignment::Center),
            );
            frame.render_widget(list, block);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(2)])
                .split(block);
            let hint = Paragraph::new(vec![
                Line::from(
                    "↑/↓ j/k move · Space toggle · n/p page · Enter finish · ← back to models",
                ),
                Line::from("Esc or q quit"),
            ])
            .alignment(Alignment::Center);
            frame.render_widget(hint, chunks[1]);
        }
    }
}
