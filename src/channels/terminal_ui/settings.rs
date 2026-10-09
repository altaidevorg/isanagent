//! Settings popup for the terminal UI. Same three areas as the web screen:
//! live model switch, harness flags that wait for the next start, and skill install.

use std::collections::HashMap;

use crossterm::event::KeyCode;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::channels::harness_settings::HarnessSettings;
use crate::channels::skill_settings::InstalledSkill;
use crate::config::ProviderConfig;

use super::theme::Theme;

const HARNESS_ROWS: usize = 6;
const SKILL_ROWS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsTab {
    Models,
    Harness,
    Skills,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsModel {
    pub key: String,
    pub provider_name: String,
    pub model_name: String,
    pub has_key: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextField {
    ApiKey,
    SkillRepo,
    SkillName,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingSkillInstall {
    pub repo: String,
    pub skill_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsEffect {
    None,
    Close,
    SwitchModel { key: String },
    SetApiKey { key: String, secret: String },
    SaveHarness(HarnessSettings),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsPopup {
    tab: SettingsTab,
    cursor: usize,
    models: Vec<SettingsModel>,
    active_key: Option<String>,
    key_draft: Option<String>,
    harness: HarnessSettings,
    skills: Vec<InstalledSkill>,
    skills_available: bool,
    skill_repo: String,
    skill_name: String,
    editing: Option<TextField>,
    /// Skill whose description is open. `Esc` returns to the list.
    skill_detail: Option<usize>,
    skill_detail_scroll: usize,
    notice: Option<String>,
    error: Option<String>,
    pending_install: Option<PendingSkillInstall>,
}

impl SettingsPopup {
    pub fn open(
        providers: &HashMap<String, ProviderConfig>,
        active_key: Option<&str>,
        harness: HarnessSettings,
        skills: Option<Vec<InstalledSkill>>,
        error: Option<String>,
    ) -> Self {
        Self {
            tab: SettingsTab::Models,
            cursor: 0,
            models: models_from_providers(providers),
            active_key: active_key.map(str::to_owned),
            key_draft: None,
            harness,
            skills: skills.clone().unwrap_or_default(),
            skills_available: skills.is_some(),
            skill_repo: String::new(),
            skill_name: String::new(),
            editing: None,
            skill_detail: None,
            skill_detail_scroll: 0,
            notice: None,
            error,
            pending_install: None,
        }
    }

    pub fn showing_skill_detail(&self) -> bool {
        self.skill_detail.is_some()
    }

    pub fn note_model_active(&mut self, key: &str) {
        self.active_key = Some(key.to_string());
        self.notice = Some(format!("Using {key}. The next message uses it."));
        self.error = None;
    }

    pub fn note_key_saved(&mut self, key: &str) {
        let provider = self
            .models
            .iter()
            .find(|model| model.key == key)
            .map(|model| model.provider_name.clone());
        if let Some(provider) = provider {
            for model in &mut self.models {
                if model.provider_name.eq_ignore_ascii_case(&provider) {
                    model.has_key = true;
                }
            }
        }
        self.active_key = Some(key.to_string());
        self.notice = Some(format!("Key saved for {key}. The next message uses it."));
        self.error = None;
    }

    pub fn note_harness_saved(&mut self) {
        self.notice = Some(
            "Saved to config.toml. Restart isanagent before these changes take effect.".to_string(),
        );
        self.error = None;
    }

    pub fn note_skills(&mut self, skills: Vec<InstalledSkill>, installed: &[String]) {
        self.skills = skills;
        self.skill_repo.clear();
        self.skill_name.clear();
        self.skill_detail = None;
        self.skill_detail_scroll = 0;
        if self.tab == SettingsTab::Skills {
            let last = self.row_count().saturating_sub(1);
            if self.cursor > last {
                self.cursor = last;
            }
        }
        self.notice = Some(if installed.is_empty() {
            "No skills found in that repository.".to_string()
        } else {
            format!(
                "Installed {}. The next message can use them.",
                installed.join(", ")
            )
        });
        self.error = None;
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        self.error = Some(message.into());
        self.notice = None;
    }

    pub fn take_pending_install(&mut self) -> Option<PendingSkillInstall> {
        self.pending_install.take()
    }

    /// Bracketed paste (Cmd+V / Ctrl+V in the terminal). A key paste opens the key box.
    pub fn on_paste(&mut self, pasted: &str) -> SettingsEffect {
        if self.editing == Some(TextField::ApiKey) || self.ready_for_key_paste() {
            let cleaned = sanitize_api_key(pasted);
            if cleaned.is_empty() {
                return SettingsEffect::None;
            }
            self.editing = Some(TextField::ApiKey);
            self.key_draft
                .get_or_insert_with(String::new)
                .push_str(&cleaned);
            self.error = None;
            return SettingsEffect::None;
        }
        if let Some(field) = self.editing {
            let cleaned: String = pasted
                .chars()
                .filter(|ch| *ch != '\r' && *ch != '\n' && !ch.is_control())
                .collect();
            match field {
                TextField::SkillRepo => self.skill_repo.push_str(cleaned.trim()),
                TextField::SkillName => self.skill_name.push_str(cleaned.trim()),
                TextField::ApiKey => {}
            }
        }
        SettingsEffect::None
    }

    fn ready_for_key_paste(&self) -> bool {
        self.editing.is_none()
            && self.tab == SettingsTab::Models
            && self
                .models
                .get(self.cursor)
                .is_some_and(|model| !model.has_key)
    }

    pub fn on_key(&mut self, code: KeyCode) -> SettingsEffect {
        if self.skill_detail.is_some() {
            return self.on_skill_detail_key(code);
        }
        if let Some(field) = self.editing {
            return self.on_text_key(field, code);
        }
        match code {
            KeyCode::Esc | KeyCode::Char('q') => SettingsEffect::Close,
            KeyCode::Left | KeyCode::Char('h') => {
                self.switch_tab(previous_tab(self.tab));
                SettingsEffect::None
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.switch_tab(next_tab(self.tab));
                SettingsEffect::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.cursor = self.cursor.saturating_sub(1);
                SettingsEffect::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let last = self.row_count().saturating_sub(1);
                if self.cursor < last {
                    self.cursor += 1;
                }
                SettingsEffect::None
            }
            KeyCode::Char('s') if self.tab == SettingsTab::Harness => {
                SettingsEffect::SaveHarness(self.harness.clone())
            }
            KeyCode::Char(' ') | KeyCode::Enter => self.activate(),
            _ => SettingsEffect::None,
        }
    }

    fn on_text_key(&mut self, field: TextField, code: KeyCode) -> SettingsEffect {
        match code {
            KeyCode::Esc => {
                self.editing = None;
                if field == TextField::ApiKey {
                    self.key_draft = None;
                }
                SettingsEffect::None
            }
            KeyCode::Enter => match field {
                TextField::ApiKey => {
                    let secret = self.key_draft.take().unwrap_or_default();
                    self.editing = None;
                    let key = self
                        .models
                        .get(self.cursor)
                        .map(|model| model.key.clone())
                        .unwrap_or_default();
                    if secret.trim().is_empty() {
                        self.fail("API key cannot be empty.");
                        SettingsEffect::None
                    } else {
                        SettingsEffect::SetApiKey {
                            key,
                            secret: secret.trim().to_string(),
                        }
                    }
                }
                TextField::SkillRepo | TextField::SkillName => {
                    self.editing = None;
                    SettingsEffect::None
                }
            },
            KeyCode::Backspace => {
                match field {
                    TextField::ApiKey => {
                        if let Some(draft) = self.key_draft.as_mut() {
                            draft.pop();
                        }
                    }
                    TextField::SkillRepo => {
                        self.skill_repo.pop();
                    }
                    TextField::SkillName => {
                        self.skill_name.pop();
                    }
                }
                SettingsEffect::None
            }
            KeyCode::Char(ch) if !ch.is_control() => {
                match field {
                    TextField::ApiKey => {
                        self.key_draft.get_or_insert_with(String::new).push(ch);
                    }
                    TextField::SkillRepo => self.skill_repo.push(ch),
                    TextField::SkillName => self.skill_name.push(ch),
                }
                SettingsEffect::None
            }
            _ => SettingsEffect::None,
        }
    }

    fn on_skill_detail_key(&mut self, code: KeyCode) -> SettingsEffect {
        match code {
            KeyCode::Esc
            | KeyCode::Char('\u{1b}')
            | KeyCode::Char('q')
            | KeyCode::Backspace
            | KeyCode::Left => {
                self.skill_detail = None;
                self.skill_detail_scroll = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.skill_detail_scroll = self.skill_detail_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.skill_detail_scroll = self.skill_detail_scroll.saturating_add(1);
                self.clamp_skill_detail_scroll();
            }
            _ => {}
        }
        SettingsEffect::None
    }

    fn switch_tab(&mut self, tab: SettingsTab) {
        self.tab = tab;
        self.cursor = 0;
        self.skill_detail = None;
        self.skill_detail_scroll = 0;
        self.notice = None;
        self.error = None;
    }

    fn activate(&mut self) -> SettingsEffect {
        match self.tab {
            SettingsTab::Models => self.activate_model(),
            SettingsTab::Harness => {
                self.toggle_harness_row();
                SettingsEffect::None
            }
            SettingsTab::Skills => self.activate_skill(),
        }
    }

    fn activate_model(&mut self) -> SettingsEffect {
        let Some(model) = self.models.get(self.cursor) else {
            return SettingsEffect::None;
        };
        if model.has_key {
            let key = model.key.clone();
            SettingsEffect::SwitchModel { key }
        } else {
            self.key_draft = Some(String::new());
            self.editing = Some(TextField::ApiKey);
            SettingsEffect::None
        }
    }

    fn toggle_harness_row(&mut self) {
        match self.cursor {
            0 => self.harness.restrict_to_workspace = !self.harness.restrict_to_workspace,
            1 => self.harness.shell_mode = next_shell_mode(&self.harness.shell_mode).to_string(),
            2 => self.harness.execution_enabled = !self.harness.execution_enabled,
            3 => self.harness.subagents_enabled = !self.harness.subagents_enabled,
            4 => self.harness.builtin_tools_enabled = !self.harness.builtin_tools_enabled,
            5 => self.harness.ml_engineer_enabled = !self.harness.ml_engineer_enabled,
            _ => {}
        }
    }

    fn activate_skill(&mut self) -> SettingsEffect {
        if !self.skills_available {
            self.fail("This session has no skill registry.");
            return SettingsEffect::None;
        }
        match self.cursor {
            0 => {
                self.editing = Some(TextField::SkillRepo);
                SettingsEffect::None
            }
            1 => {
                self.editing = Some(TextField::SkillName);
                SettingsEffect::None
            }
            2 => self.queue_install(),
            index => {
                let skill_index = index - SKILL_ROWS;
                if skill_index < self.skills.len() {
                    self.skill_detail = Some(skill_index);
                    self.skill_detail_scroll = 0;
                }
                SettingsEffect::None
            }
        }
    }

    fn clamp_skill_detail_scroll(&mut self) {
        let Some(index) = self.skill_detail else {
            return;
        };
        let Some(skill) = self.skills.get(index) else {
            self.skill_detail = None;
            self.skill_detail_scroll = 0;
            return;
        };
        let lines = skill_description_lines(&skill.description, SKILL_DETAIL_WRAP);
        let max_start = lines.len().saturating_sub(SKILL_DETAIL_VISIBLE);
        if self.skill_detail_scroll > max_start {
            self.skill_detail_scroll = max_start;
        }
    }

    fn queue_install(&mut self) -> SettingsEffect {
        let repo = self.skill_repo.trim();
        if repo.is_empty() || repo.chars().any(|ch| ch.is_whitespace() || ch.is_control()) {
            self.fail("Give a GitHub owner/repo shorthand or a repository URL with no spaces.");
            return SettingsEffect::None;
        }
        let skill_name = self.skill_name.trim();
        if skill_name
            .chars()
            .any(|ch| ch.is_whitespace() || ch.is_control())
        {
            self.fail("Skill name must be a single name with no spaces.");
            return SettingsEffect::None;
        }
        self.pending_install = Some(PendingSkillInstall {
            repo: repo.to_string(),
            skill_name: (!skill_name.is_empty()).then(|| skill_name.to_string()),
        });
        self.notice = Some("Installing…".to_string());
        self.error = None;
        SettingsEffect::None
    }

    fn row_count(&self) -> usize {
        match self.tab {
            SettingsTab::Models => self.models.len().max(1),
            SettingsTab::Harness => HARNESS_ROWS,
            SettingsTab::Skills => SKILL_ROWS + self.skills.len(),
        }
    }
}

pub fn models_from_providers(providers: &HashMap<String, ProviderConfig>) -> Vec<SettingsModel> {
    let mut models: Vec<SettingsModel> = providers
        .iter()
        .map(|(key, config)| SettingsModel {
            key: key.clone(),
            provider_name: config.provider_name.clone(),
            model_name: config.model_name.clone(),
            has_key: config.resolve_api_key().is_ok(),
        })
        .collect();
    models.sort_by(|left, right| left.key.cmp(&right.key));
    models
}

fn next_tab(tab: SettingsTab) -> SettingsTab {
    match tab {
        SettingsTab::Models => SettingsTab::Harness,
        SettingsTab::Harness => SettingsTab::Skills,
        SettingsTab::Skills => SettingsTab::Models,
    }
}

fn previous_tab(tab: SettingsTab) -> SettingsTab {
    match tab {
        SettingsTab::Models => SettingsTab::Skills,
        SettingsTab::Harness => SettingsTab::Models,
        SettingsTab::Skills => SettingsTab::Harness,
    }
}

fn next_shell_mode(mode: &str) -> &'static str {
    match mode {
        "deny" => "allow",
        "allow" => "ask",
        _ => "deny",
    }
}

pub fn render(frame: &mut Frame, area: Rect, popup: &SettingsPopup) {
    let popup_w = 72u16.min(area.width).max(1);
    let popup_h = 22u16.min(area.height).max(1);
    if popup_w < 2 || popup_h < 2 {
        return;
    }
    let popup_x = area.x + (area.width.saturating_sub(popup_w)) / 2;
    let popup_y = area.y + (area.height.saturating_sub(popup_h)) / 2;
    let popup_area = Rect::new(popup_x, popup_y, popup_w, popup_h);
    frame.render_widget(Clear, popup_area);

    let title = if popup.editing == Some(TextField::ApiKey) {
        " API key  paste or type  enter saves  esc back "
    } else if popup.skill_detail.is_some() {
        " Skill  ↑↓ read  esc or ← back "
    } else {
        " Settings  ←→ tab  ↑↓ move  enter  esc/q "
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(title, Theme::tool_call()))
        .border_style(Style::default().fg(ratatui::style::Color::Cyan));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    if popup.editing == Some(TextField::ApiKey) {
        frame.render_widget(Paragraph::new(api_key_lines(popup)), inner);
        return;
    }
    if let Some(index) = popup.skill_detail {
        frame.render_widget(
            Paragraph::new(skill_detail_lines(
                popup,
                index,
                inner.width as usize,
                inner.height as usize,
            )),
            inner,
        );
        return;
    }

    let mut lines = if popup.tab == SettingsTab::Skills {
        skill_page_lines(popup, inner.width as usize, inner.height as usize)
    } else {
        settings_page_lines(popup, inner.height as usize)
    };
    let shown = lines.len().min(inner.height as usize);
    frame.render_widget(
        Paragraph::new(lines.drain(..shown).collect::<Vec<_>>()),
        inner,
    );
}

fn settings_page_lines(popup: &SettingsPopup, height: usize) -> Vec<Line<'static>> {
    let mut lines = vec![
        tab_line(popup.tab),
        Line::from(subtitle(popup.tab)),
        Line::from(""),
    ];
    lines.extend(body_lines(popup, height.saturating_sub(4)));
    if let Some(error) = &popup.error {
        lines.push(Line::from(Span::styled(
            error.clone(),
            Style::default().fg(ratatui::style::Color::Red),
        )));
    } else if let Some(notice) = &popup.notice {
        lines.push(Line::from(Span::styled(
            notice.clone(),
            Style::default().fg(ratatui::style::Color::Green),
        )));
    } else {
        lines.push(Line::from(Span::styled(hint(popup.tab), Theme::dim())));
    }
    let shown = lines.len().min(height);
    lines.into_iter().take(shown).collect()
}

fn tab_line(active: SettingsTab) -> Line<'static> {
    let tab = |label: &str, selected: bool| {
        if selected {
            Span::styled(
                format!(" {label} "),
                Style::default()
                    .fg(ratatui::style::Color::Cyan)
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED),
            )
        } else {
            Span::styled(format!(" {label} "), Theme::dim())
        }
    };
    Line::from(vec![
        tab("Models", active == SettingsTab::Models),
        tab("Harness", active == SettingsTab::Harness),
        tab("Skills", active == SettingsTab::Skills),
    ])
}

fn subtitle(tab: SettingsTab) -> &'static str {
    match tab {
        SettingsTab::Models => "Model changes apply to the next message.",
        SettingsTab::Harness => "Saved to config.toml. Applies the next time isanagent starts.",
        SettingsTab::Skills => {
            "Install from a repository. A new skill is ready on the next message."
        }
    }
}

fn hint(tab: SettingsTab) -> &'static str {
    match tab {
        SettingsTab::Models => {
            "Enter opens a key box. Paste with Cmd+V. The full key is never shown."
        }
        SettingsTab::Harness => {
            "Space or enter toggles. s saves. Shell policy cycles ask, deny, allow."
        }
        SettingsTab::Skills => "Enter reads a skill. Esc closes. Repo and name install.",
    }
}

fn body_lines(popup: &SettingsPopup, limit: usize) -> Vec<Line<'static>> {
    let lines = match popup.tab {
        SettingsTab::Models => model_lines(popup),
        SettingsTab::Harness => harness_lines(popup),
        SettingsTab::Skills => Vec::new(),
    };
    let start = popup.cursor.saturating_sub(limit.saturating_sub(1));
    lines.into_iter().skip(start).take(limit.max(1)).collect()
}

fn model_lines(popup: &SettingsPopup) -> Vec<Line<'static>> {
    if popup.models.is_empty() {
        return vec![Line::from(
            "No providers in config.toml. Add a [providers.*] block, then reopen settings.",
        )];
    }
    popup
        .models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let marker = if index == popup.cursor { "▶ " } else { "  " };
            let state = if popup.active_key.as_deref() == Some(model.key.as_str()) {
                "active"
            } else if model.has_key {
                "key ready"
            } else {
                "needs key"
            };
            let style = if index == popup.cursor {
                Style::default()
                    .fg(ratatui::style::Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::from(Span::styled(
                format!(
                    "{marker}{:<28} {:<12} {state}",
                    model.model_name, model.provider_name
                ),
                style,
            ))
        })
        .collect()
}

fn api_key_lines(popup: &SettingsPopup) -> Vec<Line<'static>> {
    let model = popup.models.get(popup.cursor);
    let model_name = model
        .map(|item| item.model_name.as_str())
        .unwrap_or("this model");
    let provider_name = model.map(|item| item.provider_name.as_str()).unwrap_or("");
    let masked = mask_secret(popup.key_draft.as_deref().unwrap_or(""));
    let mut lines = vec![
        Line::from(Span::styled(
            format!("API key for {model_name}"),
            Style::default()
                .fg(ratatui::style::Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(provider_name.to_string()),
        Line::from(""),
        Line::from("Paste the key here with Cmd+V, or type it."),
        Line::from("It is saved in the OS keychain. The full key stays off this screen."),
        Line::from(""),
        Line::from(Span::styled(
            format!("Key  {masked}"),
            Style::default().fg(ratatui::style::Color::Yellow),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Enter saves and uses this model. Esc returns to the list.",
            Theme::dim(),
        )),
    ];
    if let Some(error) = &popup.error {
        lines.push(Line::from(Span::styled(
            error.clone(),
            Style::default().fg(ratatui::style::Color::Red),
        )));
    }
    lines
}

fn sanitize_api_key(raw: &str) -> String {
    raw.chars()
        .filter(|ch| !ch.is_whitespace() && !ch.is_control())
        .collect()
}

fn harness_lines(popup: &SettingsPopup) -> Vec<Line<'static>> {
    let rows = [
        (
            "Sandbox file tools",
            if popup.harness.restrict_to_workspace {
                "on"
            } else {
                "off"
            },
        ),
        ("Shell policy", popup.harness.shell_mode.as_str()),
        (
            "Execution harness",
            if popup.harness.execution_enabled {
                "on"
            } else {
                "off"
            },
        ),
        (
            "Subagents",
            if popup.harness.subagents_enabled {
                "on"
            } else {
                "off"
            },
        ),
        (
            "Builtin tools",
            if popup.harness.builtin_tools_enabled {
                "on"
            } else {
                "off"
            },
        ),
        (
            "ML engineer overlay",
            if popup.harness.ml_engineer_enabled {
                "on"
            } else {
                "off"
            },
        ),
    ];
    rows.into_iter()
        .enumerate()
        .map(|(index, (label, value))| {
            let marker = if index == popup.cursor { "▶ " } else { "  " };
            let style = if index == popup.cursor {
                Style::default()
                    .fg(ratatui::style::Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::from(Span::styled(format!("{marker}{label:<24} {value}"), style))
        })
        .collect()
}

fn skill_page_lines(popup: &SettingsPopup, width: usize, height: usize) -> Vec<Line<'static>> {
    let repo = field_text(
        &popup.skill_repo,
        "owner/repo",
        popup.editing == Some(TextField::SkillRepo),
    );
    let name = field_text(
        &popup.skill_name,
        "skill name, optional",
        popup.editing == Some(TextField::SkillName),
    );
    let mut lines = vec![
        tab_line(popup.tab),
        Line::from(subtitle(popup.tab)),
        Line::from(""),
        marked(0, popup.cursor, format!("Repo   {repo}")),
        marked(1, popup.cursor, format!("Name   {name}")),
        marked(2, popup.cursor, "Install".to_string()),
        Line::from(""),
        Line::from(Span::styled("Installed", Theme::dim())),
    ];
    let footer = footer_line(popup);
    let list_room = height.saturating_sub(lines.len() + 1);
    if !popup.skills_available {
        lines.push(Line::from("This session has no skill registry."));
    } else if popup.skills.is_empty() {
        lines.push(Line::from("No skills installed yet."));
    } else {
        let selected = popup.cursor.checked_sub(SKILL_ROWS);
        let start = selected
            .map(|index| index.saturating_sub(list_room.saturating_sub(1)))
            .unwrap_or(0);
        let end = (start + list_room.max(1)).min(popup.skills.len());
        for (offset, skill) in popup
            .skills
            .iter()
            .enumerate()
            .skip(start)
            .take(end - start)
        {
            let marker = if selected == Some(offset) {
                "▶ "
            } else {
                "  "
            };
            let label = format!("{marker}{}", skill_row_label(skill));
            let style = if selected == Some(offset) {
                Style::default()
                    .fg(ratatui::style::Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(fit_columns(label, width), style)));
        }
    }
    lines.push(footer);
    let shown = lines.len().min(height.max(1));
    lines.into_iter().take(shown).collect()
}

fn skill_detail_lines(
    popup: &SettingsPopup,
    index: usize,
    width: usize,
    height: usize,
) -> Vec<Line<'static>> {
    let Some(skill) = popup.skills.get(index) else {
        return vec![Line::from("That skill is no longer installed.")];
    };
    let mut lines = vec![Line::from(Span::styled(
        skill.name.clone(),
        Style::default()
            .fg(ratatui::style::Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ))];
    if let Some(status) = skill_status(skill) {
        lines.push(Line::from(status));
    }
    lines.push(Line::from(""));
    let footer = 2;
    let room = height.saturating_sub(lines.len() + footer).max(1);
    let wrapped = skill_description_lines(&skill.description, width.saturating_sub(1));
    let start = popup
        .skill_detail_scroll
        .min(wrapped.len().saturating_sub(room));
    for line in wrapped.into_iter().skip(start).take(room) {
        lines.push(Line::from(line));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Esc or ← returns to the list.",
        Theme::dim(),
    )));
    lines
}

fn skill_status(skill: &InstalledSkill) -> Option<&'static str> {
    if !skill.available {
        Some("Unavailable")
    } else if skill.always {
        Some("Always on")
    } else {
        None
    }
}

fn skill_row_label(skill: &InstalledSkill) -> String {
    if !skill.available {
        format!("{}  unavailable", skill.name)
    } else if skill.always {
        format!("{}  always on", skill.name)
    } else {
        skill.name.clone()
    }
}

const SKILL_DETAIL_WRAP: usize = 58;
const SKILL_DETAIL_VISIBLE: usize = 12;

fn skill_description_lines(description: &str, width: usize) -> Vec<String> {
    let text = if description.trim().is_empty() {
        "No description."
    } else {
        description
    };
    wrap_words(text, width.max(8))
}

fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut col = 0usize;
        for word in paragraph.split_whitespace() {
            let word_width = unicode_width::UnicodeWidthStr::width(word);
            if line.is_empty() {
                push_word(&mut lines, &mut line, &mut col, word, word_width, width);
            } else if col + 1 + word_width <= width {
                line.push(' ');
                line.push_str(word);
                col += 1 + word_width;
            } else {
                lines.push(std::mem::take(&mut line));
                col = 0;
                push_word(&mut lines, &mut line, &mut col, word, word_width, width);
            }
        }
        if !line.is_empty() {
            lines.push(line);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn push_word(
    lines: &mut Vec<String>,
    line: &mut String,
    col: &mut usize,
    word: &str,
    word_width: usize,
    width: usize,
) {
    if word_width <= width {
        *line = word.to_string();
        *col = word_width;
        return;
    }
    let mut chunk = String::new();
    let mut chunk_width = 0usize;
    for ch in word.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch)
            .unwrap_or(0)
            .max(1);
        if chunk_width + w > width && !chunk.is_empty() {
            lines.push(std::mem::take(&mut chunk));
            chunk_width = 0;
        }
        chunk.push(ch);
        chunk_width += w;
    }
    *line = chunk;
    *col = chunk_width;
}

fn fit_columns(text: String, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let display = unicode_width::UnicodeWidthStr::width(text.as_str());
    if display <= width {
        return text;
    }
    let mut kept = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        kept.push(ch);
        used += w;
    }
    kept.push('…');
    kept
}

fn footer_line(popup: &SettingsPopup) -> Line<'static> {
    if let Some(error) = &popup.error {
        Line::from(Span::styled(
            error.clone(),
            Style::default().fg(ratatui::style::Color::Red),
        ))
    } else if let Some(notice) = &popup.notice {
        Line::from(Span::styled(
            notice.clone(),
            Style::default().fg(ratatui::style::Color::Green),
        ))
    } else {
        Line::from(Span::styled(hint(popup.tab), Theme::dim()))
    }
}

fn marked(index: usize, cursor: usize, text: String) -> Line<'static> {
    let marker = if index == cursor { "▶ " } else { "  " };
    let style = if index == cursor {
        Style::default()
            .fg(ratatui::style::Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    Line::from(Span::styled(format!("{marker}{text}"), style))
}

fn field_text(value: &str, placeholder: &str, editing: bool) -> String {
    if value.is_empty() && !editing {
        placeholder.to_string()
    } else if editing {
        format!("{value}_")
    } else {
        value.to_string()
    }
}

fn mask_secret(secret: &str) -> String {
    if secret.is_empty() {
        "_".to_string()
    } else if secret.len() <= 4 {
        "*".repeat(secret.len())
    } else {
        format!("…{}", &secret[secret.len() - 4..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harness() -> HarnessSettings {
        HarnessSettings {
            restrict_to_workspace: true,
            shell_mode: "ask".to_string(),
            execution_enabled: false,
            subagents_enabled: true,
            builtin_tools_enabled: true,
            ml_engineer_enabled: false,
        }
    }

    fn provider(name: &str, model: &str) -> ProviderConfig {
        ProviderConfig {
            provider_name: name.to_string(),
            model_name: model.to_string(),
            models: None,
            api_key_env: format!("{}_API_KEY", name.to_uppercase()),
            api_key: None,
            base_url: None,
        }
    }

    fn popup() -> SettingsPopup {
        let mut providers = HashMap::new();
        providers.insert("zeta".to_string(), provider("settings-test-z", "zeta"));
        providers.insert("alpha".to_string(), provider("settings-test-a", "alpha"));
        SettingsPopup::open(&providers, Some("alpha"), harness(), Some(Vec::new()), None)
    }

    #[test]
    fn models_are_sorted_and_a_missing_key_asks_before_it_switches() {
        let mut popup = popup();
        assert_eq!(popup.models[0].key, "alpha");
        assert_eq!(popup.models[1].key, "zeta");
        popup.cursor = 1;
        assert_eq!(popup.on_key(KeyCode::Enter), SettingsEffect::None);
        assert_eq!(popup.editing, Some(TextField::ApiKey));
        assert_eq!(popup.on_key(KeyCode::Char('s')), SettingsEffect::None);
        assert_eq!(popup.on_key(KeyCode::Char('e')), SettingsEffect::None);
        assert_eq!(popup.on_key(KeyCode::Char('c')), SettingsEffect::None);
        assert_eq!(popup.on_key(KeyCode::Char('r')), SettingsEffect::None);
        assert_eq!(popup.on_key(KeyCode::Char('e')), SettingsEffect::None);
        assert_eq!(popup.on_key(KeyCode::Char('t')), SettingsEffect::None);
        match popup.on_key(KeyCode::Enter) {
            SettingsEffect::SetApiKey { key, secret } => {
                assert_eq!(key, "zeta");
                assert_eq!(secret, "secret");
            }
            other => panic!("expected a key, got {other:?}"),
        }
        assert!(popup.key_draft.is_none());
    }

    #[test]
    fn switching_tabs_drops_the_previous_notice() {
        let mut popup = popup();
        popup.note_harness_saved();
        assert!(popup.notice.is_some());
        popup.on_key(KeyCode::Right);
        popup.on_key(KeyCode::Right);
        assert_eq!(popup.tab, SettingsTab::Skills);
        assert!(popup.notice.is_none());
        assert!(popup.error.is_none());
    }

    #[test]
    fn a_saved_key_marks_every_model_from_that_provider() {
        let mut providers = HashMap::new();
        providers.insert(
            "gemini-3.5-flash".to_string(),
            provider("gemini", "gemini-3.5-flash"),
        );
        providers.insert(
            "gemini-3.8-flash".to_string(),
            provider("gemini", "gemini-3.8-flash"),
        );
        providers.insert("gpt-5.5".to_string(), provider("openai", "gpt-5.5"));
        let mut popup = SettingsPopup::open(&providers, None, harness(), Some(Vec::new()), None);
        popup.note_key_saved("gemini-3.5-flash");
        let keyed: Vec<_> = popup
            .models
            .iter()
            .filter(|model| model.has_key)
            .map(|model| model.key.as_str())
            .collect();
        assert_eq!(keyed, vec!["gemini-3.5-flash", "gemini-3.8-flash"]);
    }

    #[test]
    fn pasting_a_key_opens_the_key_box_and_drops_surrounding_whitespace() {
        let mut popup = popup();
        popup.cursor = 1;
        assert_eq!(
            popup.on_paste("\n  pasted-secret-key  \n"),
            SettingsEffect::None
        );
        assert_eq!(popup.editing, Some(TextField::ApiKey));
        assert_eq!(popup.key_draft.as_deref(), Some("pasted-secret-key"));
        match popup.on_key(KeyCode::Enter) {
            SettingsEffect::SetApiKey { key, secret } => {
                assert_eq!(key, "zeta");
                assert_eq!(secret, "pasted-secret-key");
            }
            other => panic!("expected a key, got {other:?}"),
        }
    }

    #[test]
    fn harness_toggles_cycle_and_save_without_typing_the_secret_path() {
        let mut popup = popup();
        assert_eq!(popup.on_key(KeyCode::Right), SettingsEffect::None);
        assert_eq!(popup.tab, SettingsTab::Harness);
        assert_eq!(popup.on_key(KeyCode::Char(' ')), SettingsEffect::None);
        assert!(!popup.harness.restrict_to_workspace);
        assert_eq!(popup.on_key(KeyCode::Down), SettingsEffect::None);
        assert_eq!(popup.on_key(KeyCode::Enter), SettingsEffect::None);
        assert_eq!(popup.harness.shell_mode, "deny");
        match popup.on_key(KeyCode::Char('s')) {
            SettingsEffect::SaveHarness(saved) => {
                assert!(!saved.restrict_to_workspace);
                assert_eq!(saved.shell_mode, "deny");
            }
            other => panic!("expected save, got {other:?}"),
        }
    }

    #[test]
    fn skill_install_waits_for_a_repo_and_hides_nothing_about_the_body() {
        let mut popup = popup();
        popup.on_key(KeyCode::Right);
        popup.on_key(KeyCode::Right);
        assert_eq!(popup.tab, SettingsTab::Skills);
        popup.cursor = 2;
        assert_eq!(popup.on_key(KeyCode::Enter), SettingsEffect::None);
        assert!(popup.error.is_some());
        assert!(popup.take_pending_install().is_none());

        popup.cursor = 0;
        popup.on_key(KeyCode::Enter);
        for ch in "owner/repo".chars() {
            popup.on_key(KeyCode::Char(ch));
        }
        popup.on_key(KeyCode::Enter);
        popup.cursor = 2;
        popup.on_key(KeyCode::Enter);
        let pending = popup.take_pending_install().expect("queued");
        assert_eq!(pending.repo, "owner/repo");
        assert_eq!(pending.skill_name, None);
        assert_eq!(popup.notice.as_deref(), Some("Installing…"));
    }

    #[test]
    fn escape_closes_and_a_ready_model_switches() {
        let mut ready = popup();
        ready.models[0].has_key = true;
        match ready.on_key(KeyCode::Enter) {
            SettingsEffect::SwitchModel { key } => assert_eq!(key, "alpha"),
            other => panic!("expected switch, got {other:?}"),
        }
        let mut closed = popup();
        assert_eq!(closed.on_key(KeyCode::Esc), SettingsEffect::Close);
    }

    #[test]
    fn skill_list_scrolls_under_the_form_and_enter_opens_the_description() {
        let mut popup = popup();
        popup.skills = (0..8)
            .map(|index| InstalledSkill {
                name: format!("skill-{index}"),
                description: if index == 7 {
                    "UNIQUE_DETAIL_SENTENCE that stays off the list and wraps ".repeat(40)
                } else if index == 0 {
                    "Use Afterimage for a dataset.".to_string()
                } else {
                    format!("description {index}")
                },
                available: index != 1,
                always: index == 2,
            })
            .collect();
        popup.on_key(KeyCode::Right);
        popup.on_key(KeyCode::Right);
        for _ in 0..10 {
            popup.on_key(KeyCode::Down);
        }
        assert_eq!(popup.cursor, 10);
        assert_eq!(skill_row_label(&popup.skills[0]), "skill-0");
        assert_eq!(skill_row_label(&popup.skills[1]), "skill-1  unavailable");
        assert_eq!(skill_row_label(&popup.skills[2]), "skill-2  always on");

        let page = skill_page_lines(&popup, 60, 12);
        let text = page
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Repo"), "{text}");
        assert!(text.contains("Install"), "{text}");
        assert!(text.contains("skill-7"), "{text}");
        assert!(!text.contains("skill-0"), "{text}");
        assert!(!text.contains("UNIQUE_DETAIL_SENTENCE"), "{text}");

        assert_eq!(popup.on_key(KeyCode::Enter), SettingsEffect::None);
        assert_eq!(popup.skill_detail, Some(7));
        let plain = skill_detail_lines(&popup, 0, 40, 16)
            .into_iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(plain.contains("Afterimage"), "{plain}");
        assert!(!plain.contains("Available"), "{plain}");

        let detail = skill_detail_lines(&popup, 7, 40, 16)
            .into_iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(detail.contains("skill-7"), "{detail}");
        assert!(detail.contains("UNIQUE_DETAIL_SENTENCE"), "{detail}");
        assert_eq!(popup.on_key(KeyCode::Down), SettingsEffect::None);
        assert!(popup.skill_detail_scroll > 0);
        assert_eq!(popup.on_key(KeyCode::Left), SettingsEffect::None);
        assert!(popup.skill_detail.is_none());
        popup.cursor = 10;
        assert_eq!(popup.on_key(KeyCode::Enter), SettingsEffect::None);
        assert!(popup.showing_skill_detail());
        assert_eq!(popup.on_key(KeyCode::Esc), SettingsEffect::None);
        assert!(!popup.showing_skill_detail());
        assert_eq!(popup.on_key(KeyCode::Esc), SettingsEffect::Close);
    }

    #[test]
    fn a_short_skill_description_does_not_scroll_and_is_not_a_switch() {
        let mut popup = popup();
        popup.skills = vec![InstalledSkill {
            name: "plain".into(),
            description: "Use Afterimage for a dataset.".into(),
            available: true,
            always: false,
        }];
        popup.tab = SettingsTab::Skills;
        popup.cursor = SKILL_ROWS;
        assert_eq!(popup.on_key(KeyCode::Enter), SettingsEffect::None);
        assert_eq!(popup.on_key(KeyCode::Down), SettingsEffect::None);
        assert_eq!(popup.skill_detail_scroll, 0);
        let text = skill_detail_lines(&popup, 0, 40, 16)
            .into_iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Afterimage"), "{text}");
        assert!(!text.contains("Available"), "{text}");
    }
}
