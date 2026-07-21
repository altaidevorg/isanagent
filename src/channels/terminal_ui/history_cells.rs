//! Map persisted [`ChatMessage`] rows into [`Cell`]s for the Ratatui transcript (session resume).

use std::sync::OnceLock;

use regex::Regex;

use crate::utils::{
    ChatMessage, ContentPart, MessageContent, REDACTED_THINKING_STRIP_PATTERN,
    RUNTIME_CONTEXT_END_SUFFIX,
};

use super::app::Cell;

fn redacted_thinking_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(REDACTED_THINKING_STRIP_PATTERN).expect("thinking strip regex"))
}

fn strip_runtime_context_prefix(text: &str) -> String {
    if let Some(idx) = text.find(RUNTIME_CONTEXT_END_SUFFIX) {
        return text[idx + RUNTIME_CONTEXT_END_SUFFIX.len()..]
            .trim_start()
            .to_string();
    }
    text.to_string()
}

fn strip_model_thinking_markup(text: &str) -> String {
    redacted_thinking_re()
        .replace_all(text, "")
        .trim()
        .to_string()
}

fn text_and_images_from_message(message: &ChatMessage) -> (String, Vec<String>) {
    match &message.content {
        Some(MessageContent::Parts(parts)) => {
            let mut texts = Vec::new();
            let mut urls = Vec::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => texts.push(text.as_str()),
                    ContentPart::ImageUrl { image_url } => urls.push(image_url.url.clone()),
                    ContentPart::Document { .. } => texts.push("[document attachment]"),
                }
            }
            (texts.join("\n\n"), urls)
        }
        Some(MessageContent::Text(s)) => (s.clone(), Vec::new()),
        None => (String::new(), Vec::new()),
    }
}

/// Transcript cells for the terminal: same visibility rules as the web history API (no tool/system rows).
pub fn chat_messages_to_terminal_cells(messages: &[ChatMessage]) -> Vec<Cell> {
    let mut out = Vec::new();
    for message in messages {
        if message.role == "tool" || message.role == "system" {
            continue;
        }

        let (mut text, image_urls) = text_and_images_from_message(message);

        if message.role == "user" {
            text = strip_runtime_context_prefix(&text);
        } else if message.role == "assistant" {
            text = strip_model_thinking_markup(&text);
            let visible = text.trim();
            if visible.is_empty() && message.tool_calls.is_some() && image_urls.is_empty() {
                continue;
            }
        }

        let text = text.trim().to_string();
        if text.is_empty() && image_urls.is_empty() {
            continue;
        }

        if message.role == "user" {
            let label = if image_urls.is_empty() {
                text
            } else {
                let mut s = text;
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&format!("({} image(s))", image_urls.len()));
                s
            };
            out.push(Cell::User { text: label });
        } else if message.role == "assistant" {
            out.push(Cell::Assistant { markdown: text });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::ChatMessage;

    #[test]
    fn skips_tool_rows() {
        let messages = vec![
            ChatMessage::user("hi"),
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text("x".into())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                is_error: None,
            },
        ];
        let cells = chat_messages_to_terminal_cells(&messages);
        assert_eq!(cells.len(), 1);
    }
}
