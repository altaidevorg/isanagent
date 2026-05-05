//! Submodules for Ratatui focus panes (conversations, executions, etc.).

mod cell_render;
mod conversations;
mod executions;
mod tool_history;
mod transcript;

pub use cell_render::wrap_text;
pub use conversations::{conversations_ensure_list_shows_selection, conversations_list_paragraph};
pub use executions::{
    executions_code_paragraph, executions_ensure_list_shows_selection, executions_list_paragraph,
    executions_output_paragraph,
};
pub use tool_history::tool_history_paragraph;
pub use transcript::{extract_selection_text, transcript_paragraph};
