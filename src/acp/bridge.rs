use crate::acp::types::{AcpContentBlock, AcpToolKind};
use crate::utils::ContentPart;

/// Classify an isanagent tool name into an ACP ToolKind.
pub fn classify_tool_kind(tool_name: &str) -> AcpToolKind {
    match tool_name {
        "read_file" | "list_dir" | "get_env" | "git_worktree" => AcpToolKind::Read,
        "write_file" | "edit_file" => AcpToolKind::Edit,
        "search_text" | "glob_files" | "tool_search" | "arxiv_search" => AcpToolKind::Search,
        "web_search" | "web_fetch" | "arxiv_fetch" | "hf_hub_file_fetch" => AcpToolKind::Fetch,
        "shell_exec" | "python_run" | "execution_run" | "execution_run_background" => {
            AcpToolKind::Execute
        }
        "todo_write" | "ask_user" | "compact_context" => AcpToolKind::Think,
        _ => AcpToolKind::Other,
    }
}

/// Convert a list of ACP content blocks into standard isanagent message content and attachments.
pub fn parse_acp_content_blocks(blocks: &[AcpContentBlock]) -> (String, Vec<ContentPart>) {
    let mut text_parts = Vec::new();
    let mut attachments = Vec::new();

    for block in blocks {
        match block {
            AcpContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            AcpContentBlock::ResourceLink { uri, title } => {
                let label = title.as_deref().unwrap_or(uri.as_str());
                text_parts.push(format!("[{label}]({uri})"));
            }
            AcpContentBlock::Resource { resource } => {
                if let Some(content_text) = &resource.text {
                    text_parts.push(format!(
                        "--- Resource ({}) ---\n{}",
                        resource.uri, content_text
                    ));
                } else {
                    text_parts.push(format!("[Resource]({})", resource.uri));
                }
            }
            AcpContentBlock::Image { data, mime_type } => {
                let url = format!("data:{mime_type};base64,{data}");
                attachments.push(ContentPart::ImageUrl {
                    image_url: crate::utils::ImageUrl { url, detail: None },
                });
            }
            AcpContentBlock::Audio { data, mime_type } => {
                text_parts.push(format!(
                    "[Audio attachment: {} ({})]",
                    mime_type,
                    data.len()
                ));
            }
        }
    }

    (text_parts.join("\n\n"), attachments)
}
