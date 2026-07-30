pub mod bridge;
pub mod types;

#[cfg(test)]
mod tests {
    use super::bridge::{classify_tool_kind, parse_acp_content_blocks};
    use super::types::{AcpContentBlock, AcpToolKind};

    #[test]
    fn test_classify_tool_kind() {
        assert_eq!(classify_tool_kind("read_file"), AcpToolKind::Read);
        assert_eq!(classify_tool_kind("edit_file"), AcpToolKind::Edit);
        assert_eq!(classify_tool_kind("shell_exec"), AcpToolKind::Execute);
        assert_eq!(classify_tool_kind("search_text"), AcpToolKind::Search);
        assert_eq!(classify_tool_kind("web_search"), AcpToolKind::Fetch);
        assert_eq!(classify_tool_kind("unknown_tool"), AcpToolKind::Other);
    }

    #[test]
    fn test_parse_acp_content_blocks() {
        let blocks = vec![
            AcpContentBlock::Text {
                text: "Hello ACP".to_string(),
            },
            AcpContentBlock::ResourceLink {
                uri: "file:///foo.txt".to_string(),
                title: Some("foo".to_string()),
            },
            AcpContentBlock::Image {
                data: "base64data".to_string(),
                mime_type: "image/png".to_string(),
            },
        ];

        let (text, attachments) = parse_acp_content_blocks(&blocks);
        assert!(text.contains("Hello ACP"));
        assert!(text.contains("[foo](file:///foo.txt)"));
        assert_eq!(attachments.len(), 1);
    }
}
