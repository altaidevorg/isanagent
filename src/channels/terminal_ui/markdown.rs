//! CommonMark → [`ratatui::text::Line`] for the transcript (`Line` / [`Span`] are Ratatui’s styled text model).
//!
//! There is no built-in Markdown widget; pulldown-cmark drives a small renderer: inline styles,
//! headings (bold only; no ATX `#` markers in output), links (URL appended), lists, code — then width-aware wrap.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::Theme;

const MAX_MARKDOWN_CHARS: usize = 80_000;

#[derive(Debug, Clone, Copy)]
enum Inline {
    Strong,
    Emphasis,
    Strikethrough,
}

fn style_from_inline(stack: &[Inline]) -> ratatui::style::Style {
    let mut s = Theme::text();
    for k in stack {
        match k {
            Inline::Strong => s = s.add_modifier(Modifier::BOLD),
            Inline::Emphasis => s = s.add_modifier(Modifier::ITALIC),
            Inline::Strikethrough => s = s.add_modifier(Modifier::CROSSED_OUT),
        }
    }
    s
}

/// Merge adjacent runs with identical style so wrapping can keep spans coarse.
fn push_run(
    runs: &mut Vec<(ratatui::style::Style, String)>,
    style: ratatui::style::Style,
    s: &str,
) {
    if s.is_empty() {
        return;
    }
    if let Some((st, buf)) = runs.last_mut() {
        if *st == style {
            buf.push_str(s);
            return;
        }
    }
    runs.push((style, s.to_string()));
}

/// Convert assistant markdown to styled, width-wrapped lines for Ratatui.
pub fn assistant_markdown_lines(markdown: &str, width: usize) -> Vec<Line<'static>> {
    let w = width.max(12);
    let md = truncate_chars(markdown, MAX_MARKDOWN_CHARS);
    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS;
    let parser = Parser::new_ext(md, opts);

    let mut runs: Vec<(ratatui::style::Style, String)> = Vec::new();
    let mut inline_stack: Vec<Inline> = Vec::new();
    let mut in_code_block = false;
    let mut in_blockquote = false;
    // Some(level) from Start(Heading) until End(Heading); body text is bold (no `#` in output).
    let mut heading: Option<pulldown_cmark::HeadingLevel> = None;
    let mut list_item_first_text = false;
    let mut link_dest: Option<String> = None;

    for ev in parser {
        match ev {
            Event::Start(Tag::Paragraph) => {}
            Event::End(TagEnd::Paragraph) => {
                push_run(&mut runs, Theme::text(), "\n");
            }
            Event::Start(Tag::Heading { level, .. }) => {
                heading = Some(level);
            }
            Event::End(TagEnd::Heading(_)) => {
                heading = None;
                push_run(&mut runs, Theme::text(), "\n");
            }
            Event::Start(Tag::BlockQuote(_)) => in_blockquote = true,
            Event::End(TagEnd::BlockQuote(_)) => {
                in_blockquote = false;
                push_run(&mut runs, Theme::dim(), "\n");
            }
            Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
            Event::End(TagEnd::CodeBlock) => {
                in_code_block = false;
                push_run(&mut runs, Theme::dim(), "\n");
            }
            Event::Start(Tag::Item) => list_item_first_text = true,
            Event::End(TagEnd::Item) => list_item_first_text = false,
            Event::Start(Tag::List(_)) | Event::End(TagEnd::List(_)) => {}
            Event::Start(Tag::Strong) => inline_stack.push(Inline::Strong),
            Event::End(TagEnd::Strong) => {
                let _ = inline_stack.pop();
            }
            Event::Start(Tag::Emphasis) => inline_stack.push(Inline::Emphasis),
            Event::End(TagEnd::Emphasis) => {
                let _ = inline_stack.pop();
            }
            Event::Start(Tag::Strikethrough) => inline_stack.push(Inline::Strikethrough),
            Event::End(TagEnd::Strikethrough) => {
                let _ = inline_stack.pop();
            }
            Event::Start(Tag::Link { dest_url, .. }) => {
                link_dest = Some(dest_url.to_string());
            }
            Event::End(TagEnd::Link) => {
                if let Some(url) = link_dest.take() {
                    push_run(&mut runs, Theme::dim(), &format!(" ({})", url));
                }
            }
            Event::Start(Tag::Image { dest_url, .. }) => {
                push_run(&mut runs, Theme::dim(), &format!(" [image: {}] ", dest_url));
            }
            Event::End(TagEnd::Image) => {}
            Event::Text(t) => {
                let mut sty = if in_code_block {
                    Theme::dim()
                } else if in_blockquote {
                    Theme::dim().add_modifier(Modifier::ITALIC)
                } else {
                    style_from_inline(&inline_stack)
                };

                if link_dest.is_some() {
                    sty = sty.add_modifier(Modifier::ITALIC);
                }

                if heading.is_some() {
                    sty = sty.add_modifier(Modifier::BOLD);
                }

                if list_item_first_text {
                    push_run(&mut runs, Theme::assistant_bullet(), "• ");
                    list_item_first_text = false;
                }

                push_run(&mut runs, sty, &t);
            }
            Event::Code(t) => {
                let mut sty = Theme::dim();
                if heading.is_some() {
                    sty = sty.add_modifier(Modifier::BOLD);
                }
                push_run(&mut runs, sty.add_modifier(Modifier::BOLD), "`");
                push_run(&mut runs, sty, &t);
                push_run(&mut runs, sty.add_modifier(Modifier::BOLD), "`");
            }
            Event::InlineMath(t) | Event::DisplayMath(t) => {
                push_run(&mut runs, Theme::thinking(), &t);
            }
            Event::Html(t) | Event::InlineHtml(t) => {
                push_run(&mut runs, Theme::dim(), &t);
            }
            Event::SoftBreak => {
                let sty = if heading.is_some() {
                    Theme::text().add_modifier(Modifier::BOLD)
                } else {
                    style_from_inline(&inline_stack)
                };
                push_run(&mut runs, sty, " ");
            }
            Event::HardBreak => {
                push_run(&mut runs, Theme::text(), "\n");
            }
            Event::Rule => {
                let line = "─".repeat(w.min(48));
                push_run(&mut runs, Theme::dim(), &format!("{line}\n"));
            }
            Event::TaskListMarker(done) => {
                let mark = if done { "[x] " } else { "[ ] " };
                push_run(&mut runs, Theme::assistant_bullet(), mark);
            }
            Event::FootnoteReference(l) => {
                push_run(&mut runs, Theme::dim(), &format!(" [^{}] ", l));
            }
            Event::Start(Tag::Table(_) | Tag::TableHead | Tag::TableRow | Tag::TableCell) => {}
            Event::End(TagEnd::TableCell) => {
                push_run(&mut runs, Theme::dim(), " │ ");
            }
            Event::End(TagEnd::Table | TagEnd::TableHead | TagEnd::TableRow) => {}
            Event::Start(
                Tag::MetadataBlock(_)
                | Tag::DefinitionList
                | Tag::DefinitionListTitle
                | Tag::DefinitionListDefinition,
            )
            | Event::End(
                TagEnd::MetadataBlock(_)
                | TagEnd::DefinitionList
                | TagEnd::DefinitionListTitle
                | TagEnd::DefinitionListDefinition,
            ) => {}
            Event::Start(Tag::FootnoteDefinition(_)) | Event::End(TagEnd::FootnoteDefinition) => {}
            Event::Start(Tag::HtmlBlock) | Event::End(TagEnd::HtmlBlock) => {}
        }
    }

    wrap_runs_to_lines(runs, w)
}

fn truncate_chars(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let take = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..take]
}

/// Word-wrap (with character fallback) preserving per-span styles.
fn wrap_runs_to_lines(
    runs: Vec<(ratatui::style::Style, String)>,
    width: usize,
) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut cur: Vec<Span<'static>> = Vec::new();
    let mut col = 0usize;

    let flush_line =
        |cur: &mut Vec<Span<'static>>, col: &mut usize, out: &mut Vec<Line<'static>>| {
            if !cur.is_empty() {
                out.push(Line::from(std::mem::take(cur)));
                *col = 0;
            }
        };

    for (style, text) in runs {
        for (i, segment) in text.split('\n').enumerate() {
            if i > 0 {
                flush_line(&mut cur, &mut col, &mut out);
            }
            if segment.is_empty() {
                continue;
            }
            let mut first_word = col == 0;
            for word in segment.split_whitespace() {
                let pw = word.width();
                if pw == 0 {
                    continue;
                }
                if pw > width {
                    for ch in word.chars() {
                        let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
                        if col + cw > width && !cur.is_empty() {
                            flush_line(&mut cur, &mut col, &mut out);
                        }
                        cur.push(Span::styled(ch.to_string(), style));
                        col += cw;
                    }
                    first_word = false;
                    continue;
                }
                let token = if first_word {
                    first_word = false;
                    word.to_string()
                } else {
                    format!(" {}", word)
                };
                let tw = token.as_str().width();
                if col + tw > width {
                    flush_line(&mut cur, &mut col, &mut out);
                    let token = word.to_string();
                    let tw = token.as_str().width();
                    cur.push(Span::styled(token, style));
                    col = tw;
                    first_word = false;
                } else {
                    cur.push(Span::styled(token, style));
                    col += tw;
                }
            }
        }
    }
    flush_line(&mut cur, &mut col, &mut out);
    if out.is_empty() {
        out.push(Line::from(Span::styled("", Theme::text())));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::assistant_markdown_lines;
    use ratatui::text::Line;

    fn flat(lines: &[Line]) -> String {
        lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect()
    }

    #[test]
    fn bold_list_and_code() {
        let lines = assistant_markdown_lines("**hi** and `code`\n\n- one", 40);
        let t = flat(&lines);
        assert!(t.contains("hi"), "{t}");
        assert!(t.contains("code"), "{t}");
        assert!(t.contains('•'), "{t}");
    }

    #[test]
    fn heading_keeps_bold_across_spans() {
        // ATX `#` is not shown; heading body is still bold across inline spans.
        let lines = assistant_markdown_lines("# Title **bold** end\n\npara", 50);
        let t = flat(&lines);
        assert!(!t.contains("# Title"), "should not echo ATX markers: {t}");
        assert!(t.contains("Title"), "{t}");
        assert!(t.contains("bold"), "{t}");
        assert!(t.contains("end"), "{t}");
    }

    #[test]
    fn heading_h3_no_hashes() {
        let lines = assistant_markdown_lines("### 1. The Rise of Autonomous Systems\n\nNext.", 72);
        let t = flat(&lines);
        assert!(!t.contains("###"), "{t}");
        assert!(t.contains("The Rise of Autonomous Systems"), "{t}");
    }

    #[test]
    fn link_appends_url() {
        let lines = assistant_markdown_lines("[click](https://a.example/foo)", 60);
        let t = flat(&lines);
        assert!(t.contains("click"), "{t}");
        assert!(t.contains("a.example"), "{t}");
    }

    #[test]
    fn wrap_preserves_segments() {
        let md = "**AA** **BB**";
        let lines = assistant_markdown_lines(md, 8);
        assert!(
            !lines.is_empty(),
            "expected wrapped lines, got {:?}",
            lines.len()
        );
    }
}
