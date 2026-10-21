//! Windows shell command-line construction for `cmd.exe`, Windows PowerShell and `pwsh`.
//!
//! Passing a shell command through `Command::arg` on Windows quotes it with the MSVC C-runtime
//! rules (an inner `"` becomes `\"`). `cmd.exe` does not understand that escaping and Windows
//! PowerShell 5.1 re-parses it inconsistently, so commands that contain double quotes break.
//!
//! This module builds the invocation without that quoting:
//! - `cmd`: `cmd /D /S /C "<command>"`, with the quoted command appended verbatim (`raw_arg`).
//!   `/S` makes cmd strip exactly the outer quote pair and run the rest as typed, so inner
//!   quotes, `&&`, `|`, `^` and `%VAR%` behave as they would at a prompt. `/D` skips the
//!   registry `AutoRun` commands so a user's interactive cmd setup cannot alter or break runs.
//! - PowerShell / pwsh: `-NoProfile -NonInteractive -OutputFormat Text -EncodedCommand <b64>`,
//!   where `<b64>` is base64 of the UTF-16LE command text, so no quoting is involved at all.
//!   `-OutputFormat Text` is explicit because PowerShell otherwise writes the error stream as
//!   CLIXML when the command was encoded and stderr is redirected (it assumes a PowerShell parent).
//!
//! The argument building is platform independent (and unit tested on every platform); only
//! [`WindowsShellInvocation::to_tokio_command`] differs on Windows, where raw arguments are
//! appended with `raw_arg`.

use base64::Engine as _;

use crate::config::WindowsShellRunner;

/// Longest base64 payload passed to `-EncodedCommand` (Windows caps a whole command line at
/// 32,767 UTF-16 units). Longer commands fall back to `-Command` with standard argument quoting.
pub const POWERSHELL_ENCODED_COMMAND_MAX_CHARS: usize = 30_000;

/// One argument of a Windows process command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WinArg {
    /// Quoted by the standard library with the MSVC C-runtime rules (safe for plain tokens).
    Plain(String),
    /// Appended to the command line exactly as written (Windows `raw_arg`).
    Raw(String),
}

/// Program plus arguments for running one shell command on Windows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsShellInvocation {
    pub program: &'static str,
    pub args: Vec<WinArg>,
}

impl WindowsShellInvocation {
    /// Map the invocation onto a tokio command. On Windows, [`WinArg::Raw`] uses `raw_arg`; other
    /// platforms have no raw command line, so the text is passed as a normal argument there (only
    /// meaningful for tests and diagnostics).
    pub fn to_tokio_command(&self) -> tokio::process::Command {
        let mut command = tokio::process::Command::new(self.program);
        for arg in &self.args {
            match arg {
                WinArg::Plain(value) => {
                    command.arg(value);
                }
                WinArg::Raw(value) => {
                    #[cfg(windows)]
                    {
                        command.raw_arg(value);
                    }
                    #[cfg(not(windows))]
                    {
                        command.arg(value);
                    }
                }
            }
        }
        command
    }
}

/// Build the Windows invocation that runs `command` under `runner`.
pub fn windows_shell_invocation(
    runner: WindowsShellRunner,
    command: &str,
) -> WindowsShellInvocation {
    match runner {
        WindowsShellRunner::Cmd => WindowsShellInvocation {
            program: "cmd",
            args: vec![
                WinArg::Plain("/D".to_string()),
                WinArg::Plain("/S".to_string()),
                WinArg::Plain("/C".to_string()),
                WinArg::Raw(format!("\"{command}\"")),
            ],
        },
        WindowsShellRunner::PowerShell => powershell_invocation("powershell", command),
        WindowsShellRunner::Pwsh => powershell_invocation("pwsh", command),
    }
}

fn powershell_invocation(program: &'static str, command: &str) -> WindowsShellInvocation {
    let mut args = vec![
        WinArg::Plain("-NoProfile".to_string()),
        WinArg::Plain("-NonInteractive".to_string()),
        WinArg::Plain("-OutputFormat".to_string()),
        WinArg::Plain("Text".to_string()),
    ];
    let encoded = encode_powershell_command(command);
    if encoded.len() <= POWERSHELL_ENCODED_COMMAND_MAX_CHARS {
        args.push(WinArg::Plain("-EncodedCommand".to_string()));
        args.push(WinArg::Plain(encoded));
    } else {
        // Too long to encode within the Windows command-line limit: keep the previous behavior.
        args.push(WinArg::Plain("-Command".to_string()));
        args.push(WinArg::Plain(command.to_string()));
    }
    WindowsShellInvocation { program, args }
}

/// Base64 of the UTF-16LE bytes of `command`, the format PowerShell's `-EncodedCommand` expects.
pub fn encode_powershell_command(command: &str) -> String {
    let bytes: Vec<u8> = command
        .encode_utf16()
        .flat_map(|unit| unit.to_le_bytes())
        .collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_powershell_command(encoded: &str) -> String {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap_or_default();
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        String::from_utf16(&units).unwrap_or_default()
    }

    fn cmd_raw(command: &str) -> String {
        let invocation = windows_shell_invocation(WindowsShellRunner::Cmd, command);
        assert_eq!(invocation.program, "cmd");
        assert_eq!(
            invocation.args[..3],
            [
                WinArg::Plain("/D".to_string()),
                WinArg::Plain("/S".to_string()),
                WinArg::Plain("/C".to_string()),
            ]
        );
        assert_eq!(invocation.args.len(), 4);
        match &invocation.args[3] {
            WinArg::Raw(raw) => raw.clone(),
            other => panic!("expected raw command argument, got {other:?}"),
        }
    }

    const SAMPLES: &[&str] = &[
        r#"echo "hello world""#,
        r#"findstr /C:"a b" "C:\Program Files\x.txt""#,
        r#"echo """#,
        r#"echo "" & echo """"#,
        "echo %PATH%",
        "cd src && cargo build",
        "dir | findstr foo",
        "echo a^&b",
        "echo héllo 世界 🚀",
        "echo one\necho two",
        r#"Write-Output "a`"b" 'single' $env:PATH"#,
        r"trailing backslash\",
    ];

    #[test]
    fn cmd_wraps_command_verbatim_in_one_quote_pair() {
        for sample in SAMPLES {
            assert_eq!(cmd_raw(sample), format!("\"{sample}\""));
        }
        assert_eq!(cmd_raw(r#"echo "hello world""#), r#""echo "hello world"""#);
        assert_eq!(
            cmd_raw(r#""C:\Program Files\app.exe" "arg one""#),
            r#"""C:\Program Files\app.exe" "arg one"""#
        );
        assert_eq!(
            cmd_raw("echo %PATH% && dir | more"),
            "\"echo %PATH% && dir | more\""
        );
    }

    #[test]
    fn powershell_encoded_command_round_trips() {
        for runner in [WindowsShellRunner::PowerShell, WindowsShellRunner::Pwsh] {
            for sample in SAMPLES {
                let invocation = windows_shell_invocation(runner, sample);
                assert_eq!(invocation.program, runner.as_str());
                assert_eq!(invocation.args.len(), 6);
                assert_eq!(
                    invocation.args[..5],
                    [
                        WinArg::Plain("-NoProfile".to_string()),
                        WinArg::Plain("-NonInteractive".to_string()),
                        WinArg::Plain("-OutputFormat".to_string()),
                        WinArg::Plain("Text".to_string()),
                        WinArg::Plain("-EncodedCommand".to_string()),
                    ]
                );
                let WinArg::Plain(encoded) = &invocation.args[5] else {
                    panic!("expected plain encoded argument");
                };
                // Base64 alphabet only: nothing for the C-runtime quoting to touch.
                assert!(encoded
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=')));
                assert_eq!(decode_powershell_command(encoded), *sample);
            }
        }
    }

    #[test]
    fn powershell_encoding_matches_reference_value() {
        // [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes('echo "hi"'))
        assert_eq!(
            encode_powershell_command(r#"echo "hi""#),
            "ZQBjAGgAbwAgACIAaABpACIA"
        );
    }

    #[test]
    fn overlong_powershell_command_falls_back_to_plain_command() {
        let long = "a".repeat(POWERSHELL_ENCODED_COMMAND_MAX_CHARS);
        let invocation = windows_shell_invocation(WindowsShellRunner::PowerShell, &long);
        assert_eq!(
            invocation.args[4..],
            [
                WinArg::Plain("-Command".to_string()),
                WinArg::Plain(long.clone()),
            ]
        );
    }
}
