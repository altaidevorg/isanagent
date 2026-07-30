//! Interactive approval prompts against the controlling terminal (`/dev/tty` / `CONIN$`).

use std::io::{self, BufRead, Write};

/// Open the process-controlling terminal for interactive prompts even when stdin/stdout are pipes.
pub fn open_tty() -> io::Result<(Box<dyn Write + Send>, Box<dyn BufRead + Send>)> {
    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        let writer = OpenOptions::new().write(true).open("/dev/tty")?;
        let reader = OpenOptions::new().read(true).open("/dev/tty")?;
        Ok((
            Box::new(io::BufWriter::new(writer)),
            Box::new(io::BufReader::new(reader)),
        ))
    }
    #[cfg(windows)]
    {
        use std::fs::OpenOptions;
        let writer = OpenOptions::new().write(true).open("CONOUT$")?;
        let reader = OpenOptions::new().read(true).open("CONIN$")?;
        Ok((
            Box::new(io::BufWriter::new(writer)),
            Box::new(io::BufReader::new(reader)),
        ))
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no controlling TTY API on this platform",
        ))
    }
}

/// Returns true when a controlling TTY can be opened for interactive approval.
pub fn tty_available() -> bool {
    open_tty().is_ok()
}

/// Print `prompt` to the controlling TTY and read one line of reply.
pub fn prompt_on_tty(prompt: &str) -> io::Result<String> {
    let (mut writer, mut reader) = open_tty()?;
    write!(writer, "{prompt}")?;
    writer.flush()?;
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tty_probe_does_not_panic() {
        // In CI / sandboxes /dev/tty may be absent; just ensure the probe is safe.
        let _ = tty_available();
    }
}
