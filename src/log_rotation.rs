use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Result of attempting to append one complete line record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteLineOutcome {
    Written,
    /// The caller must replace the record with a smaller, format-aware summary.
    /// The writer never splits a record or writes partial UTF-8.
    RecordTooLarge {
        record_bytes: u64,
        max_record_bytes: u64,
    },
}

/// A single-writer, size-bounded append-only file with numbered rotations.
///
/// Rotation is intentionally synchronous: callers such as LoggingActor already
/// serialize writes, so this avoids interleaved records and keeps recovery simple.
pub struct RotatingLineWriter {
    path: PathBuf,
    max_bytes: u64,
    retained_generations: usize,
    current_bytes: u64,
    writer: BufWriter<File>,
}

impl RotatingLineWriter {
    pub fn new(
        path: impl Into<PathBuf>,
        max_bytes: u64,
        retained_generations: usize,
    ) -> io::Result<Self> {
        if max_bytes < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "rotating line writer requires at least two bytes",
            ));
        }

        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let existing_bytes = file_len(&path)?;
        if existing_bytes > max_bytes {
            rotate_files(&path, retained_generations)?;
        }

        let current_bytes = file_len(&path)?;
        let writer = open_append_writer(&path)?;
        Ok(Self {
            path,
            max_bytes,
            retained_generations,
            current_bytes,
            writer,
        })
    }

    pub fn write_line(&mut self, record: &str) -> io::Result<WriteLineOutcome> {
        let record_bytes = record.len() as u64;
        let line_bytes = record_bytes.saturating_add(1);
        if line_bytes > self.max_bytes {
            return Ok(WriteLineOutcome::RecordTooLarge {
                record_bytes,
                max_record_bytes: self.max_bytes.saturating_sub(1),
            });
        }

        if self.current_bytes > 0 && self.current_bytes.saturating_add(line_bytes) > self.max_bytes
        {
            self.rotate()?;
        }

        self.writer.write_all(record.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.current_bytes = self.current_bytes.saturating_add(line_bytes);
        Ok(WriteLineOutcome::Written)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        rotate_files(&self.path, self.retained_generations)?;
        self.writer = open_append_writer(&self.path)?;
        self.current_bytes = 0;
        Ok(())
    }
}

fn open_append_writer(path: &Path) -> io::Result<BufWriter<File>> {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }
    options.open(path).map(BufWriter::new)
}

fn file_len(path: &Path) -> io::Result<u64> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.len()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(error),
    }
}

fn generation_path(path: &Path, generation: usize) -> PathBuf {
    PathBuf::from(format!("{}.{}", path.display(), generation))
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn rename_if_exists(from: &Path, to: &Path) -> io::Result<()> {
    match fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn rotate_files(path: &Path, retained_generations: usize) -> io::Result<()> {
    if retained_generations == 0 {
        remove_if_exists(path)?;
        return Ok(());
    }

    let oldest = generation_path(path, retained_generations);
    remove_if_exists(&oldest)?;

    for generation in (1..retained_generations).rev() {
        let from = generation_path(path, generation);
        let to = generation_path(path, generation + 1);
        rename_if_exists(&from, &to)?;
    }

    let first_generation = generation_path(path, 1);
    rename_if_exists(path, &first_generation)
}

#[cfg(test)]
mod tests {
    use super::{RotatingLineWriter, WriteLineOutcome};
    use std::fs;
    use tempfile::tempdir;

    fn read(path: &std::path::Path) -> String {
        fs::read_to_string(path).expect("read log")
    }

    #[test]
    fn writes_until_next_complete_record_requires_rotation() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runtime.log");
        let mut writer = RotatingLineWriter::new(&path, 9, 2).expect("writer");

        assert_eq!(
            writer.write_line("abcd").expect("first write"),
            WriteLineOutcome::Written
        );
        assert_eq!(writer.current_bytes(), 5);
        assert_eq!(
            writer.write_line("efgh").expect("second write"),
            WriteLineOutcome::Written
        );
        writer.flush().expect("flush");

        assert_eq!(read(&path), "efgh\n");
        assert_eq!(read(&path.with_extension("log.1")), "abcd\n");
    }

    #[test]
    fn retains_only_requested_number_of_generations() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("conversation.jsonl");
        let mut writer = RotatingLineWriter::new(&path, 4, 2).expect("writer");

        for record in ["one", "two", "tri", "for"] {
            assert_eq!(
                writer.write_line(record).expect("write"),
                WriteLineOutcome::Written
            );
        }
        writer.flush().expect("flush");

        assert_eq!(read(&path), "for\n");
        assert_eq!(read(&path.with_extension("jsonl.1")), "tri\n");
        assert_eq!(read(&path.with_extension("jsonl.2")), "two\n");
        assert!(!path.with_extension("jsonl.3").exists());
    }

    #[test]
    fn startup_rotates_an_oversized_active_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runtime.log");
        fs::write(&path, "oversized\n").expect("write oversized log");

        let mut writer = RotatingLineWriter::new(&path, 5, 1).expect("writer");
        assert_eq!(writer.current_bytes(), 0);
        assert_eq!(
            writer.write_line("ok").expect("write"),
            WriteLineOutcome::Written
        );
        writer.flush().expect("flush");

        assert_eq!(read(&path.with_extension("log.1")), "oversized\n");
        assert_eq!(read(&path), "ok\n");
    }

    #[test]
    fn zero_retained_generations_discards_old_active_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runtime.log");
        let mut writer = RotatingLineWriter::new(&path, 4, 0).expect("writer");

        writer.write_line("one").expect("first write");
        writer.write_line("two").expect("second write");
        writer.flush().expect("flush");

        assert_eq!(read(&path), "two\n");
        assert!(!path.with_extension("log.1").exists());
    }

    #[test]
    fn rejects_oversized_record_without_partial_write() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runtime.log");
        let mut writer = RotatingLineWriter::new(&path, 6, 1).expect("writer");

        assert_eq!(
            writer.write_line("tool-output").expect("write"),
            WriteLineOutcome::RecordTooLarge {
                record_bytes: 11,
                max_record_bytes: 5,
            }
        );
        writer.flush().expect("flush");

        assert_eq!(read(&path), "");
    }
}
