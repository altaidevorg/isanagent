//! Shared Python REPL framing for local and SSH execution providers.

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::error::ExecutionError;

/// Max UTF-8 source bytes per REPL round-trip (defense in depth; matches Python worker bound).
pub const MAX_REPL_SOURCE_BYTES: usize = 16 * 1024 * 1024;

/// Embedded worker: read `>I` length + UTF-8 source, `exec` in shared namespace, reply `>III` + payloads.
pub const PYTHON_REPL_BOOTSTRAP: &str = r#"import struct,sys,traceback,io,contextlib
_g={}
while 1:
 h=sys.stdin.buffer.read(4)
 if len(h)<4:
  sys.exit(0)
 n=int.from_bytes(h,'big')
 if n>16777216 or n<0:
  sys.exit(2)
 s=sys.stdin.buffer.read(n).decode('utf-8','replace')
 o,e=io.StringIO(),io.StringIO()
 c=0
 try:
  with contextlib.redirect_stdout(o),contextlib.redirect_stderr(e):
   exec(compile(s,'<isanagent>','exec'),_g,_g)
 except Exception:
  traceback.print_exc(file=e)
  c=1
 ob,eb=o.getvalue().encode('utf-8','backslashreplace'),e.getvalue().encode('utf-8','backslashreplace')
 sys.stdout.buffer.write(struct.pack('>III',c,len(ob),len(eb))+ob+eb)
 sys.stdout.buffer.flush()
"#;

pub async fn read_exact<R: AsyncRead + Unpin>(
    r: &mut R,
    buf: &mut [u8],
) -> Result<(), ExecutionError> {
    let mut off = 0;
    while off < buf.len() {
        let n = r
            .read(&mut buf[off..])
            .await
            .map_err(|e| ExecutionError::Provider(format!("repl read: {e}")))?;
        if n == 0 {
            return Err(ExecutionError::Provider(
                "repl: unexpected EOF from peer".into(),
            ));
        }
        off += n;
    }
    Ok(())
}

/// Read `total` bytes from `r`, keeping at most `cap` bytes in the returned buffer (rest discarded).
pub async fn read_exact_capped<R: AsyncRead + Unpin>(
    r: &mut R,
    total: usize,
    cap: usize,
) -> Result<Vec<u8>, ExecutionError> {
    let mut out = Vec::new();
    let mut remaining = total;
    let mut buf = [0u8; 16384];
    while remaining > 0 {
        let chunk = remaining.min(buf.len());
        read_exact(r, &mut buf[..chunk]).await?;
        if out.len() < cap {
            let room = cap.saturating_sub(out.len());
            let take = chunk.min(room);
            out.extend_from_slice(&buf[..take]);
        }
        remaining -= chunk;
    }
    Ok(out)
}

/// One REPL round-trip over a bidirectional byte stream (local pipes or SSH `ChannelStream`).
pub async fn repl_round_trip<W: AsyncWrite + Unpin, R: AsyncRead + Unpin>(
    writer: &mut W,
    reader: &mut R,
    code: &str,
    max_each: usize,
) -> Result<(String, String, i32), ExecutionError> {
    let cbytes = code.as_bytes();
    if cbytes.len() > MAX_REPL_SOURCE_BYTES {
        return Err(ExecutionError::InvalidArgument(format!(
            "code exceeds max repl bytes ({MAX_REPL_SOURCE_BYTES})"
        )));
    }
    let len_u = cbytes.len() as u32;
    writer
        .write_all(&len_u.to_be_bytes())
        .await
        .map_err(|e| ExecutionError::Provider(format!("repl stdin: {e}")))?;
    writer
        .write_all(cbytes)
        .await
        .map_err(|e| ExecutionError::Provider(format!("repl stdin: {e}")))?;
    writer
        .flush()
        .await
        .map_err(|e| ExecutionError::Provider(format!("repl stdin: {e}")))?;
    let mut hdr = [0u8; 12];
    read_exact(reader, &mut hdr).await?;
    let st = u32::from_be_bytes(
        <[u8; 4]>::try_from(&hdr[0..4])
            .map_err(|_| ExecutionError::Provider("repl: bad header (status)".into()))?,
    );
    let olen = u32::from_be_bytes(
        <[u8; 4]>::try_from(&hdr[4..8])
            .map_err(|_| ExecutionError::Provider("repl: bad header (stdout len)".into()))?,
    ) as usize;
    let elen = u32::from_be_bytes(
        <[u8; 4]>::try_from(&hdr[8..12])
            .map_err(|_| ExecutionError::Provider("repl: bad header (stderr len)".into()))?,
    ) as usize;
    const MAX_REPLY: usize = 64 * 1024 * 1024;
    if olen > MAX_REPLY || elen > MAX_REPLY {
        return Err(ExecutionError::Provider(
            "repl: invalid output length from worker".into(),
        ));
    }
    let out_raw = read_exact_capped(reader, olen, max_each).await?;
    let err_raw = read_exact_capped(reader, elen, max_each).await?;
    let stdout = string_from_utf8_lossy_trim_cap(out_raw, max_each);
    let stderr = string_from_utf8_lossy_trim_cap(err_raw, max_each);
    Ok((stdout, stderr, st as i32))
}

/// Truncate UTF-8 text to at most `max` **bytes** (not graphemes), on a char boundary, appending a marker when truncated.
pub fn truncate_utf8_str_cap(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut o = s[..end].to_string();
    o.push_str("\n... (truncated)");
    o
}

/// Lossy UTF-8 decode then [`truncate_utf8_str_cap`].
pub fn string_from_utf8_lossy_trim_cap(raw: Vec<u8>, max_each: usize) -> String {
    truncate_utf8_str_cap(&String::from_utf8_lossy(&raw), max_each)
}
