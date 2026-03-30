//! HTTP range reader — `Read + Seek` over HTTP with range requests.
//!
//! Enables streaming GGUF indexing directly from HuggingFace without
//! downloading the full file to disk. Uses `Range: bytes=N-M` headers.
//!
//! ```text
//! let reader = HttpRangeReader::new(url, total_size)?;
//! let gguf = gguf::read_gguf_header(&mut reader)?;  // reads ~1 MB
//! for tensor in &gguf.tensors {
//!     let data = gguf::read_tensor_f32(&mut reader, &gguf, &tensor)?;
//!     // process tensor, then drop — bounded RAM
//! }
//! ```

use std::io::{self, Read, Seek, SeekFrom};
use std::process::{Command, Stdio};

/// HTTP range reader that implements Read + Seek.
///
/// Each `read()` call fetches bytes via `curl -r start-end`.
/// Buffered: fetches chunks of `chunk_size` to avoid per-byte HTTP calls.
pub struct HttpRangeReader {
    url: String,
    position: u64,
    total_size: u64,
    buffer: Vec<u8>,
    buf_start: u64,  // file offset where buffer starts
    buf_len: usize,  // valid bytes in buffer
    chunk_size: usize,
    bytes_downloaded: u64,
}

impl HttpRangeReader {
    /// Default chunk: 256 MB (fewer HTTP round-trips, fits in RAM easily).
    const DEFAULT_CHUNK: usize = 256 * 1024 * 1024;

    /// Create a new HTTP range reader.
    ///
    /// `total_size` must be known upfront (from HEAD request or HF metadata).
    pub fn new(url: String, total_size: u64) -> Self {
        Self {
            url,
            position: 0,
            total_size,
            buffer: vec![0u8; Self::DEFAULT_CHUNK],
            buf_start: 0,
            buf_len: 0,
            chunk_size: Self::DEFAULT_CHUNK,
            bytes_downloaded: 0,
        }
    }

    /// Create with custom chunk size.
    pub fn with_chunk_size(url: String, total_size: u64, chunk_size: usize) -> Self {
        Self {
            url,
            position: 0,
            total_size,
            buffer: vec![0u8; chunk_size],
            buf_start: 0,
            buf_len: 0,
            chunk_size,
            bytes_downloaded: 0,
        }
    }

    /// Total bytes fetched from network.
    pub fn bytes_downloaded(&self) -> u64 {
        self.bytes_downloaded
    }

    /// Fetch a range of bytes from the URL via curl.
    fn fetch_range(&mut self, start: u64, len: usize) -> io::Result<usize> {
        let end = (start + len as u64 - 1).min(self.total_size - 1);
        let range = format!("{}-{}", start, end);

        let output = Command::new("curl")
            .args(&["-sL", "-r", &range, &self.url])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("curl failed: {}", e)))?;

        if !output.status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("curl returned status {}", output.status),
            ));
        }

        let fetched = output.stdout.len();
        if fetched == 0 {
            return Ok(0);
        }

        // Copy to buffer
        let copy_len = fetched.min(self.buffer.len());
        self.buffer[..copy_len].copy_from_slice(&output.stdout[..copy_len]);
        self.buf_start = start;
        self.buf_len = copy_len;
        self.bytes_downloaded += fetched as u64;

        Ok(copy_len)
    }

    /// Ensure the buffer covers `self.position` and has data ready.
    fn ensure_buffered(&mut self) -> io::Result<()> {
        if self.position >= self.total_size {
            return Ok(());
        }

        // Check if position is within current buffer
        let buf_end = self.buf_start + self.buf_len as u64;
        if self.position >= self.buf_start && self.position < buf_end {
            return Ok(()); // already buffered
        }

        // Need to fetch
        let remaining = (self.total_size - self.position) as usize;
        let fetch_len = self.chunk_size.min(remaining);
        self.fetch_range(self.position, fetch_len)?;
        Ok(())
    }
}

impl Read for HttpRangeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.total_size {
            return Ok(0); // EOF
        }

        self.ensure_buffered()?;

        let buf_offset = (self.position - self.buf_start) as usize;
        let available = self.buf_len - buf_offset;
        let to_copy = buf.len().min(available);

        if to_copy == 0 {
            return Ok(0);
        }

        buf[..to_copy].copy_from_slice(&self.buffer[buf_offset..buf_offset + to_copy]);
        self.position += to_copy as u64;
        Ok(to_copy)
    }
}

impl Seek for HttpRangeReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::Current(n) => self.position as i64 + n,
            SeekFrom::End(n) => self.total_size as i64 + n,
        };

        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before start of file",
            ));
        }

        self.position = new_pos as u64;
        Ok(self.position)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.position)
    }
}

/// Resolve a HuggingFace model file URL and get its size.
///
/// Returns (final_url, size_bytes).
pub fn resolve_hf_url(repo: &str, filename: &str) -> Result<(String, u64), String> {
    // Get redirect URL and Content-Length via curl HEAD
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo, filename
    );

    // Follow redirects, get final URL and size
    let output = Command::new("curl")
        .args(&["-sIL", &url])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    let headers = String::from_utf8_lossy(&output.stdout);
    let mut size: u64 = 0;
    let mut final_url = url.clone();

    for line in headers.lines() {
        if let Some(val) = line.strip_prefix("content-length: ").or(line.strip_prefix("Content-Length: ")) {
            if let Ok(s) = val.trim().parse::<u64>() {
                size = s;
            }
        }
        if let Some(val) = line.strip_prefix("location: ").or(line.strip_prefix("Location: ")) {
            final_url = val.trim().to_string();
        }
    }

    if size == 0 {
        // Try python fallback
        let py_out = Command::new("python3")
            .args(&["-c", &format!(
                "from huggingface_hub import hf_hub_url, get_hf_file_metadata; \
                 url = hf_hub_url('{}', '{}'); \
                 meta = get_hf_file_metadata(url); \
                 print(meta.size); print(url)",
                repo, filename
            )])
            .output()
            .map_err(|e| format!("python3 fallback failed: {}", e))?;

        let py_text = String::from_utf8_lossy(&py_out.stdout);
        let lines: Vec<&str> = py_text.lines().collect();
        if lines.len() >= 2 {
            size = lines[0].trim().parse().unwrap_or(0);
            final_url = lines[1].trim().to_string();
        }
    }

    if size == 0 {
        return Err("Could not determine file size".into());
    }

    Ok((final_url, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seek_positions() {
        let mut r = HttpRangeReader::new("http://example.com/test".into(), 1000);
        assert_eq!(r.stream_position().unwrap(), 0);

        r.seek(SeekFrom::Start(500)).unwrap();
        assert_eq!(r.stream_position().unwrap(), 500);

        r.seek(SeekFrom::Current(100)).unwrap();
        assert_eq!(r.stream_position().unwrap(), 600);

        r.seek(SeekFrom::End(-100)).unwrap();
        assert_eq!(r.stream_position().unwrap(), 900);
    }

    #[test]
    fn test_seek_before_start() {
        let mut r = HttpRangeReader::new("http://example.com/test".into(), 1000);
        assert!(r.seek(SeekFrom::Start(0)).is_ok());
        assert!(r.seek(SeekFrom::End(-2000)).is_err());
    }

    #[test]
    fn test_read_at_eof() {
        let mut r = HttpRangeReader::new("http://example.com/test".into(), 0);
        let mut buf = [0u8; 10];
        assert_eq!(r.read(&mut buf).unwrap(), 0);
    }
}
