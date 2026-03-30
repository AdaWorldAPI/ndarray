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
/// Internally splits chunks into smaller segments (default 64 MB) with
/// retry + exponential backoff. A mid-download failure only refetches
/// the failed segment, not the entire 256 MB chunk.
///
/// Caches the last `SEGMENT_CACHE_SIZE` segments so backward seeks
/// within the cache window are free (no re-fetch).
pub struct HttpRangeReader {
    url: String,
    position: u64,
    total_size: u64,
    chunk_size: usize,
    bytes_downloaded: u64,

    // Segmented cache: each entry = (file_offset, data)
    segment_size: usize,
    segment_cache: Vec<(u64, Vec<u8>)>,
    max_cached_segments: usize,

    // Current active segment (for Read trait)
    active_segment_start: u64,
    active_segment_len: usize,
    active_segment_idx: Option<usize>, // index into segment_cache
}

/// Maximum retry attempts per segment.
const MAX_RETRIES: u32 = 4;
/// Initial backoff delay in milliseconds.
const INITIAL_BACKOFF_MS: u64 = 1000;
/// Default segment size: 64 MB (4 segments per 256 MB chunk).
const DEFAULT_SEGMENT_SIZE: usize = 64 * 1024 * 1024;
/// Number of segments to cache (192 MB at 64 MB segments).
const SEGMENT_CACHE_SIZE: usize = 3;

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
            chunk_size: Self::DEFAULT_CHUNK,
            bytes_downloaded: 0,
            segment_size: DEFAULT_SEGMENT_SIZE,
            segment_cache: Vec::with_capacity(SEGMENT_CACHE_SIZE),
            max_cached_segments: SEGMENT_CACHE_SIZE,
            active_segment_start: 0,
            active_segment_len: 0,
            active_segment_idx: None,
        }
    }

    /// Create with custom chunk size.
    pub fn with_chunk_size(url: String, total_size: u64, chunk_size: usize) -> Self {
        // Segment size = chunk_size / 4, minimum 16 MB
        let seg = (chunk_size / 4).max(16 * 1024 * 1024);
        Self {
            url,
            position: 0,
            total_size,
            chunk_size,
            bytes_downloaded: 0,
            segment_size: seg,
            segment_cache: Vec::with_capacity(SEGMENT_CACHE_SIZE),
            max_cached_segments: SEGMENT_CACHE_SIZE,
            active_segment_start: 0,
            active_segment_len: 0,
            active_segment_idx: None,
        }
    }

    /// Total bytes fetched from network.
    pub fn bytes_downloaded(&self) -> u64 {
        self.bytes_downloaded
    }

    /// Fetch a segment with retry + exponential backoff.
    ///
    /// Returns the fetched bytes. On permanent failure after MAX_RETRIES, returns error.
    fn fetch_segment_with_retry(&mut self, start: u64, len: usize) -> io::Result<Vec<u8>> {
        let end = (start + len as u64 - 1).min(self.total_size - 1);
        let range = format!("{}-{}", start, end);
        let expected = (end - start + 1) as usize;

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                let delay = INITIAL_BACKOFF_MS * (1u64 << (attempt - 1));
                eprintln!("    retry {}/{} after {}ms (segment {}-{})",
                    attempt + 1, MAX_RETRIES, delay, start, end);
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }

            let result = Command::new("curl")
                .args(&[
                    "-sL",
                    "--retry", "2",          // curl-level retry for connection drops
                    "--retry-delay", "1",
                    "--connect-timeout", "30",
                    "--max-time", "300",      // 5 min max per segment
                    "-r", &range,
                    &self.url,
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output();

            match result {
                Ok(output) if output.status.success() && output.stdout.len() == expected => {
                    self.bytes_downloaded += output.stdout.len() as u64;
                    return Ok(output.stdout);
                }
                Ok(output) if output.status.success() && !output.stdout.is_empty() => {
                    // Partial read — might be near EOF, accept it
                    self.bytes_downloaded += output.stdout.len() as u64;
                    if output.stdout.len() >= expected / 2 {
                        return Ok(output.stdout);
                    }
                    eprintln!("    short read: got {}/{} bytes", output.stdout.len(), expected);
                }
                Ok(output) => {
                    eprintln!("    fetch failed: status={} got={} bytes",
                        output.status, output.stdout.len());
                }
                Err(e) => {
                    eprintln!("    curl error: {}", e);
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("segment fetch failed after {} retries: bytes {}-{}", MAX_RETRIES, start, end),
        ))
    }

    /// Find or fetch the segment containing `self.position`.
    fn ensure_segment(&mut self) -> io::Result<()> {
        if self.position >= self.total_size {
            return Ok(());
        }

        // Check if position is within any cached segment
        for (idx, (seg_start, seg_data)) in self.segment_cache.iter().enumerate() {
            let seg_end = *seg_start + seg_data.len() as u64;
            if self.position >= *seg_start && self.position < seg_end {
                self.active_segment_start = *seg_start;
                self.active_segment_len = seg_data.len();
                self.active_segment_idx = Some(idx);
                return Ok(());
            }
        }

        // Not cached — fetch new segment
        let remaining = (self.total_size - self.position) as usize;
        let fetch_len = self.segment_size.min(remaining);
        let data = self.fetch_segment_with_retry(self.position, fetch_len)?;
        let data_len = data.len();

        // Evict oldest segment if cache is full
        if self.segment_cache.len() >= self.max_cached_segments {
            self.segment_cache.remove(0);
        }

        self.segment_cache.push((self.position, data));
        let idx = self.segment_cache.len() - 1;

        self.active_segment_start = self.position;
        self.active_segment_len = data_len;
        self.active_segment_idx = Some(idx);

        Ok(())
    }
}

impl Read for HttpRangeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.total_size {
            return Ok(0); // EOF
        }

        self.ensure_segment()?;

        let idx = match self.active_segment_idx {
            Some(i) if i < self.segment_cache.len() => i,
            _ => return Ok(0),
        };

        let (seg_start, ref seg_data) = self.segment_cache[idx];
        let offset = (self.position - seg_start) as usize;
        let available = seg_data.len() - offset;
        let to_copy = buf.len().min(available);

        if to_copy == 0 {
            return Ok(0);
        }

        buf[..to_copy].copy_from_slice(&seg_data[offset..offset + to_copy]);
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
