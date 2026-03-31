//! HTTP range reader — `Read + Seek` over HTTP with range requests.
//!
//! Enables streaming GGUF indexing directly from HuggingFace without
//! downloading the full file to disk. Uses `Range: bytes=N-M` headers.
//!
//! Features:
//! - Segment-aligned fetches (no overlapping reads)
//! - HuggingFace CDN URL resolution via `resolve_hf_url()`
//! - Stall detection via `--speed-limit` / `--speed-time`
//! - Retry with exponential backoff (4 retries per segment)
//! - Re-resolve URL on 403/redirect failure (CDN token expiry)
//! - LRU segment cache (192 MB at 64 MB segments)

use std::io::{self, Read, Seek, SeekFrom};
use std::process::{Command, Stdio};

/// HTTP range reader that implements Read + Seek.
///
/// Internally splits reads into segment-aligned fetches (default 64 MB)
/// with retry + exponential backoff + stall detection. A mid-download
/// failure only refetches the failed segment, not the entire chunk.
///
/// Caches the last `SEGMENT_CACHE_SIZE` segments so backward seeks
/// within the cache window are free (no re-fetch).
pub struct HttpRangeReader {
    url: String,
    repo: Option<String>,       // for re-resolve on 403
    filename: Option<String>,   // for re-resolve on 403
    position: u64,
    total_size: u64,
    chunk_size: usize,
    bytes_downloaded: u64,

    // Segmented cache: each entry = (segment_aligned_offset, data)
    segment_size: usize,
    segment_cache: Vec<(u64, Vec<u8>)>,
    max_cached_segments: usize,

    // Current active segment (for Read trait)
    active_segment_start: u64,
    active_segment_len: usize,
    active_segment_idx: Option<usize>,
}

/// Maximum retry attempts per segment.
const MAX_RETRIES: u32 = 6;
/// Initial backoff delay in milliseconds.
const INITIAL_BACKOFF_MS: u64 = 2000;
/// Default segment size: 64 MB (4 segments per 256 MB chunk).
const DEFAULT_SEGMENT_SIZE: usize = 64 * 1024 * 1024;
/// Number of segments to cache (192 MB at 64 MB segments).
const SEGMENT_CACHE_SIZE: usize = 3;
/// Minimum transfer speed before curl aborts (bytes/sec). 100 KB/s.
const SPEED_LIMIT: u32 = 100_000;
/// Seconds below SPEED_LIMIT before curl aborts the transfer.
const SPEED_TIME: u32 = 30;

impl HttpRangeReader {
    /// Default chunk: 256 MB (fewer HTTP round-trips, fits in RAM easily).
    const DEFAULT_CHUNK: usize = 256 * 1024 * 1024;

    /// Create a new HTTP range reader.
    ///
    /// `total_size` must be known upfront (from HEAD request or HF metadata).
    pub fn new(url: String, total_size: u64) -> Self {
        Self {
            url,
            repo: None,
            filename: None,
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
        let seg = (chunk_size / 4).max(16 * 1024 * 1024);
        Self {
            url,
            repo: None,
            filename: None,
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

    /// Create from HuggingFace repo + filename. Resolves CDN URL and exact size.
    ///
    /// Uses `resolve_hf_url()` to get the final CDN URL and Content-Length.
    /// Stores repo/filename for re-resolution on 403 (token expiry).
    pub fn from_hf(repo: &str, filename: &str, chunk_size: usize) -> Result<Self, String> {
        let (url, size) = resolve_hf_url(repo, filename)?;
        let seg = (chunk_size / 4).max(16 * 1024 * 1024);
        Ok(Self {
            url,
            repo: Some(repo.to_string()),
            filename: Some(filename.to_string()),
            position: 0,
            total_size: size,
            chunk_size,
            bytes_downloaded: 0,
            segment_size: seg,
            segment_cache: Vec::with_capacity(SEGMENT_CACHE_SIZE),
            max_cached_segments: SEGMENT_CACHE_SIZE,
            active_segment_start: 0,
            active_segment_len: 0,
            active_segment_idx: None,
        })
    }

    /// Total bytes fetched from network.
    pub fn bytes_downloaded(&self) -> u64 {
        self.bytes_downloaded
    }

    /// Exact file size (from HEAD or HF API).
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Re-resolve the URL (e.g. after CDN token expiry).
    fn re_resolve_url(&mut self) -> io::Result<()> {
        if let (Some(repo), Some(filename)) = (&self.repo, &self.filename) {
            eprintln!("    re-resolving URL for {}/{}", repo, filename);
            match resolve_hf_url(repo, filename) {
                Ok((new_url, size)) => {
                    eprintln!("    resolved: {} ({} bytes)", new_url, size);
                    self.url = new_url;
                    if size > 0 && size != self.total_size {
                        eprintln!("    WARNING: size changed {} → {}", self.total_size, size);
                    }
                    Ok(())
                }
                Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
            }
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "cannot re-resolve: no repo/filename stored (use from_hf())",
            ))
        }
    }

    /// Align a position to segment boundaries.
    fn segment_start_for(&self, pos: u64) -> u64 {
        (pos / self.segment_size as u64) * self.segment_size as u64
    }

    /// Fetch a segment with retry + exponential backoff + stall detection.
    ///
    /// On 403 or repeated failure, attempts to re-resolve the URL (CDN token expiry).
    fn fetch_segment_with_retry(&mut self, start: u64, len: usize) -> io::Result<Vec<u8>> {
        let end = (start + len as u64 - 1).min(self.total_size - 1);
        let range = format!("{}-{}", start, end);
        let expected = (end - start + 1) as usize;
        let mut resolved_this_call = false;

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                let delay = INITIAL_BACKOFF_MS * (1u64 << (attempt - 1).min(4));
                eprintln!("    retry {}/{} after {}ms (segment {}-{})",
                    attempt + 1, MAX_RETRIES, delay, start, end);
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }

            let speed_limit_str = SPEED_LIMIT.to_string();
            let speed_time_str = SPEED_TIME.to_string();

            let result = Command::new("curl")
                .args(&[
                    "-sL",
                    "--retry", "2",
                    "--retry-delay", "2",
                    "--connect-timeout", "30",
                    "--max-time", "600",          // 10 min max per 64 MB segment
                    "--speed-limit", &speed_limit_str, // abort if < 100 KB/s
                    "--speed-time", &speed_time_str,   // for > 30 seconds
                    "-r", &range,
                    &self.url,
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output();

            match result {
                Ok(output) if output.status.success() && output.stdout.len() == expected => {
                    self.bytes_downloaded += output.stdout.len() as u64;
                    return Ok(output.stdout);
                }
                Ok(output) if output.status.success() && !output.stdout.is_empty() => {
                    self.bytes_downloaded += output.stdout.len() as u64;
                    if output.stdout.len() >= expected / 2 {
                        return Ok(output.stdout);
                    }
                    eprintln!("    short read: got {}/{} bytes", output.stdout.len(), expected);
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let code = output.status.code().unwrap_or(-1);
                    eprintln!("    fetch failed: exit={} got={} bytes stderr={}",
                        code, output.stdout.len(), stderr.trim());

                    // 403 or curl exit 22 (HTTP error) → re-resolve CDN URL
                    if (code == 22 || stderr.contains("403") || stderr.contains("expired"))
                        && !resolved_this_call
                    {
                        if self.re_resolve_url().is_ok() {
                            resolved_this_call = true;
                            eprintln!("    URL re-resolved, retrying immediately");
                            continue;
                        }
                    }

                    // Curl exit 28 = timeout, 56 = recv failure → stall detected
                    if code == 28 || code == 56 {
                        eprintln!("    stall/timeout detected (curl exit {})", code);
                    }
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
    ///
    /// Segments are aligned to `segment_size` boundaries to avoid
    /// overlapping fetches when position advances sequentially.
    fn ensure_segment(&mut self) -> io::Result<()> {
        if self.position >= self.total_size {
            return Ok(());
        }

        let aligned_start = self.segment_start_for(self.position);

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

        // Not cached — fetch aligned segment
        let remaining = (self.total_size - aligned_start) as usize;
        let fetch_len = self.segment_size.min(remaining);
        let data = self.fetch_segment_with_retry(aligned_start, fetch_len)?;
        let data_len = data.len();

        // Evict oldest segment if cache is full
        if self.segment_cache.len() >= self.max_cached_segments {
            self.segment_cache.remove(0);
        }

        self.segment_cache.push((aligned_start, data));
        let idx = self.segment_cache.len() - 1;

        self.active_segment_start = aligned_start;
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

/// Resolve a HuggingFace model file URL and get its exact size.
///
/// Tries three methods in order:
/// 1. `huggingface_hub` Python API (most reliable — handles auth, gated models)
/// 2. curl HEAD with redirect follow (fast but may fail on gated repos)
/// 3. HuggingFace Hub REST API (no Python dependency)
///
/// Returns (final_url, size_bytes).
pub fn resolve_hf_url(repo: &str, filename: &str) -> Result<(String, u64), String> {
    // Method 1: Python huggingface_hub (handles auth tokens, gated models)
    if let Ok(py_out) = Command::new("python3")
        .args(&["-c", &format!(
            "from huggingface_hub import hf_hub_url, get_hf_file_metadata; \
             url = hf_hub_url('{}', '{}'); \
             meta = get_hf_file_metadata(url); \
             print(meta.size); print(meta.location if hasattr(meta, 'location') else url)",
            repo, filename
        )])
        .output()
    {
        if py_out.status.success() {
            let text = String::from_utf8_lossy(&py_out.stdout);
            let lines: Vec<&str> = text.lines().collect();
            if lines.len() >= 2 {
                if let Ok(size) = lines[0].trim().parse::<u64>() {
                    if size > 0 {
                        let url = lines[1].trim().to_string();
                        eprintln!("  resolved via HF API: {} bytes", size);
                        return Ok((url, size));
                    }
                }
            }
        }
    }

    // Method 2: curl HEAD with redirect follow
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo, filename
    );

    if let Ok(output) = Command::new("curl")
        .args(&["-sIL", "--connect-timeout", "15", "--max-time", "30", &url])
        .output()
    {
        let headers = String::from_utf8_lossy(&output.stdout);
        let mut size: u64 = 0;
        let mut final_url = url.clone();

        for line in headers.lines() {
            let lower = line.to_lowercase();
            if lower.starts_with("content-length:") {
                if let Some(val) = line.split(':').nth(1) {
                    if let Ok(s) = val.trim().parse::<u64>() {
                        size = s;
                    }
                }
            }
            if lower.starts_with("location:") {
                if let Some(val) = line.split_once(':').map(|(_, v)| v.trim()) {
                    if val.starts_with("http") {
                        final_url = val.to_string();
                    }
                }
            }
        }

        if size > 0 {
            eprintln!("  resolved via curl HEAD: {} bytes", size);
            return Ok((final_url, size));
        }
    }

    // Method 3: HuggingFace Hub REST API (no Python needed)
    let api_url = format!(
        "https://huggingface.co/api/models/{}/tree/main/{}",
        repo, filename.rsplit('/').next().unwrap_or(filename)
    );
    if let Ok(output) = Command::new("curl")
        .args(&["-sL", "--connect-timeout", "15", "--max-time", "30", &api_url])
        .output()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        // Parse JSON-ish for "size": NNN
        if let Some(pos) = text.find("\"size\":") {
            let after = &text[pos + 7..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(size) = num_str.parse::<u64>() {
                if size > 0 {
                    eprintln!("  resolved via HF REST API: {} bytes", size);
                    return Ok((url, size));
                }
            }
        }
    }

    Err(format!("Could not resolve {}/{}", repo, filename))
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

    #[test]
    fn test_segment_alignment() {
        let r = HttpRangeReader::new("http://example.com/test".into(), 1_000_000_000);
        // Default segment = 64 MB = 67108864
        assert_eq!(r.segment_start_for(0), 0);
        assert_eq!(r.segment_start_for(1000), 0);
        assert_eq!(r.segment_start_for(67_108_864), 67_108_864);
        assert_eq!(r.segment_start_for(67_108_865), 67_108_864);
        assert_eq!(r.segment_start_for(134_217_728), 134_217_728);
    }

    #[test]
    #[ignore] // Requires network
    fn test_resolve_hf_url() {
        let (url, size) = resolve_hf_url(
            "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
            "BF16/Llama-4-Scout-17B-16E-Instruct-BF16-00005-of-00005.gguf",
        ).expect("resolve_hf_url");
        assert!(size > 0, "size should be > 0");
        assert!(url.contains("http"), "url should be HTTP: {}", url);
        eprintln!("resolved: {} ({} bytes)", url, size);
    }
}
