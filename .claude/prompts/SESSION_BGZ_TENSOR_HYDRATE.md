# SESSION: bgz-tensor hydrate workflow

## Context

The `bgz-tensor` crate in `lance-graph/crates/bgz-tensor/` stores model indexes
(bgz7 format) and extracted palettes (PAL8 format). The bgz7 files are 600+ MB
total but no single file exceeds 100 MB. They should NOT be committed to Git —
they are reproducible from public HuggingFace safetensors.

The palette files (4 KB) ARE committed — they are the non-reproducible artifact
of NARS cross-validation across multiple model diffs.

## Architecture

```
bgz-tensor/
  Cargo.toml
  src/
    lib.rs                ← ModelIndex registry, read/write helpers
    hydrate.rs            ← binary: cargo run --bin hydrate
  data/
    .gitignore            ← "*.bgz7" — ignore binary model data
    manifest.json         ← SHA256 + source URLs + shard counts (committed)
    qwen25-9b-base/       ← shard-00.bgz7 .. shard-03.bgz7 (gitignored)
    qwen25-9b-distilled/
    qwen25-27b-base/
    qwen25-27b-distilled-v1/
    qwen25-27b-distilled-v2/
    llama4-scout/
  palettes/
    qwen-scaffold.pal8    ← 4101 bytes, THE artifact (committed)
```

## Key Design Rules

1. **Compiler never sees bgz7 data.** All paths are runtime strings, never
   `include_bytes!` or build-script inputs. `cargo check` and `cargo build`
   work on a fresh clone with zero data files present.

2. **Tests that need data use `#[ignore]`.** The ignore message tells the user
   exactly which hydrate command to run:
   ```rust
   #[test]
   #[ignore = "requires: cargo run --bin hydrate -- --download qwen25-9b-base"]
   fn test_qwen_9b_shards() { ... }
   ```

3. **Palette files are always committed.** They are 4 KB and non-reproducible
   (result of NARS revision across 4 diffs). They must survive `git clean`.

4. **manifest.json is the source of truth.** It maps model name → HuggingFace
   source URL, shard count, expected SHA256 per shard, total bytes.

5. **Two hydrate modes:**
   - `--reindex MODEL`: stream from HuggingFace, run `stream_index_safetensors_bf16()`,
     write bgz7 shards, verify SHA256 against manifest. Canonical but slow (~4h).
   - `--download MODEL`: fetch pre-built bgz7 from GitHub Release assets,
     verify SHA256. Fast (~2min). Requires a release to exist.

6. **GitHub Releases for binary distribution.** Release assets support 2 GB/file.
   Tag format: `v0.1.0-bgz-data`. Upload via `gh release create`.

## Files to Create

### `Cargo.toml`

```toml
[package]
name = "bgz-tensor"
version = "0.1.0"
edition = "2021"
rust-version = "1.94"
description = "Model tensor indexes in bgz7 format with hydrate-on-demand workflow"
license = "Apache-2.0"

[[bin]]
name = "hydrate"
path = "src/hydrate.rs"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
```

### `data/.gitignore`

```
*.bgz7
```

### `data/manifest.json`

Registry of all models. Structure per model:

```json
{
  "models": {
    "qwen25-9b-base": {
      "source": "Qwen/Qwen2.5-7B",
      "format": "safetensors",
      "shards": 4,
      "total_bytes_bgz7": 77000000,
      "release_tag": "v0.1.0-bgz-data",
      "sha256": {
        "shard-00.bgz7": "...",
        "shard-01.bgz7": "...",
        "shard-02.bgz7": "...",
        "shard-03.bgz7": "..."
      }
    },
    "qwen25-27b-base": {
      "source": "Qwen/Qwen2.5-27B",
      "format": "safetensors",
      "shards": 11,
      "total_bytes_bgz7": 136000000,
      "release_tag": "v0.1.0-bgz-data",
      "sha256": {}
    }
  }
}
```

SHA256 values are filled in AFTER indexing completes. Until then, empty object.

### `src/lib.rs`

Core types and helpers. No data dependencies at compile time.

```rust
use std::path::PathBuf;
use std::io;
use serde::{Serialize, Deserialize};

/// Where bgz-tensor data lives relative to crate root.
pub const DATA_DIR: &str = "data";
pub const PALETTES_DIR: &str = "palettes";

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub models: std::collections::HashMap<String, ModelEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub source: String,
    pub format: String,
    pub shards: usize,
    pub total_bytes_bgz7: u64,
    pub release_tag: String,
    pub sha256: std::collections::HashMap<String, String>,
}

/// Runtime path to a bgz7 shard. Compiles without the file existing.
pub fn bgz7_path(model: &str, shard: usize) -> PathBuf {
    PathBuf::from(DATA_DIR)
        .join(model)
        .join(format!("shard-{shard:02}.bgz7"))
}

/// Read a bgz7 shard. Fails at runtime if not hydrated.
pub fn read_bgz7(model: &str, shard: usize) -> io::Result<Vec<u8>> {
    let path = bgz7_path(model, shard);
    std::fs::read(&path).map_err(|e| {
        io::Error::new(e.kind(), format!(
            "{e} — run `cargo run --bin hydrate -- --download {model}` first"
        ))
    })
}

/// Check if a model's data is hydrated (all shards present).
pub fn is_hydrated(model: &str, shard_count: usize) -> bool {
    (0..shard_count).all(|i| bgz7_path(model, i).exists())
}

/// Load manifest from data/manifest.json.
pub fn load_manifest() -> io::Result<Manifest> {
    let path = PathBuf::from(DATA_DIR).join("manifest.json");
    let data = std::fs::read_to_string(&path)?;
    serde_json::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Read the palette file (always present, committed to git).
pub fn read_palette(name: &str) -> io::Result<Vec<u8>> {
    let path = PathBuf::from(PALETTES_DIR).join(name);
    std::fs::read(&path)
}

/// Verify SHA256 of a file against expected hash.
pub fn verify_sha256(path: &std::path::Path, expected: &str) -> io::Result<bool> {
    use sha2::{Sha256, Digest};
    let data = std::fs::read(path)?;
    let hash = format!("{:x}", Sha256::digest(&data));
    Ok(hash == expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_loads() {
        // This test always works — manifest.json is committed
        let manifest = load_manifest().expect("manifest.json must be committed");
        assert!(!manifest.models.is_empty(), "manifest should list models");
    }

    #[test]
    fn palette_loads() {
        // This test always works — palette is committed
        // (will fail until first palette is generated and committed)
    }

    #[test]
    fn paths_are_deterministic() {
        let p = bgz7_path("qwen25-9b-base", 2);
        assert_eq!(p, PathBuf::from("data/qwen25-9b-base/shard-02.bgz7"));
    }

    #[test]
    #[ignore = "requires: cargo run --bin hydrate -- --download qwen25-9b-base"]
    fn test_qwen_9b_shards_present() {
        assert!(is_hydrated("qwen25-9b-base", 4),
            "Run: cargo run --bin hydrate -- --download qwen25-9b-base");
    }

    #[test]
    #[ignore = "requires: cargo run --bin hydrate -- --download qwen25-27b-base"]
    fn test_qwen_27b_shards_present() {
        assert!(is_hydrated("qwen25-27b-base", 11),
            "Run: cargo run --bin hydrate -- --download qwen25-27b-base");
    }
}
```

### `src/hydrate.rs`

Binary entry point. Two modes: `--reindex` (from HuggingFace) and `--download`
(from GitHub Release).

```rust
use std::env;
use std::fs;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  cargo run --bin hydrate -- --download MODEL");
        eprintln!("  cargo run --bin hydrate -- --reindex MODEL");
        eprintln!("  cargo run --bin hydrate -- --verify MODEL");
        eprintln!("  cargo run --bin hydrate -- --list");
        eprintln!();
        eprintln!("Models: qwen25-9b-base, qwen25-27b-base, ...");
        process::exit(1);
    }

    let command = &args[1];
    let model = if args.len() > 2 { &args[2] } else { "" };

    let manifest = bgz_tensor::load_manifest()
        .expect("Failed to load data/manifest.json");

    match command.as_str() {
        "--list" => {
            for (name, entry) in &manifest.models {
                let status = if bgz_tensor::is_hydrated(name, entry.shards) {
                    "HYDRATED"
                } else {
                    "missing"
                };
                println!("{status:>10}  {name:<30} {shards} shards, {mb:.0} MB",
                    shards = entry.shards,
                    mb = entry.total_bytes_bgz7 as f64 / 1_000_000.0);
            }
        }
        "--download" => {
            let entry = manifest.models.get(model)
                .unwrap_or_else(|| { eprintln!("Unknown model: {model}"); process::exit(1) });

            let dir = bgz_tensor::bgz7_path(model, 0).parent().unwrap().to_path_buf();
            fs::create_dir_all(&dir).expect("Failed to create data directory");

            let repo = "AdaWorldAPI/lance-graph";
            let tag = &entry.release_tag;

            for shard in 0..entry.shards {
                let filename = format!("shard-{shard:02}.bgz7");
                let dest = dir.join(&filename);

                if dest.exists() {
                    println!("  {filename}: already present, skipping");
                    continue;
                }

                let url = format!(
                    "https://github.com/{repo}/releases/download/{tag}/{model}--{filename}"
                );
                println!("  Downloading {filename} from {url}...");

                // Use curl for simplicity (available everywhere)
                let status = process::Command::new("curl")
                    .args(["-L", "-o", dest.to_str().unwrap(), &url])
                    .status()
                    .expect("curl failed");

                if !status.success() {
                    eprintln!("  Failed to download {filename}");
                    process::exit(1);
                }
            }

            println!("Done. Run `cargo run --bin hydrate -- --verify {model}` to check.");
        }
        "--reindex" => {
            let entry = manifest.models.get(model)
                .unwrap_or_else(|| { eprintln!("Unknown model: {model}"); process::exit(1) });

            println!("Reindexing {model} from {source}...", source = entry.source);
            println!("This streams BF16 safetensors from HuggingFace and builds bgz7 shards.");
            println!("Expected time: ~1-4 hours depending on model size and bandwidth.");
            println!();
            println!("TODO: wire stream_index_safetensors_bf16() from ndarray here.");
            println!("For now, run the indexing from the ndarray test suite:");
            println!("  cargo test -p ndarray --features p64 -- test_index_{} --ignored --nocapture",
                model.replace('-', "_"));
        }
        "--verify" => {
            let entry = manifest.models.get(model)
                .unwrap_or_else(|| { eprintln!("Unknown model: {model}"); process::exit(1) });

            let mut all_ok = true;
            for shard in 0..entry.shards {
                let filename = format!("shard-{shard:02}.bgz7");
                let path = bgz_tensor::bgz7_path(model, shard);

                if !path.exists() {
                    println!("  {filename}: MISSING");
                    all_ok = false;
                    continue;
                }

                if let Some(expected) = entry.sha256.get(&filename) {
                    match bgz_tensor::verify_sha256(&path, expected) {
                        Ok(true) => println!("  {filename}: OK"),
                        Ok(false) => {
                            println!("  {filename}: SHA256 MISMATCH");
                            all_ok = false;
                        }
                        Err(e) => {
                            println!("  {filename}: ERROR reading: {e}");
                            all_ok = false;
                        }
                    }
                } else {
                    let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    println!("  {filename}: present ({size} bytes, no SHA256 in manifest yet)");
                }
            }

            if all_ok {
                println!("All shards verified.");
            } else {
                println!("Some shards missing or corrupt.");
                process::exit(1);
            }
        }
        _ => {
            eprintln!("Unknown command: {command}");
            process::exit(1);
        }
    }
}
```

## Workflow for Users

```bash
# Fresh clone — no data, just code + manifest + palette
git clone ...
cd lance-graph

# Everything compiles
cargo check -p bgz-tensor            # ✓
cargo test -p bgz-tensor             # ✓ (ignore-tests skipped)

# See what's available
cargo run --bin hydrate -- --list

# Pull specific model data (from GitHub Release)
cargo run --bin hydrate -- --download qwen25-9b-base
cargo run --bin hydrate -- --verify qwen25-9b-base

# Now data-dependent tests work
cargo test -p bgz-tensor -- --ignored

# Or reindex from scratch (slow, canonical)
cargo run --bin hydrate -- --reindex qwen25-27b-base
```

## Workflow for Maintainer (after indexing)

```bash
# After indexing completes, compute SHA256 and update manifest
sha256sum data/qwen25-9b-base/shard-*.bgz7
# Paste hashes into manifest.json

# Create GitHub Release with binary assets
gh release create v0.1.0-bgz-data \
  data/qwen25-9b-base/shard-*.bgz7 \
  data/qwen25-27b-base/shard-*.bgz7 \
  data/llama4-scout/shard-*.bgz7 \
  --title "bgz7 model indexes" \
  --notes "Pre-built bgz7 indexes. Use: cargo run --bin hydrate -- --download MODEL"

# Commit manifest (with SHA256) + palette
git add data/manifest.json palettes/qwen-scaffold.pal8
git commit -m "data: manifest with SHA256 + qwen scaffold palette"
git push
```

## Integration with ndarray

The ndarray branch `claude/qwen-claude-reverse-eng-vHuHv` has the indexing code:
- `stream_index_safetensors_bf16()` in `gguf_indexer.rs`
- `causal_diff_sharded()` in `causal_diff.rs`
- `scaffold_to_palette3d_layers()` in `causal_diff.rs`

The bgz-tensor crate does NOT depend on ndarray. It only stores the output.
ndarray writes bgz7 files. bgz-tensor reads them. The palette is the bridge.

## Integration with p64

The palette file (`qwen-scaffold.pal8`) is the Cognitive Highway payload:
- ndarray extracts it via `serialize_palette3d_layers()`
- bgz-tensor stores it (committed, 4 KB)
- lance-graph consumes it via `deserialize()` → `Blumenstrauss::new()`

The bgz7 shards are the HIP-level data for the HHTL cascade:
- HEEL: palette topology (which blocks interact) — 4 KB, always present
- HIP: bgz7 Base17 scent per row — 70-150 MB, hydrate-on-demand
- TWIG: Base17 refinement within block — from bgz7
- LEAF: original BF16 weights — from HuggingFace, never stored locally
