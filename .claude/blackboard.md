# Polyglot Notebook — Single Binary Architecture

## The Binary

One `cargo build`. Ships as one executable. Contains:

```
reactive runtime     (transcoded from marimo Python)
graph query engines  (transcoded from graph-notebook Python)
kernel protocol      (Rust-native ZMQ, from kernel-protocol spec)
document publisher   (transcoded from quarto TS/Deno)
local graph database (lance-graph, already Rust)
SIMD kernels         (ndarray, already Rust)
graph compiler       (rs-graph-llm, already Rust)
web frontend         (marimo's JS/React, served by the binary)
```

External process: R only (Bardioc/almato). Speaks Arrow IPC to the binary.

## Repos → Crates

| Repo (source) | Becomes | Work |
|------|---------|------|
| marimo | `crate::runtime` + `crate::server` | Transcode Python→Rust |
| graph-notebook | `crate::query::{cypher,gremlin,sparql,nars}` | Transcode Python→Rust |
| kernel-protocol | `crate::kernel` | Implement from spec in Rust |
| quarto | `crate::publish` | Transcode TS→Rust |
| quarto-r | external R process | Stays R, Arrow IPC bridge |
| lance-graph | `crate::graph` | Already Rust, integrate |
| ndarray | `crate::simd` + `crate::linalg` | Already Rust, integrate |
| rs-graph-llm | `crate::compiler` | Already Rust, fix build |

## Scopes (parallel, non-overlapping)

### SCOPE A: Reactive Runtime (marimo → Rust)
Transcode marimo's reactive cell execution model to Rust.
The core insight: cells have dependencies, when a cell's input changes,
downstream cells re-execute. That's a DAG scheduler — natural in Rust.

### SCOPE B: Query Engines (graph-notebook → Rust)
Transcode graph-notebook's Cypher/Gremlin/SPARQL executors to Rust.
Bolt protocol client, WebSocket client, HTTP client — all Rust-native.
Add local path: Cypher → lance-graph semiring (no network).

### SCOPE C: Kernel Protocol (kernel-protocol spec → Rust)
Implement Jupyter kernel wire protocol in Rust.
Only needed for R (IRkernel) — everything else runs in-process.
ZMQ via zeromq-rs. Connection file parsing. Message ser/de.

### SCOPE D: Publisher (quarto TS → Rust)
Transcode Quarto's document rendering pipeline to Rust.
Pandoc AST manipulation. Markdown → PDF/HTML.
Custom graph visualization extension.

### SCOPE E: Integration (lance-graph + ndarray + rs-graph-llm)
Wire the existing Rust crates into the binary.
Fix rs-graph-llm build. SIMD kernels for graph ops.
This is mostly Cargo.toml workspace wiring + API surface.

## Decisions
[DECISION] One binary, no Python runtime
[DECISION] marimo's JS frontend served by Rust HTTP server (axum/actix)
[DECISION] R is the ONLY external process (Arrow IPC bridge)
[DECISION] Cypher executes locally via lance-graph semiring by default
[DECISION] Remote DB connections (Neo4j, FalkorDB) via native Bolt client
[DECISION] vis.js graph rendering served as static assets by the binary
