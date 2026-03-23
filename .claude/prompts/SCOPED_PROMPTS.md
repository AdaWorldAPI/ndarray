# SCOPE A: Reactive Runtime Transcode (marimo Python → Rust)

## You touch: marimo
## You do NOT touch: graph-notebook, kernel-protocol, lance-graph, ndarray, quarto, quarto-r, rs-graph-llm

## Goal
Transcode marimo's reactive cell execution model from Python to Rust.
The output is a `crate::runtime` that schedules cell execution based on
a dependency DAG. When cell A's output changes, all cells that depend on
A re-execute.

## Step 1: Read (before any code)
```bash
# The reactive runtime
find marimo/marimo/_runtime/ -name "*.py" | sort
cat marimo/marimo/_runtime/runtime.py
cat marimo/marimo/_runtime/dataflow.py

# How cells declare dependencies
grep -rn "def cell\|@app.cell\|refs\|defs" marimo/marimo/_runtime/ | head -30

# The server (what serves the frontend)
find marimo/marimo/_server/ -name "*.py" | sort | head -20

# The frontend (JS — stays JS, served as static assets)
ls marimo/frontend/src/ | head -20
```

## Step 2: Map (write findings before coding)
Write `.claude/SCOPE_A_FINDINGS.md`:
1. What is marimo's dependency tracking model? (refs/defs? AST analysis?)
2. What is the execution order algorithm? (topological sort?)
3. What is the cell state model? (inputs, outputs, status?)
4. What server framework does marimo use? (starlette? uvicorn?)
5. What WebSocket protocol does the frontend speak?

## Step 3: Design the Rust crate
```
src/runtime/
    mod.rs          — DAG scheduler, cell execution
    cell.rs         — Cell definition (code, refs, defs, output)
    dataflow.rs     — Dependency graph, topological sort
    executor.rs     — Cell execution engine
src/server/
    mod.rs          — axum HTTP server
    ws.rs           — WebSocket handler (same protocol as marimo frontend)
    static_files.rs — serve marimo's JS frontend as-is
```

## Constraints
- The JS frontend stays JavaScript. Don't rewrite React in Rust.
- The binary serves the frontend as static files.
- WebSocket protocol must match marimo's existing frontend expectations.
- Cell execution for graph queries delegates to SCOPE B's query engines.

---

# SCOPE B: Query Engines Transcode (graph-notebook Python → Rust)

## You touch: graph-notebook
## You do NOT touch: marimo, kernel-protocol, lance-graph, ndarray, quarto, quarto-r, rs-graph-llm

## Goal
Transcode graph-notebook's query executors from Python to Rust.
Bolt client for Cypher, WebSocket client for Gremlin, HTTP client for
SPARQL. Plus a NEW local path: Cypher → lance-graph semiring.

## Step 1: Read (before any code)
```bash
find graph-notebook/src/graph_notebook/magics/ -name "*.py" | sort
cat graph-notebook/src/graph_notebook/magics/graph_magic.py

grep -rn "bolt\|websocket\|http\|connect" \
    graph-notebook/src/graph_notebook/ --include="*.py" | head -30

find graph-notebook/src/graph_notebook/visualization/ -name "*.py" | sort
```

## Step 2: Map (write findings before coding)
Write `.claude/SCOPE_B_FINDINGS.md`:
1. What protocol does %%oc use to talk to Neo4j? (Bolt binary protocol)
2. What protocol does %%gremlin use? (WebSocket + Gremlin bytecode?)
3. What protocol does %%sparql use? (HTTP POST + application/sparql-query?)
4. What does each executor return? (rows? graph? both?)
5. What does vis.js need as input JSON?

## Step 3: Design the Rust crates
```
src/query/
    mod.rs          — QueryEngine trait
    cypher.rs       — Bolt client (tokio + bolt-proto crate or hand-rolled)
    gremlin.rs      — WebSocket client (tokio-tungstenite)
    sparql.rs       — HTTP client (reqwest)
    nars.rs         — NEW: NARS executor
    local.rs        — Local path: parse Cypher → call lance-graph planner
    result.rs       — QueryResult: rows (Arrow RecordBatch) + graph (nodes/edges JSON)
```

## Constraints
- Arrow RecordBatch as the universal result format
- Local Cypher path calls into lance-graph (SCOPE E wires the dependency)
- vis.js rendering: the binary serves graph JSON, frontend renders with vis.js

---

# SCOPE C: Kernel Protocol (spec → Rust)

## You touch: kernel-protocol
## You do NOT touch: marimo, graph-notebook, lance-graph, ndarray, quarto, quarto-r, rs-graph-llm

## Goal
Implement Jupyter kernel wire protocol in Rust. This is ONLY for R
(IRkernel). Everything else runs in-process in the binary.

## Step 1: Read
```bash
cat kernel-protocol/docs/messaging.rst
cat kernel-protocol/docs/kernels.rst
```

## Step 2: Map
Write `.claude/SCOPE_C_FINDINGS.md`:
1. What ZMQ socket types are needed? (ROUTER, DEALER, SUB, REP?)
2. What message types for basic execute? (execute_request, execute_reply, display_data?)
3. How does HMAC signing work?
4. What is a kernelspec? How does IRkernel register?
5. Minimal message set for: connect, execute R code, get result?

## Step 3: Design
```
src/kernel/
    mod.rs          — KernelClient: connect, execute, receive
    protocol.rs     — Message types, header, ser/de
    zmq.rs          — ZMQ socket management (zeromq crate)
    connection.rs   — Parse connection file JSON
    r_bridge.rs     — Arrow IPC: send DataFrame to R, receive DataFrame back
```

## Constraints
- Only needed for R. Rust and Python execute in-process.
- Arrow IPC for data exchange (not JSON serialization of DataFrames)
- Minimal implementation: execute + result. No completion, no inspection.

---

# SCOPE D: Publisher Transcode (quarto TS → Rust)

## You touch: quarto, quarto-r
## You do NOT touch: marimo, graph-notebook, kernel-protocol, lance-graph, ndarray, rs-graph-llm

## Goal
Transcode Quarto's document rendering pipeline from TypeScript to Rust.
Notebook cells → Pandoc AST → PDF/HTML. Custom extension for graph viz.

## Step 1: Read
```bash
cat quarto/claude.md
ls quarto/packages/
find quarto/packages/ -name "*.ts" -maxdepth 3 | sort | head -30

# How quarto-r calls quarto CLI
grep -rn "system\|processx\|quarto" quarto-r/R/ | head -20
```

## Step 2: Map
Write `.claude/SCOPE_D_FINDINGS.md`:
1. What does quarto's rendering pipeline look like? (stages?)
2. What is Pandoc's AST format? (JSON AST?)
3. What does quarto add on top of Pandoc? (cell execution? cross-refs?)
4. How do quarto extensions work? (Lua filters? custom renderers?)
5. What would quarto-r call if the CLI is a Rust binary instead of Deno?

## Step 3: Design
```
src/publish/
    mod.rs          — render(notebook) → PDF/HTML
    pandoc_ast.rs   — Pandoc AST types in Rust
    markdown.rs     — Markdown parser → Pandoc AST
    pdf.rs          — AST → PDF (via embedded Pandoc or tectonic)
    html.rs         — AST → HTML
    extensions/
        graph_viz.rs — Graph JSON → vis.js HTML embed in output
```

## Constraints
- quarto-r must still work — it calls CLI, we just replace the CLI binary
- Graph visualization must render in both PDF (static image) and HTML (interactive vis.js)
- Don't reimplement all of Pandoc — embed it or use a subset

---

# SCOPE E: Integration (lance-graph + ndarray + rs-graph-llm → workspace)

## You touch: lance-graph, ndarray, rs-graph-llm
## You do NOT touch: marimo, graph-notebook, kernel-protocol, quarto, quarto-r

## Goal
Wire the existing Rust crates into a single Cargo workspace that the
binary crate depends on. Fix rs-graph-llm build. Define the API surface
that SCOPE A (runtime) and SCOPE B (query engines) call.

## Step 1: Read
```bash
# lance-graph: what's the public API?
grep -rn "pub fn\|pub struct\|pub trait" lance-graph/crates/blasgraph/src/lib.rs | head -20

# ndarray: what's the public API for SIMD?
grep "pub fn\|pub use" ndarray/src/simd.rs | head -20

# rs-graph-llm: what's broken?
cat rs-graph-llm/CLAUDE.md 2>/dev/null
cargo check --manifest-path rs-graph-llm/Cargo.toml 2>&1 | tail -30
```

## Step 2: Map
Write `.claude/SCOPE_E_FINDINGS.md`:
1. What is blasgraph's public API for executing a query plan?
2. What is ndarray's public API for SIMD ops that query engines need?
3. What are rs-graph-llm's build errors? (list them all)
4. What Cargo workspace structure fits all crates?
5. What API does SCOPE B's local Cypher path need from lance-graph?

## Step 3: Design workspace
```
Cargo.toml (workspace)
    members = [
        "crates/runtime",      # SCOPE A output
        "crates/query",        # SCOPE B output
        "crates/kernel",       # SCOPE C output
        "crates/publish",      # SCOPE D output
        "crates/graph",        # lance-graph
        "crates/simd",         # ndarray
        "crates/compiler",     # rs-graph-llm
        "crates/notebook",     # the binary (depends on all above)
    ]
```

## Constraints
- Fix rs-graph-llm build FIRST — it blocks integration
- Don't restructure lance-graph or ndarray internals — wrap their APIs
- The binary crate is thin: main() starts the server, wires everything together
