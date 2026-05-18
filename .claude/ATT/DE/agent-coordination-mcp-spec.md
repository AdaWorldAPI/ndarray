# Agent-Coordination-MCP-Spezifikation

> **Sprache:** Deutsch · siehe `../agent-coordination-mcp-spec.md` für die englische Quellfassung.
>
> **Status:** DRAFT  ·  **Version:** 0.1.0  ·  **Stand:** 2026-05-17
> **Format:** NLSpec (nach [strongdm/attractor](https://github.com/strongdm/attractor))
> **Substrat:** Designed als Schwester zu `unified-llm-spec.md` —
> ein *Koordinations*-SDK so wie Attractors Spec ein *LLM*-SDK ist.

---

## §1 Überblick

### §1.1 Zweck

Claude Code spawned jeden `Agent()` als isolierten Subprocess. Jeder
Subprocess bekommt ein frisches Context-Window, kann nicht andere
Agents' Tools aufrufen, kann nicht andere Agents' In-Flight-State
lesen, und returned einen einzelnen Blob an seinen Parent. Das
bricht drei Patterns, die in früheren Claude- / Gemini-Setups
funktioniert haben:

1. **Role-Teleportation** — Persona-Switch in-context ohne Verlust.
2. **Mid-Flight-Coordination** — Agent A sagt Agent B, was er gefunden hat.
3. **Cross-Session-Handoff** — Session A's Arbeit feedet Session B in Echtzeit.

Diese Spec definiert das **Agent-Coordination-MCP**: das Drei-Layer-
Koordinations-Modell, das nötig ist, um diese Patterns wiederherzustellen,
und die File- / Git- / MCP-Primitives, die es implementieren.

### §1.2 Zwei Operating-Modes

| Mode | Substrat | Wann nutzen |
|---|---|---|
| **Native** | Ein zukünftiger MCP-Server der `post_entry` / `read_entries` / `subscribe`-Endpoints exponiert (§5). | Wenn verfügbar — bevorzugt für Low-Latency. |
| **Workaround** | `tee -a`-Markdown-Files in Git + GitHub-PR-Webhooks via `mcp__github__subscribe_pr_activity`. | Heute, in jeder Claude-Code-Session. Hier als kanonischer Fallback definiert. |

Beide Modes implementieren das gleiche Drei-Layer-Modell (§3) mit
den gleichen Schemata (§6). Der Native-Server ist eine dünne
serde-Schicht über dem Wire-Format des Workaround.

### §1.3 Das universelle Wire-Format: `tee -a [bug/proposal]`

Der gleiche Markdown-Blob dient **drei Zwecken gleichzeitig**:

| Zweck | Leser | Mutability |
|---|---|---|
| **MCP-Orchestrations-Message** | der Orchestrator (und jede subscribed Session via `subscribe_pr_activity`) | append-only |
| **A2A-Wire-Format** | Sibling-Agents (Worker → Worker, Worker → Meta, Savant → Savant) | append-only |
| **Log-Format (Audit + Replay)** | jede zukünftige Session, die Git-History liest | immutable once committed |

Ein Write, drei Uses. Das ist by design: derselbe Code, der einen
Eintrag ans File-Blackboard (Layer 1) postet, emittiert AUCH die
Log-Zeile, wird AUCH zu einer cross-session-readable Orchestrations-
Message in dem Moment, in dem es auf dem Branch eines
Coordination-PRs landet.

Die Form des Blobs ist **immer ein typisierter Envelope + ein Body**:

```
## YYYY-MM-DDTHH:MM — KIND[severity]: one-line headline (author, branch)

**Author:** <agent-id | session-id>
**Kind:** BUG | PROPOSAL | HANDOVER | FINDING | DECISION | RFC | STATUS
**Severity:** P0 | P1 | INFO       (optional; default INFO)
**Refs:** <commit-sha> | <PR-#> | <handover-id>
**Proof-of-read:**
- file=<path> sha256=<...> lines=<N> depth=<D>
- file=<path> sha256=<...> lines=<N> depth=<D>

---

<body — frei-form Markdown, aber `## ` Sub-Headings bevorzugt vor Prosa>
```

Der `tee -a`-Envelope ist das Wire; der Body ist die Payload. Router
(Orchestrator, Meta-Agent, MCP-Server) dispatchen nach `Kind` und
`Severity`; Leser greppen nach `Author` + `Refs`; Auditors replayen
in der immutable Git-Log-Order.

### §1.4 Tabellen vor Prosa (NLSpec-Disziplin)

Implementierungen SOLLTEN tabellarischen Content vor Prosa bevorzugen
für jeden Katalog (Anti-Patterns, Validierungs-Regeln,
Verdict-Vokabulare, Konfigurations-Defaults, Parity-Matrizen).
Begründung:

| Eigenschaft | Tabelle | Prosa |
|---|---|---|
| Maschinell parsbar | ja | nein |
| Erzwingt Konsistenz (alle Zeilen haben die gleichen Spalten) | ja | nein |
| Komprimierbar (jede Zeile ist ein Fakt) | ja | nein |
| Diffed sauber in PRs | ja | partial |
| Widersteht hedging Language | ja (Cell-Breite cappt Qualifier) | nein |
| Geeignet für englische Narrative | nein | ja |

Nutze Prosa für: Section-Overviews, Rationale-Absätze, Conflict-
Erklärungen. Nutze Tabellen für: alles andere.

### §1.5 Was diese Spec NICHT ist

- Keine Workflow-Engine. Wave- / Sprint-Struktur ist in
  [`autoattended-orchestrator-spec.md`](./autoattended-orchestrator-spec.md).
- Kein Per-Agent-Loop. Worker-Verhalten ist in
  [`anti-skim-agent-spec.md`](./anti-skim-agent-spec.md).
- Nicht Provider-gekoppelt. Beide Modes sind LLM-Provider-agnostisch.

---

## §2 Terminologie

| Begriff | Definition |
|---|---|
| **Layer-0 Teleportation** | Der Main-Thread lädt eine Agent-Card und „trägt ihren Hut" in-context. Zero Transport. |
| **Layer-1 File-Blackboard** | Eine per-session append-only Markdown-File (`AGENT_LOG.md`), die Workers beim Eintritt lesen und beim Austritt anhängen. |
| **Layer-2 Branch-Pub/Sub** | Der Push-Activity-Feed eines Draft-PRs, genutzt als Real-Time-Cross-Session-Message-Bus. |
| **Handover** | Eine strukturierte Markdown-File (`.claude/handovers/YYYY-MM-DD-HHMM-*.md`), geschrieben am Session-Ende für die nächste Session, beim Start zu lesen. |
| **Blackboard-Entry** | Ein Append an `AGENT_LOG.md`; Schema in §6.1. |
| **Cross-Session-Broadcast** | Eine committed-and-protected File getrennt von `AGENT_LOG.md`; der durable Mirror wichtiger Einträge. |
| **Coordination-PR** | Ein Draft-PR (nie gemerged), dessen Push-Activity der Layer-2-Transport ist. |

---

## §3 Die drei Koordinations-Layer

### §3.1 Layer-0: Teleportation

**Wann:** die Aufgabe braucht den VOLLEN Conversation-Context (keine
Summary) und der Role-Switch ist temporär.

**Wie:** auf dem Main-Thread (NICHT via `Agent()`):

```
1. Read .claude/EN/agents/<role-card>.md
2. Lade ihre Tier-1-Knowledge-Docs (per der Card §Inputs)
3. Mach die Arbeit mit vollem Session-Context intakt
4. Wenn fertig, optional Switch: lies eine andere Role-Card
5. Review die Arbeit aus der Perspektive der anderen Rolle
6. Zurück zur Original-Rolle — nichts verloren
```

**Cost:** Zero Transport, Single-Threaded, Context-Budget füllt
sich, während Role-Cards gelesen werden. Keine Isolation: ein
Fehler gemacht „als Rolle-X" ist für ein nachfolgendes „als Rolle-Y"-
Review sichtbar (Feature, kein Bug).

**Nicht für:** mechanische Grind-Arbeit (nutze Layer-1 + `Agent()`)
oder wirklich independent parallele Arbeit (nutze `Agent()` direkt).

### §3.2 Layer-1: File-Blackboard

**Ersetzt:** Mid-Flight-Coordination (partiell).
**Pfad:** `.claude/board/AGENT_LOG.md` (oder `META/AGENT_LOG.md`
per Repo-Konvention).
**Permission:** vor-erlaubt in `.claude/settings.json` als
`Bash(tee -a *)` und `Bash(tee -a **)`.

**Setup (Native-Mode):**

Ein zukünftiger MCP-Server exponiert die File als strukturierten
Stream (`post_entry`, `read_entries`, `subscribe`). Bis das
existiert, ist der Fallback die Markdown-File direkt.

**Agent-Prompt-Template (Workaround-Mode):**

```
Bevor du anfängst zu arbeiten, lies .claude/board/AGENT_LOG.md
um zu sehen, was andere Agents schon shipped oder gefunden haben.

Nach dem Commit, häng deinen Eintrag an:

tee -a .claude/board/AGENT_LOG.md > /dev/null <<'EOF'

## YYYY-MM-DDTHH:MM — description (model, branch)

**Agent-id:** <id>
**Bundle:** <name>
**Sentinel:** <token>
**Commit:** `abc1234`
**Tests:** N pass (M new)
**Outcome:** One-line summary.
**Proof-of-read:**
- file=<path> sha256=<...> lines=<N> depth=<D>
EOF
```

**Limitierungen:**

- Nicht Real-Time: Agent B sieht nur, was Agent A *committed* hat,
  nicht was A gerade in Arbeit hat.
- Git-Staging: wenn Agent A und B appenden ohne zu committen, gewinnt
  nur das letzte `git add`. Mitigation: sofort nach Append committen.
- Ordering: Einträge erscheinen unten (per `tee -a`); Konvention ist
  newest-first. Main-Thread reordered bei Board-Hygiene.

### §3.3 Layer-2: Branch-Pub/Sub

**Ersetzt:** Cross-Session-Handoff.
**Wie:** öffne einen Coordination-PR. Beide Sessions subscriben.
Push-Events kommen als `<github-webhook-activity>`-Tags via
`mcp__github__subscribe_pr_activity`.

**Setup:**

```bash
# Session A (creates the bus):
git checkout -b claude/blackboard
echo "# Coordination Blackboard" > .claude/board/AGENT_LOG.md
git add .claude/board/AGENT_LOG.md
git commit -m "init coordination blackboard"
git push -u origin claude/blackboard

mcp__github__create_pull_request(
  owner="<O>", repo="<R>",
  title="A2A coordination blackboard",
  head="claude/blackboard", base="main",
  body="Cross-session pub/sub bus. Do not merge.",
  draft=true
)

mcp__github__subscribe_pr_activity(owner="<O>", repo="<R>", pullNumber=NNN)

# Session B (joins):
mcp__github__subscribe_pr_activity(owner="<O>", repo="<R>", pullNumber=NNN)
git fetch origin claude/blackboard
git checkout claude/blackboard
# Read AGENT_LOG.md
```

**Koordinations-Loop:**

```
Session A:                              Session B:
  [does work]
  tee -a AGENT_LOG.md > /dev/null <<EOF ...
  git add && commit && push
                                        ← <github-webhook-activity>
                                        git pull origin claude/blackboard
                                        cat AGENT_LOG.md
                                        [builds on A's findings]
                                        tee -a AGENT_LOG.md ...
                                        git add && commit && push
  ← <github-webhook-activity>
  git pull
  # reads B's entry, continues
```

**Warum es funktioniert:**

- `subscribe_pr_activity` ist schon im MCP-Toolkit — Zero Infra.
- GitHub-Webhooks feuern bei jedem Push, unabhängig vom Inhalt.
- Append-only-Files mergen sauber (kein Konflikt bei concurrenten
  Appends an unterschiedlichen Positionen).
- Der Draft-PR mergt nie — er ist der Bus, nicht ein Deliverable.

**Limitierungen:**

- GitHub-Webhook-Latency: Sekunden bis niedrige Minuten.
- Rate-Limits: GitHub-API 5000/h authenticated.
- Braucht Netzwerk: funktioniert nicht offline.
- PR MUSS offen bleiben: Schließen tötet die Subscription.

### §3.4 Layer-3 (Handover): Strukturierte Handover-Files

**Ersetzt:** Session-zu-Session-Context-Transfer.
**Pfad:** `.claude/handovers/YYYY-MM-DD-HHMM-<from>-to-<to>.md`.

Der `SessionStart`-Hook (`.claude/hooks/session-start.sh`) cat'ed
die letzte Handover-File in den Context der nächsten Session.

Schema ist in §6.3.

---

## §4 Decision-Matrix

| Bedarf | Workaround | Native-MCP-Equivalent | Cost |
|---|---|---|---|
| Agent A's Findings feeden Agent B (gleiche Session) | Layer-1 File-Blackboard | `post_entry` + `read_entries` | Low: `tee -a` + `git add` |
| Session A's Arbeit feedet Session B (Real-Time) | Layer-2 Branch-Pub/Sub | `subscribe` auf Entries | Medium: PR + subscribe |
| Full-Context-Role-Switch (kein Verlust) | Layer-0 Teleportation | n/a (in-process) | Zero: Card lesen reicht |
| Session-zu-Session-Knowledge-Transfer | Handover-Files | `post_handover` + `read_latest_handover` | Low: einmal schreiben, beim Start lesen |
| Parallele independent Grind-Arbeit | Standard `Agent()`-Spawns | n/a | Low: fire and forget |
| Multi-Source-Synthese die Judgment braucht | Teleportation auf Opus-Main-Thread | n/a | Zero |

---

## §5 Native-MCP-Server-Contract

Ein zukünftiger MCP-Server SOLLTE die folgenden Endpoints exponieren.
Sie sind hier spezifiziert, damit das Wire-Format des Workaround-Modes
forward-kompatibel zum Native-Mode ist.

### §5.1 `post_entry`

```yaml
endpoint: post_entry
params:
  board: string                          # z. B. ".claude/board/AGENT_LOG.md"
  entry:
    timestamp: string (ISO 8601)
    agent_id: string
    bundle: string | null
    sentinel: string | null
    commit: string | null                # short SHA
    outcome: enum (SUCCESS | PARTIAL_SUCCESS | RETRY | FAIL | SKIPPED)
    summary: string
    proof_of_read: list of ProofOfRead   # siehe §6.2
returns:
  entry_id: string
  position: integer
errors:
  - INVALID_OUTCOME
  - SENTINEL_MISMATCH
  - PROOF_OF_READ_MISSING_REQUIRED
```

### §5.2 `read_entries`

```yaml
endpoint: read_entries
params:
  board: string
  since: string (entry_id) | null
  limit: integer (default 100, max 1000)
  filter:
    agent_id: string | null
    outcome: enum | null
    bundle: string | null
returns:
  entries: list of BlackboardEntry
  next_cursor: string | null
```

### §5.3 `subscribe`

```yaml
endpoint: subscribe
params:
  board: string
  cursor: string | null                  # Start-Position
streams:
  type: server_sent_events
  event_kinds:
    - entry_appended
    - entry_redacted
    - board_archived
returns_initial:
  cursor: string
errors:
  - BOARD_NOT_FOUND
  - SUBSCRIPTION_LIMIT_EXCEEDED
```

### §5.4 `post_handover` / `read_latest_handover`

```yaml
endpoint: post_handover
params:
  from_session: string
  to_session: string | null              # null = „nächste Session, wer auch immer"
  handover: Handover                     # siehe §6.3
returns:
  handover_id: string

endpoint: read_latest_handover
params:
  to_session: string                     # current session id
returns:
  handover: Handover | null
```

### §5.5 Authentifizierung

Der Native-Server SOLLTE das gleiche Auth-Substrat wie der GitHub-
MCP-Server nutzen (per-session OAuth oder PAT). Workaround-Mode
nutzt Git-Push-Permissions auf dem Branch des Coordination-PRs.

---

## §6 Schemata

### §6.1 BlackboardEntry

```json
{
  "entry_id": "01HXYZ-WAVE12-A4-SUCCESS",
  "timestamp": "2026-05-17T17:42:00Z",
  "agent_id": "A4",
  "bundle": "customer-master-data",
  "sentinel": "WAVE-12-A4-7f3c",
  "commit": "a1b2c3d4",
  "outcome": "SUCCESS",
  "summary": "customer-master portiert; 7 Parity-Tests grün; keine Iron-Rule-Verletzungen.",
  "proof_of_read": [
    { "file": "META/INVARIANTS.md", "sha256": "...", "lines": 412, "depth": "thorough" }
  ]
}
```

### §6.2 ProofOfRead

```json
{
  "file": "path/relative/to/repo-root",
  "sha256": "64-hex",
  "lines": 412,
  "depth": "thorough"
}
```

`depth`-Werte: `grep` | `sed-partial` | `skim` | `read` | `thorough`
| `troubleshooting` | `fan-out` | `truncated:head-N` | `truncated:tail-N`.

### §6.3 Handover

```yaml
handover:
  id: 2026-05-17-2200-wave12-to-wave13
  from_session: 2026-05-17-evening
  to_session: null                       # any next session
  topic: customer-master-port

  what_i_did:
    - bullet: 12 Bundles (A1..A12) für customer-master portiert
      commits: [a1b2c3, d4e5f6, ...]

  finding:
    - bullet: BBB-Barrier-Compile-Error tritt bei A5 auf wenn sea-orm 0.12 im Path
      severity: P0
      source: META/INVARIANTS.md §BBB

  conjecture:
    - bullet: Switch zu sea-orm 0.13 könnte den Migrations-Path brechen; braucht Probe
      severity: P1

  blockers:
    - description: A5 kann nicht gemerged werden bis der sea-orm-RFC akzeptiert ist
      filed_in: META/REQUESTS-FROM-AGENTS.md#A5

  open_questions:
    - "Sollte A11 ein separater PR sein oder in die Consolidation rolled?"

  proof_of_read:
    - file: META/INVARIANTS.md
      sha256: "..."
      lines: 412
      depth: thorough
```

### §6.4 RequestEntry (`META/REQUESTS-FROM-AGENTS.md`)

```yaml
request:
  agent_id: A5
  file: src/customer/master/sea_orm.rs
  timestamp: 2026-05-17T18:14:32Z

  question: |
    Das Reference-Python nutzt ein idempotentes UPSERT, aber der
    sea-orm-Builder den ich habe macht INSERT-OR-FAIL. Spec ist
    mehrdeutig.

  tried:
    - sea-orm `on_conflict().do_nothing()` — failt mit `Error: ...`
    - manueller Transaction-Wrap — funktioniert aber verdoppelt Latency

  blocker: AMBIGUITY

  proof_of_read:
    - file: vendor/ogit/NTO/WorkOrder/entities/customer.ttl
      sha256: "..."
      lines: 137
      depth: full
    - file: ../WoA/woa/blueprints/customer.py
      sha256: "..."
      lines: 891
      depth: full
```

### §6.5 AnswerEntry (`META/ANSWERS-TO-AGENTS.md`)

```yaml
answer:
  agent_id: A5
  in_response_to: 2026-05-17T18:14:32Z

  decision: |
    Nutze sea-orm-Transaction-Wrap. Der Latency-Cost ist akzeptabel.

  propagation:
    invariants_section: META/INVARIANTS.md §UPSERT-PATTERN added
    rfcs: null

  signed_off_by: meta-agent (auto)
  timestamp: 2026-05-17T18:42:00Z
```

---

## §7 Append-only-Governance

### §7.1 Mutability-Invariante

Über alle Coordination-Files hinweg:

| File | Mutability |
|---|---|
| `Stand.md` / `STATUS_BOARD.md` | Mutable (die EINZIGE). |
| `AGENT_LOG.md` | Append-only. |
| `REQUESTS-FROM-AGENTS.md` | Append-only. |
| `ANSWERS-TO-AGENTS.md` | Append-only. |
| `INVARIANTS.md` | Append-only nach Section; Sections können SUPERSEDED markiert werden, aber nie gelöscht. |
| Handover-Files | Immutable nach Write. |
| Cross-Session-Broadcast-Files (`PR_ARC_INVENTORY.md`, `LATEST_STATE.md`, `EPIPHANIES.md`, `IDEAS.md`, `INTEGRATION_PLANS.md`, `ISSUES.md`, `TECH_DEBT.md`, `STATUS_BOARD.md`) | Append-only auf Entry-Level; spezifische Felder sind mutable State. |

### §7.2 Enforcement

Implementierungen SOLLTEN Append-only-Governance auf der
Permissions-Schicht durchsetzen. Die lance-graph-Referenz (siehe
[`AdaWorldAPI/lance-graph` `.claude/settings.json`](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/settings.json))
denied `Edit` und `Write` auf den 8 Bookkeeping-Files und erlaubt
nur `Bash(tee -a *)`. Das ist das empfohlene Pattern.

---

## §8 Relation zu Runtime-A2A (Layer 1, In-Process)

Wenn ein Projekt auch ein In-Process-Blackboard in seiner eigenen
Runtime fährt (z. B. `lance_graph_contract::a2a_blackboard::Blackboard`),
sind die Session-Level-Workarounds strukturell isomorph dazu:

| Runtime (in-process) | Session (diese Spec) |
|---|---|
| `Blackboard.entries` | `AGENT_LOG.md`-Entries |
| `BlackboardEntry.expert_id` | Agent-description + Model |
| `BlackboardEntry.capability` | Bundle / D-ids |
| `BlackboardEntry.result` | Commit-Hash + Outcome |
| `BlackboardEntry.confidence` | Tier-1-Gate-Pass-Count |
| `Blackboard.round` | Git-Commit-Sequence |
| Experts lesen prior Rounds | Agents lesen prior Log-Entries |

Die Isomorphie ist absichtlich: das gleiche Koordinations-Pattern
funktioniert auf beiden Layern, weil das Problem dasselbe ist —
unabhängige Experten komponieren Resultate auf einem geteilten
Substrat.

---

## §9 Definition von Fertig

Eine Implementierung ist konform, wenn sie ALLE erfüllt:

- [ ] Drei Layer (Teleport / Blackboard / Branch-Pub/Sub) sind
      implementierbar mit den existierenden Primitives (File + Git +
      MCP `subscribe_pr_activity`).
- [ ] Native-MCP-Endpoints §5.1–§5.4 existieren ODER der
      Workaround-Mode ist als expliziter Fallback dokumentiert.
- [ ] `BlackboardEntry`, `ProofOfRead`, `Handover`, `RequestEntry`,
      `AnswerEntry`-Schemata sind implementiert und validiert.
- [ ] Append-only-Governance (§7) ist auf der
      `.claude/settings.json`-Schicht für die 8 Bookkeeping-Files
      durchgesetzt.
- [ ] Die Single-Mutable-File-Invariante ist durchgesetzt: `Stand.md`
      / `STATUS_BOARD.md` ist die EINZIGE File, die Workers
      überschreiben dürfen.
- [ ] Decision-Matrix (§4) wird befolgt: Workers nutzen den
      richtigen Transport für die richtige Art Message.
- [ ] Coordination-PRs sind Draft, benamt `claude/<topic>`, und
      explizit als `Do not merge` im Body markiert.
- [ ] Handover-Files nutzen das §6.3-Schema mit Required Sections:
      What-I-did / FINDING / CONJECTURE / Blockers / Open-Questions
      / Proof-of-Read.

---

## §10 Cross-Provider-Parity-Matrix

| Fähigkeit | Claude Code | OpenAI Codex / codex-rs | Gemini CLI | Notizen |
|---|---|---|---|---|
| Layer-0 Teleportation | Read Agent-Card auf Main-Thread | Same | Same | Provider-agnostisch; hängt nur von File-Read ab. |
| Layer-1 File-Blackboard | `Bash(tee -a)` + `git add` | Same | Same | POSIX-only; Provider-agnostisch. |
| Layer-2 Branch-Pub/Sub | `mcp__github__subscribe_pr_activity` | `gh pr`-Polling | `gh pr`-Polling | Claude Code hat natives MCP; andere pollen. |
| Native-MCP-Server (§5) | TBD (diese Spec) | TBD | TBD | Alle drei Provider können MCP sprechen. |
| Append-only-Governance | `.claude/settings.json`-Deny-Liste | Provider-spezifische Permission-File | Provider-spezifisch | Pattern überträgt sich; Allow-List-Syntax unterscheidet sich. |
| Handover-Files | `.claude/hooks/session-start.sh` cat'ed letzte | Equivalent Shell-Hook | Equivalent Shell-Hook | Hook-Script ist Bash; Provider-agnostisch. |

---

## §11 Appendix A — Sample `AGENT_LOG.md`-Eintrag (verbatim)

```markdown
## 2026-05-17T17:42 — customer-master ported; 7 parity tests green (sonnet, claude/wave-12-customer-master)

**Agent-id:** A4
**Bundle:** customer-master-data
**Sentinel:** WAVE-12-A4-7f3c
**Commit:** `a1b2c3d4`
**Tests:** 7 pass (3 new)
**Outcome:** SUCCESS — alle Tier-1-Gates grün; keine Iron-Rule-Verletzungen.
**Proof-of-read:**
- file=META/INVARIANTS.md sha256=fa39a3... lines=412 depth=thorough
- file=../WoA/woa/blueprints/customer.py sha256=8e1a45... lines=891 depth=full
- file=vendor/ogit/NTO/WorkOrder/entities/customer.ttl sha256=4c0b... lines=137 depth=full
```

---

## §12 Appendix B — Wann NICHT Coordination nutzen

Wenn die Aufgabe ist...

- wirklich independent und parallel → nutze Standard-`Agent()`-Spawns; kein Coordination-Layer nötig.
- eine einzelne One-Shot-Frage → nutze `Agent()` mit sub-agent_type=Explore; kein `AGENT_LOG.md`-Write.
- ein security-sensitives Review → nutze Layer-0-Teleportation auf dem Main-Thread; NICHT einen separaten Agent() spawnen, der von einer untrusted Source kompromittiert werden könnte, die er liest.

---

## §13 Cooperative-Savant-Scratchpad-Bus

> Hinzugefügt 2026-05-18. Die Transport-Schicht für den Cooperative
> Savant Council, definiert in `autoattended-orchestrator-spec.md` §15.

### §13.1 Was der Scratchpad-Bus ist

Eine zweck-geformte Erweiterung des Layer-1 File-Blackboard (§3.2),
genutzt spezifisch für die iterative Cooperation des Savant-Councils.
Gleiches Wire-Format (`tee -a`-Markdown), gleiche Append-only-
Governance (§7), eine neue Directory-Konvention:

```
.claude/board/savant-council-<topic>/
├── ROUND-0-ARTIFACT.md                  Snapshot des Artifacts unter Review
├── ROUND-<N>-<savant>.md                eine File pro (Round, Savant)
└── COUNCIL-VERDICT.md                   Chairman-Synthese nach CONVERGE
```

Topic-Naming: `savant-council-<sprint-id>` (z. B.
`savant-council-SPRINT-17-attractor-fold-in/`).

### §13.2 Per-Savant-File-Schema

```markdown
# ROUND-{N} {savant_id} — {topic}

**Round:** N
**Savant:** PP-13 | PP-14 | PP-15 | PP-16
**Read at round start:** [Liste der diese Round gelesenen Peer-Files]
**Status:** IN-PROGRESS | ROUND-COMPLETE

## New findings (this round)

### F{round}-{savant}-{N}: {one-line headline}
**Axis:** AP1..AP9 | BAP1..BAP10 | PD1..PD10 | EP1..EP8
**Severity:** P0 | P1
**File:** <path>:<line>
**Detail:** <ein Absatz>
**Remediation:** <ein Absatz>

## Cross-references to peers' findings

- F{R}-PP-X-N: NOTED — mein Winkel: <one-line>
- F{R}-PP-Y-N: SUPERSEDED-BY meinem F{R+1}-{self}-M

## Withdrawals (meine prior Findings, die ich zurückziehe)

- F{R-1}-{self}-N: WITHDRAWN — covered by PP-X round R aus stärkerem Winkel
```

### §13.3 Reading-and-Writing-Protokoll

Jeder Savant in jeder Round:

1. **Lies** ALLE `ROUND-(R-1)-*.md`-Files (Peer-Findings aus der
   vorigen Round) plus die eigene `ROUND-(R-1)-<self>.md` falls vorhanden.
2. **Entscheide** Cross-Refs, Withdrawals, neue Findings.
3. **Schreib** `ROUND-R-<self>.md` in einem `tee -a`-Call (keine
   partial Writes — Savants teilen sich ein Directory und partial
   Writes verwirren die nächste Peer-Round).
4. **Emittiere** `Status: ROUND-COMPLETE` wenn keine neuen Findings
   UND keine un-addressed Cross-Refs von Peers.

### §13.4 Chairman-Synthese

Nach CONVERGE (alle vier Savants `ROUND-COMPLETE` in der gleichen
Round) liest der Chairman-Savant (deklariert in Sprint-Config;
default PP-16 für Protokoll A, PP-15 für Protokoll B) ALLE
ROUND-*.md-Files in lexikalischer Ordnung und schreibt eine
einzelne `COUNCIL-VERDICT.md` matching das Schema in
`autoattended-orchestrator-spec.md` §15.5.

Der Chairman:
- MUSS jede gehaltene Finding mit ihrem raised-by + Cross-Referrals
  zitieren.
- MUSS Duplikate konsolidieren, die die Cooperation-Rounds
  überlebten (selten — die meisten sollten sich selbst zurückziehen,
  aber Races können ein Paar hinterlassen, das in eines gemerged werden muss).
- DARF NICHT eigene Findings hinzufügen, die in keinem `ROUND-*.md`
  auftauchten (der Chairman ist ein Synthesizer, keine fünfte Stimme).
- MUSS `super_verdict` per §15.5-Regeln setzen.

### §13.5 Concurrency-Contract

Savants können in separaten Sessions laufen; der Scratchpad-Bus ist
der Synchronisations-Punkt.

- **Round-Boundary** ist implicit: ein Savant startet Round R erst,
  nachdem er alle vier `ROUND-(R-1)-*.md`-Files (inklusive seiner
  eigenen) gelesen hat.
- **Last-Writer-Wins** ist akzeptabel für die Per-Savant-Per-Round-
  File, da jedes (Savant, Round)-Paar by construction genau einen
  Writer hat. Wenn ein Savant zweimal in derselben Round schreibt
  (z. B. ein Retry nach Tool-Call-Loop), ersetzt der zweite Write
  den ersten.
- **Round-Skipping** ist verboten: ein Savant DARF NICHT
  `ROUND-(R+2)` schreiben, bevor `ROUND-(R+1)` für ihn existiert.

### §13.6 Decision-Matrix-Update

Erweitert §4 mit einer neuen Zeile:

| Bedarf | Workaround | Native-MCP-Equivalent | Cost |
|---|---|---|---|
| Cooperative Multi-Agent-Review mit iterativem Cross-Refer + Withdraw | Layer-1 File-Blackboard scoped auf `savant-council-<topic>/`-Directory (§13) | `post_council_finding` + `read_council_round` + `subscribe_council_round` | Medium: `tee -a` pro Round + Git-Commit pro Round |

### §13.7 Native-MCP-Endpoints (Sketch)

```yaml
endpoint: post_council_finding
params:
  topic: string                          # z. B. "SPRINT-17-attractor-fold-in"
  round: integer
  savant: enum (PP-13 | PP-14 | PP-15 | PP-16)
  payload: FindingsFile                  # matched §13.2-Schema
returns:
  file_id: string

endpoint: read_council_round
params:
  topic: string
  round: integer
returns:
  findings_by_savant: map[savant -> FindingsFile]
  complete_savants: list[savant]         # die, die ROUND-COMPLETE emittiert haben

endpoint: subscribe_council_round
params:
  topic: string
streams:
  event_kinds: [ findings_appended, savant_round_complete, council_converged, council_stalled, council_blocked ]
```

### §13.8 Validierungs-Regeln

| Regel | Beschreibung | Severity |
|---|---|---|
| `COORD-001 directory-naming` | Council-Scratchpad MUSS unter `.claude/board/savant-council-<topic>/` liegen. | ERROR |
| `COORD-002 file-naming` | Per-Savant-Files MÜSSEN `ROUND-<N>-PP-<NN>.md` benamt sein. | ERROR |
| `COORD-003 append-only` | Das Scratchpad-Directory erbt die Append-only-Governance von §7. | ERROR |
| `COORD-004 chairman-no-new-findings` | Die `COUNCIL-VERDICT.md` des Chairmans DARF NICHT Findings einführen, die in keinem `ROUND-*.md` vorkommen. | ERROR |
| `COORD-005 round-monotonic` | Jeder Savant MUSS `ROUND-(R+1)` erst schreiben, nachdem `ROUND-R-<self>.md` existiert. | ERROR |

---

*Ende der Datei agent-coordination-mcp-spec.md.*
