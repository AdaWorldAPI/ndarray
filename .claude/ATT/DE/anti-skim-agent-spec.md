# Anti-Skim-Agent-Spezifikation

> **Sprache:** Deutsch · siehe `../anti-skim-agent-spec.md` für die englische Quellfassung.
>
> **Status:** DRAFT  ·  **Version:** 0.1.0  ·  **Stand:** 2026-05-17
> **Format:** NLSpec (nach [strongdm/attractor](https://github.com/strongdm/attractor))
> **Substrat:** Komponiert auf `coding-agent-loop-spec.md`; ergänzt die
> Verifikations- und Eskalations-Schicht, die Attractors Spec auslässt.

---

## §1 Überblick

### §1.1 Zweck

Der Anti-Skim-Agent ist ein Per-Worker-LLM-Loop, der **verifizierbar
lesen-vor-handeln** macht. Er behandelt **Overconfidence** des Agents
als primären Failure-Mode (vgl. Kahneman / Tversky System-1-Easy-Path,
Dunning-Kruger) und verschiebt die Beweislast auf den Agent:

- Jede Behauptung über den Inhalt einer Datei MUSS mit einer
  SHA-256 + Line-Count + Reading-Depth-Deklaration belegt sein.
- Jedes Brief enthält ein Sentinel-Token, das der Agent literal
  zurückspielen MUSS.
- Ein Stichproben-Check besteht je Supervisor-Pass einen der fünf
  Lie-Detector-Tests.

### §1.2 Relation zu `coding-agent-loop-spec.md`

Der Anti-Skim-Agent ist als Coding-Agent-Loop-Session implementierbar,
mit den Ergänzungen in §3–§6. Konkret:

- Der `Session.run()`-Loop (coding-agent §2.5) wird mit einer
  Proof-of-Read-Vorbedingung umwickelt (§7).
- Die `steering_queue` (coding-agent §2.6) wird um die typisierten
  Stuck-Protocol-Blocker aus §5 erweitert.
- Der Tool-Call-Loop-Detector aus coding-agent §2.10 wird wortgleich
  übernommen und auf System-Level-Invariante gehoben (§6).
- Tool-Output-Truncation (coding-agent §5) wird übernommen mit einer
  Ergänzung: getrunkierte Reads MÜSSEN im Proof-of-Read mit
  `depth=truncated` auftauchen, NICHT `depth=full`.

### §1.3 Was diese Spec NICHT ist

- Kein Orchestrator. Wave-Level-Koordination ist in
  [`autoattended-orchestrator-spec.md`](./autoattended-orchestrator-spec.md).
- Kein Per-Agent-Loop. Worker-Verhalten ist in
  [`anti-skim-agent-spec.md`](./anti-skim-agent-spec.md).
- Kein Transport. Cross-Agent-Messages laufen über das File-Blackboard
  + Handover-Files in
  [`agent-coordination-mcp-spec.md`](./agent-coordination-mcp-spec.md).

---

## §2 Terminologie

| Begriff | Definition |
|---|---|
| **Sentinel-Token** | Ein eindeutiger, opaker String im Worker-Brief, den der Worker literal in seiner ersten Antwort zurückspielen MUSS. |
| **Proof-of-Read** | Ein typisiertes Eintrags-Tupel `{ file, sha256, lines, depth }`, das der Worker für jede konsumierte Datei deklariert. |
| **Reading-Depth** | Eine von sieben eskalierenden Stufen: `grep` (anti) / `sed-partial` (anti) / `skim` / `read` / `thorough` / `troubleshooting` / `fan-out`. |
| **Lie-Detector-Test** | Einer von fünf Tests (LD-1..5), den der Meta-Agent / Orchestrator stichprobenartig auf den Worker-Output anwendet. |
| **Drift-Signal** | Ein passiver Indikator für Skim oder Overconfidence (§4.3). |
| **Stuck-Protocol-Blocker** | Ein typisierter Eintrag in `META/REQUESTS-FROM-AGENTS.md` (einer von fünf Typen), der signalisiert „ich komme nicht weiter". |
| **Tool-Call-Loop** | Ein wiederholendes Pattern der Länge 1, 2 oder 3 in den letzten 10 Tool-Calls des Workers. |

---

## §3 Die Reading-Depth-Ladder

### §3.1 Sieben Tiefen, zwei Anti-Tiefen

| Depth | Wann angemessen | Proof |
|---|---|---|
| `grep` (anti) | Symbol-Locator NACH oder PARALLEL zu einem echten Read. | Worker deklariert `depth=grep`; NIEMALS `depth=full`. |
| `sed-partial` / `head-only` (anti) | Snapshot weniger Zeilen zur Orientierung. | Worker deklariert `depth=sed-partial`; NIEMALS `depth=full`. |
| `skim` | Riesige Datei; eine Section lokalisieren, dann voll lesen. | Worker macht NUR Aussagen über die gelesene Section. |
| `read` | Datei < ~500 Zeilen; Standard-Full-Read. | Top to Bottom, kein Skip; bei größeren Files: offset/limit-Chunks aber jeder Chunk tatsächlich gelesen. |
| `thorough` | Iron Rules, INVARIANTS, RFCs, Schema-Migrationen. | Zweimal lesen: einmal Verständnis, einmal Verifikation. Worker kann 3 Sections namentlich nennen. |
| `troubleshooting` | Bekannter Bug + Error-Message. | Error → grep Symbol → READ Funktion voll → READ Caller(s). |
| `fan-out` | Cross-File-Pattern, Refactor-Planung, Audit über > 5 Dateien. | Inventur-File als Output; Explore-Subagent spawnen. |

### §3.2 Eskalations-Default

**Default-Tendenz: lieber zu tief lesen als zu flach.** Bei Zweifel:
Depth eine Stufe upgraden; niemals downgraden.

### §3.3 Wenn → minimale Tiefe

| Wenn der Worker im Begriff ist ... | ... dann mindestens |
|---|---|
| `INVARIANTS.md`, `CLAUDE.md`, `BOOT.md` öffnen | `thorough` |
| Eine Memory-File (`CONTEXT.md`, `JOURNAL.md`, `TODO.md`) öffnen | `thorough` |
| Eine Schema- oder Migrations-Datei anfassen | `thorough` + Downstream-Drift-Detector mit prüfen |
| Eine unbekannte Datei zum ersten Mal öffnen | mindestens `read`, eher `thorough` |
| Reine Symbol-Lookup („wo ist `foo` definiert?") | `grep` OK aber dann Definition voll lesen |
| Bug-Bericht triagieren | `troubleshooting` (Error → grep → READ Funktion + Caller) |
| Refactor / Wave / Audit planen | `fan-out` (Inventur-File Pflicht) |
| Unsicher welche Depth | eine Stufe upgraden |

### §3.4 Interaktion mit Truncation

Wenn `coding-agent-loop-spec` einen Tool-Output truncated (per
coding-agent §5.2), MUSS der Proof-of-Read-Eintrag des Workers für
diese Datei `depth=truncated:head-N` oder `depth=truncated:tail-N`
deklarieren — NIEMALS `depth=full`.

---

## §4 Der Lie-Detector

### §4.1 Fünf Tests

Billig zu teuer:

| # | Test | Wie | Ehrlicher Agent | Flacher Agent |
|---|---|---|---|---|
| LD-1 | **Sentinel-Token** | Brief endet mit: „Wenn du dies vollständig gelesen hast, beginne deine erste Antwort mit `<TOKEN>`" | spielt Token literal zurück | Token fehlt / falsch / paraphrasiert |
| LD-2 | **Proof-of-Read mit SHA** | Output muss enthalten `{ file, sha256, lines, depth }` pro konsumierter Datei | SHA + Line-Count matchen die echte Datei | SHA fehlt, falsch, oder `<computed>`-Platzhalter |
| LD-3 | **3-Sections-Namens-Challenge** | „Nenne 3 Sections aus File X namentlich (Heading + ungefähre Zeilen-Spanne)" | 3 konkrete Headings, plausible Spannen | vage Themen-Labels; keine echten Headings |
| LD-4 | **Negative-Knowledge-Test** | „Steht in File X etwas zu Thema Y?" — wobei Y NICHT drin ist | „nein, nicht enthalten" | halluziniert plausibel klingenden Inhalt zu Y |
| LD-5 | **Line-Range-Quote** | „Quote Zeilen N-M aus File X verbatim" | exakter Quote ODER „Range existiert nicht (File nur K Zeilen)" | paraphrasiert, weicht ab, oder verweigert ohne Grund |

### §4.2 Trigger-Heuristik

| Worker-Antwort-Pattern | Lie-Detector |
|---|---|
| Vage („nicht sicher", „ungefähr 60", „etwa 5 Sections") | KEIN Trigger — ehrliche Unsicherheit. |
| Selbstbewusst **falsch** | FEUERT — Overconfidence + Drift. |
| Selbstbewusst **richtig** ohne Proof-of-Read | FEUERT — Glück und Halluzination sind ohne LD-1..5 nicht unterscheidbar. |

### §4.3 Drift-Signale (passive Detection, Meta-Agent-Pflicht)

Der Meta-Agent scannt JEDEN Worker-Status vor dem Goal-Gate-Verdict.
Wenn ein Signal triggert, MUSS der Worker re-dispatched werden ODER
ein Lie-Detector-Test angewendet werden:

| Signal | Aktion |
|---|---|
| `depth=grep` deklariert bei semantischer Aufgabe | Re-Dispatch mit Pflicht-`depth=full`. |
| `depth=sed-partial` deklariert bei semantischer Aufgabe | Re-Dispatch; SHA-256 über die ganze Datei verlangen. |
| `depth=full` deklariert ohne `sha256` im Proof-of-Read | Stop; Worker MUSS SHA nachliefern. |
| Status-`notes` zitiert eine nicht-existente Section einer Datei | LD-3-Äquivalent fehlgeschlagen post-facto → P0 Re-Dispatch. |
| Status-`notes` ist wortgleich zu einem Sibling-Agent ohne Cross-Read-Vermerk | Lockstep-Drift; Stichprobe LD-5 auf einen der beiden. |
| Eröffnungs-Output zählt eine spezifische Zahl („62 Lehren") OHNE Proof-of-Read auf der Quelle | Overconfidence-Trigger; Agent MUSS SHA + 3 Einträge namentlich nachliefern. |
| Status fasst eine INVARIANT an, aber `proof_of_read` enthält `INVARIANTS.md` nicht | Mandatory `thorough` Re-Read. |

### §4.4 Beweislast-Regel

Bei Verdacht: der Worker MUSS aufrüsten. Niemals den Verdacht
„wegerklären" — entweder Proof nachliefern, oder ehrlich
downgraden (`depth=skim`/`grep` mit Notiz).

---

## §5 Stuck-Protokoll

### §5.1 Fünf typisierte Blocker

| Typ | Bedeutung |
|---|---|
| `AMBIGUITY` | Spec ist mehrdeutig. Mehr als eine sinnvolle Interpretation. |
| `MISSING_INVARIANT` | Eine Iron Rule sollte den Fall abdecken, tut es aber nicht; die Konvention fehlt. |
| `SPEC_SOURCE_MISMATCH` | Die autoritative Spec sagt X; die Reference-Source tut Y. |
| `BEHAVIOUR_QUESTION` | Möglicher Bug in der Reference-Source; Behaviour-Divergenz braucht RFC. |
| `EXTERNAL_DEPENDENCY` | Drittsystem zickt; Workaround unklar. |

### §5.2 Das Protokoll

Wenn stuck, der Worker MUSS:

1. STOP schreiben von Code.
2. EINEN Eintrag an `META/REQUESTS-FROM-AGENTS.md` anhängen mit
   dem §5.3-Schema.
3. Sein `status.json` auf `outcome=RETRY` und `notes=stuck:<typ>`
   setzen.
4. Idle bis Meta-Agent in `META/ANSWERS-TO-AGENTS.md` schreibt
   ODER `META/INVARIANTS.md` aktualisiert.

### §5.3 Request-Schema

```markdown
## Agent {{agent_id}}, file <path>, timestamp <ISO 8601>

### Question
<ein Absatz>

### Tried
<was probiert wurde, was fehlschlug>

### Blocker
<eines von: AMBIGUITY | MISSING_INVARIANT | SPEC_SOURCE_MISMATCH | BEHAVIOUR_QUESTION | EXTERNAL_DEPENDENCY>

### Proof-of-read attached
- file=<path-A> sha256=<...> lines=<...> depth=<...>
- file=<path-B> sha256=<...> lines=<...> depth=<...>
```

Der Proof-of-Read MUSS JEDE Datei abdecken, die der Worker in der
Question referenziert. Stuck-Requests ohne Proof-of-Read werden bei
Intake zurückgewiesen (§7.5 in `autoattended-orchestrator-spec.md`).

### §5.4 Verbotene Alternativen

Der Worker DARF NICHT:

- Raten.
- Einen `TODO:`-Kommentar in Code schreiben und weitermachen.
- Außerhalb des Bundles refaktorieren „während er wartet".
- Den gleichen Request zweimal ohne neue Information filen.

---

## §6 Tool-Call-Loop-Detection

### §6.1 Übernahme von Attractor coding-agent §2.10

Ein Worker MUSS seine eigenen letzten `N=10` Tool-Calls (Signature =
Name + Arguments-Hash) auf wiederholende Patterns der Länge 1, 2
oder 3 scannen.

```
Pattern-Länge 1:  [A][A][A]                — gleiches Tool, gleiche Args, 3+ Mal
Pattern-Länge 2:  [A][B][A][B]              — alternierend, 2+ Zyklen
Pattern-Länge 3:  [A][B][C][A][B][C]        — 3-Zyklus, 2+ Zyklen
```

### §6.2 Aktion bei Detektion

Bei Detektion:

1. Der Worker DARF NICHT den nächsten vorhergesagten Call im Pattern
   ausgeben.
2. Der Worker MUSS einen typisierten RETRY-WITH-STEER-Status emittieren:
   `outcome=RETRY`, `notes=tool-call-loop:length=<L>:pattern=<sig>`.
3. Der Worker MAG einen `META/REQUESTS-FROM-AGENTS.md`-Eintrag unter
   dem `AMBIGUITY`-Blocker-Typ (§5.1) filen, wenn der Loop keine
   offensichtliche externe Auflösung hat.

### §6.3 Anti-Pattern AP9

Diese Regel wird auch von PP-13 brutally-honest-tester als
Anti-Pattern AP9 (tool-call-loop) durchgesetzt. PP-13s Post-hoc-
Detection greift, wenn der In-Loop-Detector versagt hat.

---

## §7 Proof-of-Read-Schema

### §7.1 Per-File-Eintrag

```json
{
  "file": "path/relative/to/repo-root.md",
  "sha256": "abc...64-hex...def",
  "lines": 412,
  "depth": "thorough",
  "comment": "Two-Pass-Read; INVARIANTS §3.1-3.4 im Worker-Output zitiert"
}
```

### §7.2 Erlaubte Depth-Werte

```
grep         sed-partial      skim
read         thorough         troubleshooting
fan-out      truncated:head-N truncated:tail-N
```

`grep`, `sed-partial`, `truncated:*` DÜRFEN NICHT mit Behauptungen
kombiniert werden, die volles Wissen über die Datei implizieren
(LD-3 / LD-4 / LD-5).

### §7.3 Multi-File-Einträge

Workers MÜSSEN einen Eintrag pro konsumierter Datei emittieren. Der
Orchestrator-Goal-Gate (PP-13) DARF bis zu drei Einträge pro Worker
pro Wave per LD-2 stichprobenartig prüfen (SHA neu berechnen,
Line-Count vergleichen).

### §7.4 Speicherung

Proof-of-Read-Einträge leben in der `status.json` des Workers
(per `autoattended-orchestrator-spec.md` §9.1). Sie werden auch
nach `META/AGENT_LOG.md` als One-Line-Summary nach jedem gelandeten
Chunk gespiegelt:

```
## YYYY-MM-DDTHH:MM — A<id> <file> chunk N/M
- chunk N landed (sha <short>)
- typecheck: green | <error count>
- proof-of-read: <file> sha=<short> lines=<N> depth=<D>
```

### §7.5 Sentinel-Token-Speicherung

Das Sentinel-Token wird aus der ersten Reply-Zeile des Workers in
das Feld `sentinel_token` der `status.json` gespiegelt. Mismatch =
FAIL.

---

## §8 Toolchain-Tiers

### §8.1 Tier 1 (jeder PR — verpflichtend)

Owned by PP-13 brutally-honest-tester. Per-Language-Adapter:

| Zweck | Rust | Python | TypeScript | Go |
|---|---|---|---|---|
| Lint, keine Warnings | `cargo clippy --all-targets --all-features -- -D warnings` | `ruff check` | `eslint --max-warnings 0` | `golangci-lint run` |
| Formatter-Check | `cargo fmt --check` | `ruff format --check` | `prettier --check` | `gofmt -l` |
| Advisory-Scan | `cargo audit` | `pip-audit` | `npm audit --omit=dev` | `govulncheck ./...` |
| Dep-Policy / Vet | `cargo deny check` | `deptry .` | (Projekt-definiert) | `go vet -all` |
| Typecheck | (Clippy impliziert) | `mypy --strict` | `tsc --noEmit --strict` | (Compiler impliziert) |
| Tests | `cargo test --all-features` | `pytest` | `vitest run` | `go test ./...` |

### §8.2 Tier 2 (Quality / Maintenance — Opt-in)

| Zweck | Rust | Python | TypeScript |
|---|---|---|---|
| Unused-Dep-Detector | `cargo machete` | `deptry .` | `depcheck` |
| Unsafe-Scan | `cargo geiger` | `bandit` | (n/a) |
| Public-API-SemVer-Compat | `cargo semver-checks check-release` | `griffe check` | `api-extractor` |
| Spellcheck | `cargo spellcheck` | CSpell | CSpell |

### §8.3 Tier 3 (Heavier — Opt-in, alle stable)

| Zweck | Rust | Python |
|---|---|---|
| Bounded Model Checker | `kani` | (none standard) |
| Concurrency Model Checker | `loom` (lib) | (none standard) |
| Mutation Testing | `cargo mutants` | `mutmut` |
| Coverage | `cargo tarpaulin` | `pytest --cov` |

### §8.4 Tier-1-Invariante

Für jeden PR, jede Sprache: der Worker beweist, dass er ein
No-Warning-Gate erfüllt BEVOR der Orchestrator den PR öffnet.
Tier-1-Failure mappt direkt auf `outcome=FAIL` in der Status-File
(§9.2 von `autoattended-orchestrator-spec.md`).

---

## §9 Anti-Pattern-Katalog (AP1..AP9)

PP-13 brutally-honest-tester jagt diese aktiv:

| # | Anti-Pattern | Detektion | Finding |
|---|---|---|---|
| AP1 | Silent Fallback der Errors schluckt (`unwrap_or_default()`, bare `try { ... } catch {}`, `result, _ := f()`) | grep PR-Diff nach diesen Idiomen | P0 wenn im Nicht-Test-Codepath |
| AP2 | Hardcoded Secret / Token / URL in Source | grep nach plausiblen Secret-Regex | P0 immer |
| AP3 | Compile-Time-Guard fallengelassen (z. B. `#[cfg(test)]`-Block entfernt im Nicht-Test-Kontext) | Diff-Inspektion | P0 wenn Guard load-bearing war |
| AP4 | Test besteht durch Tautologie (asserts `true`, `assert x == x`) | jeden neuen Test-Body lesen | P1 außer wenn es der EINZIGE Test für die Funktion ist (dann P0) |
| AP5 | Behaviour-Divergenz von Spec ohne gemergten RFC | Spec-vs-Implementation-Diff | P0 immer |
| AP6 | Fehlender Parity-Test für einen portierten Handler | `tests/parity_<bundle>/` prüfen | P0 immer |
| AP7 | Neue Workspace-Dependency nicht in RFC deklariert | Metadata-Diff vs `git log --oneline rfcs/` | P0 immer |
| AP8 | `#[allow(clippy::*)]` / `# noqa` / `// eslint-disable` ohne Begründung | grep nach diesen Directives, Commit-Body prüfen | P1 außer wenn Begründung fehlt (dann P0) |
| **AP9** | **Tool-Call-Loop (Länge 1/2/3 über letzte 10 Calls; siehe §6)** | **In-Loop-Detector ODER Post-hoc-Replay der Tool-Call-History** | **P1; Re-Dispatch mit engerem Scope** |

---

## §10 Definition von Fertig

Eine Implementierung ist konform, wenn sie ALLE erfüllt:

- [ ] Jedes Worker-Brief enthält ein eindeutiges `sentinel_token`.
- [ ] Jeder Worker beginnt seine erste Reply MIT dem Sentinel literal.
- [ ] Jeder Worker emittiert eine `status.json` per
      `autoattended-orchestrator-spec.md` §9.1 mit einem
      Proof-of-Read-Eintrag pro konsumierter Datei.
- [ ] Proof-of-Read-Einträge deklarieren `sha256` + `lines` + eine
      Depth aus der §7.2-Liste.
- [ ] Workers fahren den §6.1-Tool-Call-Loop-Detector nach JEDEM
      Tool-Call mit `N=10`.
- [ ] Bei Loop-Detection emittiert der Worker `outcome=RETRY` mit
      `notes=tool-call-loop:...`.
- [ ] Bei Stuck nutzt der Worker einen der fünf §5.1-Blocker-Typen
      und hängt Proof-of-Read für jede referenzierte Datei an.
- [ ] Der Meta-Agent / Supervisor stichprobenartig EINEN von LD-1..LD-5
      pro Worker pro Wave (rotierend, sodass Workers nicht gamen
      können, welcher Test gefaked wird).
- [ ] Drift-Signale (§4.3) werden für jeden Worker vor dem
      Goal-Gate-Verdict gescannt.
- [ ] PP-13 fährt die §8.1-Tier-1-Toolchain für die Sprache des
      Workers; Tier-1-Failure → `outcome=FAIL`.
- [ ] PP-13s Anti-Pattern-Scan deckt AP1..AP9 ab.
- [ ] Tool-Output-Truncation (per `coding-agent-loop-spec.md` §5)
      taucht im Proof-of-Read als `depth=truncated:head-N` oder
      `depth=truncated:tail-N` auf; NIEMALS `depth=full`.
- [ ] `auto_status=false` ist Pflicht; fehlende `status.json` = FAIL.

---

## §11 Cross-Provider-Parity-Matrix

| Fähigkeit | Claude Code | OpenAI Codex / codex-rs | Gemini CLI | Notizen |
|---|---|---|---|---|
| Sentinel-Token-Replay (LD-1) | Verfügbar via System-Prompt-Addendum | Same | Same | Provider-agnostisch; nutzt Standard-Chat-Completion. |
| Proof-of-Read-Schema (LD-2) | Worker schreibt status.json | Same | Same | File-basiert; Provider-agnostisch. |
| 3-Sections-Challenge (LD-3) | Supervisor-Prompt | Same | Same | Provider-agnostisch. |
| Negative-Knowledge-Test (LD-4) | Supervisor-Prompt | Same | Same | Provider-agnostisch. |
| Line-Range-Quote (LD-5) | Supervisor-Prompt + File-Re-Read durch Supervisor | Same | Same | Provider-agnostisch. |
| Tool-Call-Loop-Detection (§6) | Implementierbar in `Session.run()`-Loop-Wrapper | Same | Same | Loop-Detector-Code ist Provider-unabhängig (operiert auf lokaler Tool-Call-History). |
| Toolchain-Tiers (§8) | Bash-Tools | Bash-Tools | Bash-Tools | Per-Language-Commands identisch. |
| Truncation → `depth=truncated:*` | Wrapper um Tool-Output | Wrapper | Wrapper | Provider-spezifisch in WAS truncated wird (unterschiedliche Default-Toolsets), uniform in WIE es aufgezeichnet wird. |

---

## §12 Appendix A — Worker-Brief-Template

```markdown
Du bist Agent {{agent_id}} in Wave {{wave_id}} von Sprint {{sprint_id}}.

Du besitzt Bundle: {{bundle_name}}
Owned Files (read-write):
{{owned_files_table}}

Read-only Files:
{{read_only_files_table}}

Spec-Files (autoritativ):
{{spec_files_table}}

INVARIANTS (lies diese zuerst, depth=thorough):
- META/INVARIANTS.md  sha256={{invariants_sha}}  lines={{invariants_lines}}

Reading-Depth required (per anti-skim-agent-spec.md §3.3):
- INVARIANTS.md → thorough
- Spec-Files → full
- Reference-Source-Files → full
- Skeleton-Files → read
- Alles andere → mindestens read

Status-File: schreib nach {{status_file_path}} matching das Schema in
autoattended-orchestrator-spec.md §9.1.

Wenn stuck: file EINEN Eintrag in META/REQUESTS-FROM-AGENTS.md per
anti-skim-agent-spec.md §5; setze Status auf outcome=RETRY; idle.

Done-Kriterien: per autoattended-orchestrator-spec.md §10 + §10 dieser Spec.

Tool-Call-Loop-Detector: fahre nach jedem Tool-Call per §6.

SENTINEL TOKEN: {{sentinel_token}}

Beginne deine erste Reply mit dem Sentinel-Token literal. Dann sage,
in einem Absatz, was du als Bundle verstehst. Dann beginn die Arbeit.
```

---

## §13 Skeleton-Fill-Contract (Protokoll A)

> Hinzugefügt 2026-05-18. Gilt für Workers, die in
> `autoattended-orchestrator-spec.md` Protokoll A (Implementation)
> operieren, wo PP-16s Preflight ein auskommentiertes Rust-Skelett
> erzeugt hat (per §14.4 der Orchestrator-Spec).

### §13.1 Was der Worker bekommt

Drei Inputs:

1. Die Skelett-Datei(en) am `skeleton_output_path` mit
   `todo!("SOURCE: <path>:<lines>")`-Macros an jeder Body-Site.
2. Die originalen Spec- / Reference-Source-Files, die in jeder
   `SOURCE:`-Annotation referenziert sind.
3. Ein SHA-256 vom Skelett UND jeder referenzierten Source-File,
   gepinnt von PP-16 zur Preflight-Zeit.

### §13.2 Worker-Pflichten

Für jedes `todo!("SOURCE: <path>:<lines>")`, das der Worker füllt:

1. **Lies den Source-Range** in der deklarierten Depth (§3.3); Default
   für Ports ist `full`.
2. **Zeichne Proof-of-Read auf** für die Source-File im deklarierten
   Line-Range (§7.1).
3. **Ersetze `todo!(...)` mit dem Body**, die umgebende Signatur
   exakt beibehaltend. Die Signatur kam von PP-16; der Worker
   ändert sie nicht ohne Iron-Rule-amendierende RFC.
4. **Quote den Source-Line-Range im Commit-Message-Body** (per
   `autoattended-orchestrator-spec.md` §5.6).
5. **Verifiziere die SHA-256 des Skeletts gegen den gepinnten Wert**
   beim Arbeitsstart. Wenn das Skelett gedriftet ist (z. B. ein
   Sibling-Worker hat es angefasst), MUSS der Worker STOPPEN und
   einen `EXTERNAL_DEPENDENCY`-Blocker in
   `META/REQUESTS-FROM-AGENTS.md` filen.

### §13.3 Verboten im Skeleton-Fill-Mode

- Eine Signatur ändern, die das Skelett liefert (RFC nötig).
- Ein `todo!()` hinzufügen, das NICHT vom Skelett kam (würde
  PP-16s Source-Annotation-Disziplin umgehen).
- Einen `// SAFETY:`-Kommentar von einem `unsafe`-Block entfernen
  (PP-13 wird ablehnen; PP-16 hat ihn aus einem Grund geschrieben).
- Bodies in einer File außerhalb von `skeleton_output_path` füllen
  (§5.1 Unique-File-Write-Disziplin).

### §13.4 Required Reading-Depth

| Source-Art | Minimale Depth |
|---|---|
| `SOURCE: <reference-source>:<lines>` (die zu portierende Funktion) | `full` für den genannten Line-Range |
| `UNSAFE-SOURCE: <reference-source>:<lines>` | `thorough` für den genannten Range UND die Caller der Funktion |
| Die Skelett-Datei selbst | `read` (um die umgebende Signatur zu bestätigen, in die du füllst) |
| `META/INVARIANTS.md` | `thorough` (per §3.3) |

### §13.5 Validierungs-Regeln

| Regel | Beschreibung | Severity |
|---|---|---|
| `FILL-001 source-range-read` | Jedes `todo!("SOURCE: P:L-M")`, das der Worker füllt, MUSS im Proof-of-Read des Workers mit `file=P, lines die L-M abdecken, depth=full` auftauchen. | ERROR |
| `FILL-002 skeleton-sha-pin` | Worker MUSS die SHA-256 des Skeletts gegen den Wert verifizieren, den PP-16 gepinnt hat, beim Arbeitsstart. Drift → STOP + `EXTERNAL_DEPENDENCY`. | ERROR |
| `FILL-003 signature-preserved` | Worker DARF NICHT eine Signatur ändern, die das Skelett liefert, ohne gemergten RFC. Diff-Inspektion durch PP-15 setzt durch. | ERROR |
| `FILL-004 unsafe-safety-preserved` | `// SAFETY:`-Kommentare auf `unsafe`-Blöcken vom Skelett MÜSSEN in den gefüllten Body überleben. PP-13 setzt durch. | ERROR |
| `FILL-005 no-new-todo` | Worker DARF NICHT neue `todo!()`-Calls einführen; jedes `todo!()` in der gefüllten Datei MUSS auf einen Skelett-Eintrag zurückverfolgbar sein. | ERROR |

### §13.6 Definition von Fertig (Skeleton-Fill)

- [ ] Jedes `todo!()` in den zugewiesenen Skelett-Datei(en) ist ersetzt.
- [ ] Keine neuen `todo!()` eingeführt (FILL-005).
- [ ] Skelett-SHA beim Start gematcht; finale SHA different (FILL-002).
- [ ] Proof-of-Read deckt jede `SOURCE:`-Annotation ab (FILL-001).
- [ ] Alle Signaturen erhalten (FILL-003).
- [ ] Alle `// SAFETY:`-Kommentare erhalten (FILL-004).
- [ ] Tier-1-Toolchain grün auf gefüllter Datei (§8.1).
- [ ] Status-File geschrieben per `autoattended-orchestrator-spec.md` §9.1.

---

## §14 Reading-Phasen (orthogonal zu Depth)

> Die Reading-Depth-Ladder in §3 sagt **wie viel** einer Datei du
> abdeckst. Die Reading-Phasen in dieser Section sagen **was du
> tust** mit dem, was du abgedeckt hast. Beides braucht's, damit ein
> Read als complete zählt.

### §14.1 Die vier Phasen

| # | Phase | Frage, die sie beantwortet | Output den der Worker produzieren können MUSS |
|---|---|---|---|
| 1 | **Survey** | „Was ist in dieser Datei? Welche Form hat sie?" | Section-Liste mit Zeilennummern; File-Shape (N Sections, K LOC, Language); Top-Level-Headline. |
| 2 | **Evaluation** | „Was davon matters für meine Aufgabe?" | Eine Relevanz-Map: welche Sections / Line-Ranges sind relevant fürs aktuelle Bundle, priorisiert P0 / P1 / unused. |
| 3 | **Kritische Findings** | „Was ist falsch, fehlt, widerspricht sich, ist gedriftet?" | Eine typisierte Finding-Liste (Severity P0 / P1) — Iron-Rule-Verletzungen, Spec-vs-Source-Drift, fehlende Sections, Anchors die nicht mehr resolven, tote Referenzen. |
| 4 | **Internalize** | „Kann ich darauf handeln ohne nochmal zu lesen?" | LD-3 bestehen (3-Section-Namens-Challenge), LD-4 (Negative-Knowledge), LD-5 (Line-Range-Quote). Kann faithful paraphrasieren; kann beantworten „was steht NICHT in dieser Datei?". |

Phasen-Reihenfolge ist typisch **Survey → Evaluation → Kritische
Findings → Internalize**, aber Findings tauchen oft WÄHREND
Internalize auf (der Akt des Internalizens enthüllt Widersprüche).
Ein vollständiges Reading erreicht alle vier; ein partial Reading
stoppt früher und MUSS das deklarieren.

### §14.2 Phase × Depth-Matrix

| Depth | Survey | Evaluation | Kritische Findings | Internalize |
|---|:-:|:-:|:-:|:-:|
| `grep` (anti) | partial | nein | nein | nein |
| `sed-partial` / `head-only` (anti) | partial | partial | nein | nein |
| `skim` | ja | partial | partial (nur per-Section) | nein |
| `read` | ja | ja | partial | partial |
| `thorough` | ja | ja | **ja** | **ja** |
| `troubleshooting` | ja | ja | ja (fokussiert auf den Bug) | ja (fokussiert) |
| `fan-out` | ja (per-File flach) | ja | ja | partial (per File) |
| `truncated:head-N` / `truncated:tail-N` | partial | partial | nein | nein |

Nur `thorough` und `troubleshooting` (innerhalb des fokussierten
Scopes) erreichen die volle Ladder. `fan-out` erreicht Internalize
*per-File*, aber der Inventur-Output ist die Synthese — der Worker
MUSS jede File im Inventar als noch-`thorough`-bedürftig behandeln
bevor er auf sie handelt.

### §14.3 Phasen-Outputs in der status.json des Workers

Das `proof_of_read`-Schema (§7.1) wird um ein `phase_reached`-Feld
erweitert, das die höchste completed-Phase pro File benennt:

```json
{
  "file": "META/INVARIANTS.md",
  "sha256": "fa39a3...",
  "lines": 412,
  "depth": "thorough",
  "phase_reached": "internalize",
  "phases_evidence": {
    "survey": "9 Sections; INVARIANTS-kanonische Struktur erkannt",
    "evaluation": "§BBB + §UPSERT-PATTERN sind load-bearing für dieses Bundle",
    "critical_findings": "§UPSERT-PATTERN Zeile 187 widerspricht der customer.ttl ogit:CustomerWriter mandatory-attributes (filed REQUESTS-FROM-AGENTS.md#A4-2026-05-18T14:22)",
    "internalize": "kann LD-3/4/5 literal beantworten"
  }
}
```

Die `phases_evidence`-Map ist OPTIONAL wenn `phase_reached=survey`
(kein Evidence über die Section-Liste hinaus required), WIRD aber
required bei `phase_reached >= evaluation`, weil diese Phase
Judgment claimt.

### §14.4 Mapping Lie-Detector-Tests auf Phasen

Jeder LD-1..LD-5-Test (§4.1) probt eine spezifische Phase. Der
Meta-Agent SOLLTE den Spot-Check über die vier Phasen rotieren,
damit Workers nicht gamen können, welche Phase zu faken ist.

| Test | Probt Phase | Was eine passing Answer beweist |
|---|---|---|
| LD-1 Sentinel-Token | **Survey** des Briefs | Der Worker hat den Brief tatsächlich in den Context geladen. |
| LD-2 Proof-of-Read mit SHA | **Survey** der File | Der Worker accessed die File am deklarierten Content. |
| LD-3 3-Sections-Namens-Challenge | **Survey + Evaluation** | Der Worker kann Struktur lokalisieren UND wählte welche Sections zu beachten. |
| LD-4 Negative-Knowledge-Test | **Internalize** | Der Worker hat ein faithful Mental-Model gebaut — kann beantworten „was steht NICHT in dieser Datei" ohne zu halluzinieren. |
| LD-5 Line-Range-Quote | **Internalize + Kritische Findings** | Der Worker kann verbatim recallen UND Drift zwischen Recall und Source detektieren. |

Ein Worker, der `phase_reached=internalize` deklariert, MUSS LD-3,
LD-4 UND LD-5 bestehen können. Spot-Check-Failure bei einer
geclaimten Phase ⇒ Phase-Claim wird abgelehnt und der Proof-of-Read
wird auto-downgraded auf die höchste nachweisbar bestandene Phase.

### §14.5 Per-File-Required-Phase nach File-Art

Unterschiedliche File-Arten BENÖTIGEN unterschiedliche Minimum-
Phasen. Das Worker-Brief MUSS die Per-File-Phase-Requirements
zusammen mit den Depth-Requirements deklarieren:

| File-Art | Minimum-Depth | Minimum-Phase |
|---|---|---|
| `META/INVARIANTS.md` | `thorough` | **internalize** |
| `CLAUDE.md` / `BOOT.md` / RFCs | `thorough` | **internalize** |
| Memory-Files (`CONTEXT.md` / `JOURNAL.md` / `TODO.md`) | `thorough` | **internalize** |
| Spec-Files fürs Bundle (z. B. TTL, OpenAPI, JSON-Schema) | `full` (read) | **internalize** |
| Reference-Source für Ports | `full` | **internalize** |
| Skeleton-Files die der Worker füllt | `read` | **evaluation** |
| Sibling-Bundle-Files (read-only Context) | `skim` | **evaluation** |
| Files referenziert für allgemeine Orientation | `skim` | **survey** |
| Files referenziert nur für Symbol-Lookup | `grep` | **survey** |

Wenn ein Worker bei einer Phase unter dem Required stoppt, zählt
der Read NICHT als complete — selbst wenn die Depth ausreichend
war. Beide Achsen müssen die Bar klären.

### §14.6 Kritische-Findings-Eskalation

Findings produziert in Phase 3 (Kritische Findings) werden nach
Severity geroutet:

| Finding-Severity | Filed wo | Worker-Aktion |
|---|---|---|
| P0 — Iron-Rule-Verletzung im Input, Spec-vs-Source-Widerspruch, gebrochener Anchor der das Bundle betrifft | `META/REQUESTS-FROM-AGENTS.md` mit Blocker-Typ aus §5.1; Worker idled | STOP Arbeit; nicht zum Commit weitergehen |
| P1 — kleinere Inkonsistenz, tote Referenz außerhalb des Bundle-Scopes, Typo, stale Comment | `Altlasten.md` / `TECH_DEBT.md`-Zeile mit der Bundle-ID des Workers | Arbeit fortsetzen; der Orchestrator triaged später |
| INFO — Observation die das Bundle nicht betrifft | Notes-Feld in `status.json`; nirgendwo sonst filed | Arbeit fortsetzen |

Ein Worker der internalized, aber NICHT ein P0-Finding escaliert
ist in Verletzung: Missing-Escalation ist selbst ein P0-Finding,
das PP-13 catchen wird (Anti-Pattern AP1 — „silent Fallback der
Errors schluckt" — generalisiert hier als „silent Skim der
Findings schluckt").

### §14.7 Validierungs-Regeln

| Regel | Beschreibung | Severity |
|---|---|---|
| `PHASE-001 phase-declared` | Jeder `proof_of_read`-Eintrag MUSS ein `phase_reached`-Feld enthalten. Absent ⇒ behandelt als `phase_reached=survey` (der schwächste Claim). | ERROR |
| `PHASE-002 phase-monotonic-with-depth` | Ein `phase_reached`-Claim MUSS konsistent mit der §14.2-Matrix sein. Z. B. `depth=grep` + `phase_reached=internalize` ist invalid. | ERROR |
| `PHASE-003 phase-evidence-required` | Wenn `phase_reached >= evaluation`, MUSS die `phases_evidence`-Map vorhanden sein mit non-empty Entries für jede geclaimte Phase. | ERROR |
| `PHASE-004 file-kind-phase-bar` | Wenn das Worker-Brief eine Minimum-Phase per §14.5 für eine File deklariert, MUSS der `phase_reached` des Workers für diese File ≥ dem deklarierten Minimum sein. | ERROR |
| `PHASE-005 critical-findings-routed` | P0-Findings produziert während Phase 3 MÜSSEN nach `META/REQUESTS-FROM-AGENTS.md` filed sein BEVOR der Worker irgendwelchen Code commited, der vom betroffenen Input abhängt. | ERROR |
| `PHASE-006 internalize-passes-LD-3-4-5` | Ein Claim von `phase_reached=internalize` MUSS Spot-Checks von LD-3, LD-4 und LD-5 in Rotation überleben. Failure ⇒ Auto-Downgrade auf höchst-bestandene Phase. | ERROR |

### §14.8 Definition von Fertig (Per-File-Read)

Ein Read einer einzelnen File ist complete wenn ALLE:

- [ ] Depth aus §3 auf dem deklarierten Level.
- [ ] Phase aus §14.1 erreicht das required Minimum per §14.5.
- [ ] `proof_of_read`-Eintrag enthält `sha256`, `lines`, `depth`,
      `phase_reached` und `phases_evidence` für `evaluation+`.
- [ ] Jedes P0-kritische-Finding ist filed bevor der Worker commited.
- [ ] LD-3 / LD-4 / LD-5-Spot-Checks bestanden wenn
      `phase_reached=internalize` geclaimt ist.

---

## §15 Kognitive Anti-Patterns (CA1..CA4)

> Wo AP1..AP9 (§9) **Output**-Probleme fangen — der Code selbst sieht
> falsch aus — fangen CA1..CA4 **Cognition**-Probleme — die Art wie
> der Worker zum Output gekommen ist, ist falsch. CA-Findings tauchen
> oft zur gleichen Zeit auf wie AP-Findings; die Unterscheidung ist:
> CA-Fixes brauchen Process-Change (Re-Read, Re-Think, Re-Spawn),
> während AP-Fixes manchmal in-place editiert werden können.
>
> Die kognitiven Anti-Patterns sind joint owned: der Meta-Agent spotted
> sie beim PR-Review via Lie-Detector (§4); PP-13 spotted sie beim
> Code-Review durch Korrelation von Commit-Timestamps mit
> Proof-of-Read-Timestamps.

### §15.1 Die vier kognitiven Anti-Patterns

| # | Name | Wie es aussieht | Detection-Signature | Counter-Pattern | Severity |
|---|---|---|---|---|---|
| **CA1** | **Cognitive Dissonance** | Worker sieht einen Widerspruch zwischen zwei autoritativen Sources (Spec vs. Reference-Source, INVARIANT vs. Comment, TTL vs. Code) und löst durch Hand-Waving oder Picken eines ohne Investigation. Die Dissonance wird übermalt statt eskaliert. | Output enthält Phrasen wie „I went with", „I chose", „I preferred Y because it feels right / is already there / compiles", ohne entsprechenden `SPEC_SOURCE_MISMATCH`-Eintrag in `META/REQUESTS-FROM-AGENTS.md`. | File `SPEC_SOURCE_MISMATCH` (§5.1) und idle bis der Meta-Agent ein RFC schreibt. Niemals Dissonance unilateral auflösen. | **P0** |
| **CA2** | **Dunning-Kruger Overconfidence** | Worker claimt confident Knowledge in einem Bereich, wo seine tatsächliche Depth shallow ist. Der Agent weiß nicht, was er nicht weiß, also ist sein Sense of Certainty un-kalibriert. Der Output liest sich definitiv, wo der Read dünn war. | Ein `phase_reached=internalize`-Claim (§14), der LD-3 / LD-4 / LD-5-Spot-Checks failt. Eine spezifische numerische Behauptung („62 Lehren", „die Signatur hat 3 Argumente") ohne entsprechenden `proof_of_read`-SHA. Confident Paraphrase, wo die Source eigentlich etwas anderes sagt. | Auto-Downgrade des Phase-Claims auf die höchst-bestandene Phase (per `PHASE-006`). Re-Read mit der proper Depth + Phase. Wenn der Worker weiter übersteuert, route ihn zu einem kleineren Bundle. | **P0** |
| **CA3** | **Kahneman/Tversky Easy-Path (System-1-Short-Circuit)** | Worker pattern-matched Surface-Features des Inputs — „File sieht aus wie ein CRUD-Handler, also muss es ein CRUD-Handler sein" — ohne den System-2-Check zu fahren (Read den tatsächlichen Function-Body, vergleich gegen die Spec). Easy-Path ist am schnellsten wenn Surface die Reality matched, aber verheerend wenn nicht. | Die erste Reply ist plausibel-correct-klingend, aber der Proof-of-Read ist `depth=grep` oder `depth=sed-partial`. Output beschreibt Struktur in generischen Termen („standard Route-Handler", „typische Migration") statt spezifischen Termen („die Function auf Zeilen 47-91 dispatched auf req.path"). | Force System-2: verlange LD-2 (SHA + Line-Count) und LD-5 (Line-Range-Quote) bevor irgendein structural Claim akzeptiert wird. Reading-Depth MUSS mindestens `read` sein; Phase MUSS mindestens `evaluation` sein, bevor irgendein Output, der einen structural Claim macht. | **P0** |
| **CA4** | **Eager Amok** | Worker fängt an Code zu schreiben (oder zu committen, oder zu pushen), bevor die required Reading- + Planning-Phasen complete sind. Enthusiasm rennt der Disziplin voraus. Die Arbeit *sieht* produktiv aus — es gibt einen Diff — aber sie ist auf incomplete Understanding gebaut. | Der erste Code-Write-Timestamp ist vor den `proof_of_read`-SHA-Pin-Timestamps für eine oder mehrere required Files (§14.5). `status.json` zeigt Commits, die landen, während `phase_reached` noch `survey` oder `evaluation` auf Files ist, die bei `internalize` sein sollten. Worker-Narrative springt von Brief-Read zu First-Commit ohne sichtbaren Thinking-Step. | STOP und verlange complete Proof-of-Read für ALLE Files an ihrer required §14.5-Minimum-Phase BEVOR irgendein Commit. Die Iron Rule gilt, egal wie „obvious" die Implementation scheint. | **P0** |

### §15.2 Warum alle vier P0 sind

Jedes von CA1..CA4 produziert Output, der *aussieht* als wäre er
richtig. AP1..AP9 produzieren Output, der falsch aussieht (ein
`unwrap_or_default()` ist sichtbar; ein fehlender Parity-Test ist
sichtbar). Kognitive Anti-Patterns produzieren Output, dessen
Correctness komplett auf dem unverifierbaren Claim ruht, dass der
Worker gelesen + verstanden + gedacht hat, bevor er schrieb. Sie
sind P0 weil sie den Trust-Contract brechen, auf den der Meta-Agent
sich verlässt, wenn er Diffs reviewed ohne den Read des Workers zu
wiederholen.

### §15.3 Joint Ownership: wer fängt was

| Anti-Pattern | Catch-Site | Detection-Methode |
|---|---|---|
| CA1 Cognitive Dissonance | Meta-Agent (PR-Review) + PP-15 (Cross-Source-Diff) | Grep PR-Commit-Messages nach „I went with" / „I chose" / „preferred" gegen Spec-vs-Source-Mismatch; check `REQUESTS-FROM-AGENTS.md` für Abwesenheit des entsprechenden Blockers. |
| CA2 Dunning-Kruger Overconfidence | Meta-Agent (Lie-Detector-Spot-Check) | Rotate LD-3 / LD-4 / LD-5 auf einem Worker pro Wave; cross-check `phase_reached`-Claims gegen tatsächliches Passing. |
| CA3 Kahneman/Tversky Easy-Path | Meta-Agent (Proof-of-Read-Inspection) + PP-13 (Output-vs-Source-Diff) | Worker-Output liest sich als Paraphrase statt Quote; SHA fehlt; Reading-Depth deklariert inkonsistent mit Output-Claims. |
| CA4 Eager Amok | PP-13 (Commit-Timestamp-Audit) + Meta-Agent (status.json-Ordering-Check) | Commit-Timestamps sind vor den `proof_of_read`-SHA-Pin-Timestamps; `phase_reached` war unter dem required Minimum beim First-Commit. |

### §15.4 Counter-Patterns: wie ein gesunder Worker sich verhält

| Anti-Pattern | Healthy-Alternative |
|---|---|
| CA1 | „Ich habe bemerkt, die Spec sagt X aber der existierende Code tut Y. Ich file `SPEC_SOURCE_MISMATCH`. Idle." |
| CA2 | „Ich bin confident über §3 von INVARIANTS.md (depth=thorough, phase=internalize). Ich bin uncertain über §6 (depth=skim, phase=survey). Ich vertiefe §6 bevor ich darüber etwas behaupte." |
| CA3 | „Bevor ich diese File-Struktur beschreibe: hier ist `proof_of_read: { file=X, sha256=..., depth=read, phase_reached=evaluation }`. Die Struktur ist: [spezifische Section-Namen mit Zeilennummern]." |
| CA4 | „Reading-Phase complete auf allen 4 required Files. Kritische Findings filed (Zero P0). Status: `phase_reached=internalize` auf allen Files. JETZT beginne ich den ersten Commit." |

### §15.5 Validierungs-Regeln

| Regel | Beschreibung | Severity |
|---|---|---|
| `CA-001 dissonance-escalation` | Wenn der Output eines Workers Spec-vs-Source-Divergenz erwähnt, MUSS `META/REQUESTS-FROM-AGENTS.md` einen entsprechenden `SPEC_SOURCE_MISMATCH`-Blocker-Eintrag enthalten. Abwesenheit ⇒ CA1 P0-Finding. | ERROR |
| `CA-002 confidence-calibration` | Ein `phase_reached=internalize`-Claim, der LD-3 / LD-4 / LD-5-Spot-Check failt ⇒ CA2 P0; Phase-Claim auto-downgraded. | ERROR |
| `CA-003 system-2-required-before-structural-claim` | Jeder structural Claim über den Content einer File (Function-Count, Section-Namen, Signature-Shapes) MUSS preceded sein von einem `proof_of_read`-Eintrag mit `depth >= read` UND `phase_reached >= evaluation`. Sonst CA3 P0. | ERROR |
| `CA-004 read-before-write-ordering` | `status.json` MUSS zeigen, dass alle required-Minimum `proof_of_read`-Einträge timestamped sind BEVOR dem First-Code-Commit-Timestamp für Files an der §14.5-required-Minimum-Phase. Sonst CA4 P0. | ERROR |

### §15.6 Mindset-Level-Relation zu den vier Savants

Die vier Savants in `autoattended-orchestrator-spec.md` §4.0 haben
jeweils eine *kognitive* Posture, die einer spezifischen CA
widersteht:

| Savant | Mindset | Widersteht primär |
|---|---|---|
| PP-13 brutally-honest-tester | „was würde in Production um 3 Uhr morgens brechen, was der Author sich weggeredet hat?" | CA1 (Sich-Wegreden IST Cognitive Dissonance) + CA4 (Production bricht nicht wegen Enthusiasmus) |
| PP-14 convergence-architect | „was könnte das werden, das wir nicht sehen?" | CA3 (Easy-Path schließt Possibilities premature) |
| PP-15 baton-handoff-auditor | „lining sich diese Contracts an den Nähten wirklich auf?" | CA1 (Dissonance versteckt sich oft an Handoffs) |
| PP-16 preflight-drift-auditor | „matched der Plan immer noch das System wie es jetzt gerade ist?" | CA2 (Overconfidence in einen Plan der von der Reality gedriftet ist) |

Der volle 4-Savant-Pass ist jointly ein Antibody gegen alle vier
kognitiven Anti-Patterns. Eine Wave, die nur einen Savant fährt,
catched nur die CAs, denen der Mindset dieses Savants widersteht;
eine Wave, die das Cooperative Council fährt (Orchestrator-Spec
§15), catched alle vier.

### §15.7 Definition von Fertig (Kognitive Hygiene)

Ein Worker besteht das Kognitive-Hygiene-Gate wenn ALLE:

- [ ] `CA-001` clean: jede Spec-vs-Source-Erwähnung ist matched mit einem filed `SPEC_SOURCE_MISMATCH`-Blocker.
- [ ] `CA-002` clean: jeder `phase_reached=internalize`-Claim überlebt LD-3/4/5-Spot-Checks.
- [ ] `CA-003` clean: jeder structural Claim ist backed durch ausreichenden `proof_of_read`.
- [ ] `CA-004` clean: Read-Timestamps sind vor Commit-Timestamps für alle required-Minimum-Files.
- [ ] Worker hat nicht `I went with` / `I chose` / `I preferred` ausgegeben ohne associated Blocker-Filing.

---

*Ende der Datei anti-skim-agent-spec.md.*
