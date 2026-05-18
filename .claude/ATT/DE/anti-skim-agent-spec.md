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

*Ende der Datei anti-skim-agent-spec.md.*
