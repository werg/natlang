# Interactive application delivery

## Inquiry Lab extension

The [Inquiry Lab](../../ts-host/studio/research/README.md) implements a first
integrated consumer of the five directions in the
[semantic software plan](../SEMANTIC_SOFTWARE.md). Natlang can own domain State,
generate checked source and view handlers, retain candidate schemas and
changes, and maintain explicit evidence links. The host provides exact
versioned storage, execution receipts, branch activation, graph queries and
DOM projection. Browser and integration checks cover the mechanics; live
model quality, schema migration judgment and semantic merge quality remain
scenario gates. This extends the twenty existing project families.

The shared [Natlang Studio](../../ts-host/studio/README.md) now supplies 22
browser application surfaces for P01–P20, including separate economy, combat
and NPC games. The studio is launched with
`node ts-host/scripts/serve-studio.mjs` after building `ts-host`.

## Architectural correction: natlang drives operations

The implementation must not reduce natlang to selecting one action while a
crisp handler owns the application's algorithm. Each submitted interaction is
a natlang reduction which can call operations repeatedly, inspect their typed
results, correct a proposal, and continue until the goal is fulfilled or a
concrete blocker remains. There is no one-operation-per-event restriction.

The distinction is between **algorithm ownership** and **operation mechanics**.
Natlang traverses a notebook dependency graph, chooses an investigation path,
iterates tests, compares experiments, interprets a spell and merges prose.
Crisp code executes a cell, queries SQLite, checks a literal citation, updates
exact balances, renders DOM, or launches a process. These helpers must not
silently absorb the semantic or algorithmic work intended for interpreter
training. A host constraint needs a concrete justification in the operation's
contract; caution alone is not a reason to move application logic to JavaScript.

The UI uses natlang-generated view plans with application-specific crisp
renderers. This keeps text editing and frequent UI events responsive while
allowing the interpreter to drive the submitted workflow. Streaming input is
represented by ordinary event records, not new language syntax.

## Shared extensions delivered

- `BrowserNatlangApplication` has an awaited commit hook, initial revision,
  cancellation and view refresh without event replay.
- IndexedDB state/draft/trace/operation journals, provenance and branch restoration.
- Browser writer locks and stale-tab detection before executing an interaction.
- Worker-owned child execution with shared-model turn forwarding; worker
  cancellation also stops non-yielding crisp child code.
- Host-owned child traces with explicit frame reads, avoiding full-trace state
  in the model's subsequent inputs.
- A loopback native companion with authenticated same-origin jobs, inspection,
  process-group cancellation, idempotency keys and atomic receipts. Its eval
  environment is trusted host authority, not a mandatory sandbox abstraction.
- Python and TypeScript source loaders now read nested type aliases without
  truncating on a record semicolon. Both type parsers accept record semicolons.

No new core evaluation form, UI primitive, process primitive or binary-reference
type was needed. Job and trace identities are ordinary typed application data.

## Scope and remaining product work

| Project | UI and executable integration delivered | Remaining product work |
|---|---|---|
| P01 | Import/sample MP4, exact transform controls, CPU FFmpeg render, video preview, metadata/hash evidence | Transformation library management, long asset collections, visual-model review |
| P02 | Ten-family ancestor/replica editor, semantic proposal, unresolved issues and adoption history | Multi-replica transport, pinned-model repeatability trials, per-family structured editors |
| P03 | Declared-input build, actual verified cache, artifact and receipt view | General project DAG editor and broader task/repair recipes |
| P04 | Trusted Bash session, recipes, output and cancellable native jobs | Interactive PTY, durable shell environment and richer recipe libraries |
| P05 | Source context, semantic signature proposals, evidence and diagnostic collection | Cross-file navigation/index, reviewed semantic type-check runs |
| P06 | Editor, worker child runs, trace cursor, scenario collection and per-case evaluation | Integrated semantic highlighting, source graph navigation, training-job management |
| P07 | Registry, exact locks and atomic installation | Remote registries, upgrade UI and broader packages |
| P08 | Inventory market/ledger; simultaneous combat arena/journal; NPC conversation/memory | Shared persistent world, autonomous actor schedules, policy quality |
| P09 | Incantation input, typed spell actions, mana/health arena and grimoire | Larger mechanics, trained spell interpreter and multiplayer |
| P10 | Mixed engine cells, natlang dependency orchestration, lineage/invalidation, worker execution and SQLite | Streaming tables, richer visualizations, collaborative notebook editing |
| P11 | Page collection/editor/preview, collaborator draft staging, semantic merge and executable cells | Network replication, offline/reconnect protocol and published wiki server |
| P12 | Ingested log windows, evidence-linked incident management and local escalation ledger | Persistent streaming source connectors and real escalation sinks |
| P13 | JSON import, schema mapping, transformation preview and export | Transactional destination adapters and richer entity reconciliation |
| P14 | Passage collection/search, exact quotations, cited claim notebook | Better retrieval, entailment review and external source connectors |
| P15 | Visual saga, simulated provider, fault injection, reconciliation and compensation receipts | Real provider credentials/outcome contracts and durable cross-process workflow service |
| P16 | Hypotheses, seeded inventory trials, matched comparison and charts | More worlds, policy source editing and distributed trials |
| P17 | Source/contract editor, case authoring, real per-case child execution/results | Failure minimization and training admission/review workflow |
| P18 | Section editor, reorder operations, document preview and paired HTML/Markdown export | Citation-aware publishing integration, PDF and external release targets |
| P19 | UTC timeline, tasks, conflict checks and completion | Calendar connectors, timezone/DST presentation and dependency-aware calendar events |
| P20 | Exact candidate patching, before/after display, actual isolated syntax/behavior checks | Opening arbitrary repository manifests and reviewable worktree publication |

These remaining gates are intentionally explicit. A polished local interface
is not evidence that network sync, provider integration, semantic quality or
long-running recovery at every underlying effect boundary has been solved.

## Verification and training boundary

Integration tests execute the actual generated natlang source under scripted
turns for every app and exercise multiple operation calls in one interaction.
Independent host checks use real SQLite, FFmpeg, Bash, cache verification,
package installation and isolated repository tests, including a failing
candidate. Browser checks visit all routes, operate controls, reload persisted
state, preserve drafts, inspect history, execute notebook cells, cancel a
non-yielding worker and inspect mobile layout.

These checks are suitable plumbing regressions, not teacher-quality judgments.
The next semantic evaluation should use the same UI events and frozen source,
model identity, root seed and environment, preserving failed runs and intermediate
operation states. No UI trace is automatically admitted as a training sample.
