# Application implementation ledger

Each row names a running natlang application slice and the next concrete
product gate. A scripted model in an integration test proves the interpreter
and host boundary; it does not prove a teacher or student model will make good
semantic decisions. None of these prototypes has passed a live teacher pilot
in this application buildout. The teacher server was not started for this work.

| Project | Implemented and checked | Next product gate |
|---|---|---|
| P01 media | CPU FFmpeg probe, transforms, hash and metadata checks, optional visual inspector | Retained long-running jobs, revisioned library, live model quality |
| P02 semantic merge | Ten semantic data families, whole-history and incremental examples, seeded case corpus | Live pinned-model replica comparison and semantic review |
| P03 build | Serial graph execution, declared-input checks and verified built-in cache | Parallel ready tasks and a documented arbitrary-process read-set contract |
| P04 terminal | Event Fold, real build/media recipes, correlated jobs and cancellation outcome | Broader Git/Bash recipes and durable session recovery |
| P05 types | Finite inference and call checking | Cross-file evidence and live model quality |
| P06 IDE | Revisioned source edits, child runs, trace cursor, generated view and frozen scenario evaluation | Interactive editor and training pipeline/checkpoint management |
| P07 packages | Offline dependency resolution, checked bundles and atomic install pointer | Upgrade flow, remote archive trust and actual ecosystem compatibility |
| P08 games | Merchant economy, simultaneous combat and evidence-linked NPC actions | Strategy quality, renderer/game loop and interacting worlds |
| P09 spells | Typed spell interpretation and exact arena resolution | Broader mechanics, graphical play and tuned spell model |
| P10 notebook | SQL and TypeScript cells, dependency invalidation and result lineage | Selected SQL engine contract, streaming data and interactive UI |
| P11 wiki | Semantic page merge, profile pinning and child cells with stale-result check | Transport/reconnection, browser runtime, live merge repeatability |
| P12 logs | Evidence index, incident Fold, exact escalation receipts | Durable source cursors, live buffering and operational sink |
| P13 data | Finite SQLite migration | Larger schema/data fixtures and broader sources |
| P14 evidence | Versioned local passage search and literal citation checks | Better retrieval and semantic entailment review |
| P15 workflows | Durable local intents, idempotent fake receipts and reconciliation | Cross-process locking and real provider outcome contracts |
| P16 experiments | Seeded local trials, journals and report checks across several domains | Live teacher admission and checkpoint comparison |
| P17 test explorer | Finite dependency-plan case selection and assessment | More programme families and failure minimization |
| P18 publisher | Structured cited document to paired Markdown/HTML with local pointer | Visual review, printable output and external release adapter |
| P19 scheduling | Exact UTC slot enumeration, dependency/conflict checks, conditional commit | Calendar adapter, durable event stream and DST fixture suite |
| P20 repositories | Exact isolated candidate patches, declared checks and repair revisions | Native natlang loader/type gates, broader repositories, reviewed worktree output |

## Shared architecture findings

The applications did not require a larger language core. Natlang makes
semantic choices in small typed functions and event folds; crisp evaluators
own exact IO, state, durable receipts, source loading and rendering. Native
objects remain in selected host environments. The main cross-product needs
are better observed child runs, engine-specific diagnostics, and explicit
source/effect provenance, rather than new built-in syntax.

One actual host harness fault appeared in P20: a nested Node test inherited
`NODE_TEST_CONTEXT`, reported success and skipped its test files. The
repository workbench clears that variable for child checks; other child
process adapters should be audited for inherited runner state. The publisher,
workflow and wiki fixtures have local single-writer assumptions; concurrent
production adapters need an explicit transaction or reconciliation contract.

The next evaluation pass should freeze model identity, source revision,
engine/environment bindings, root seed, scenario inputs and exact checks for
each family. Preserve traces and unsuccessful trajectories. Semantic judgments
need reviewed rubrics; mechanical checks alone admit no training sample.
