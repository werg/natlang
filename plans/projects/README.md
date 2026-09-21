# Individual project implementation plans

## Further semantic application development

The [semantic software plan](../SEMANTIC_SOFTWARE.md) specifies five extensions
to the application's role: [learned tools](S01_LEARNED_TOOLS.md),
[revisable schemas](S02_REVISABLE_SCHEMAS.md),
[generated interactions](S03_GENERATED_INTERACTIONS.md),
[intent-preserving changes](S04_INTENT_CHANGES.md) and
[beliefs and investigation](S05_BELIEFS.md). It identifies current studio
constraints, concrete host/library changes and an integrated notebook delivery
sequence. These are proposed work, separate from the delivered scope below.

## Existing portfolio

Planning status: all twenty families have an executable first slice; see the
[implementation ledger](IMPLEMENTATION_STATUS.md) for tested scope and open
product gates. These plans refine the [product catalogue](../AMBITIOUS_PROJECTS.md)
using the settled [execution interface direction](../EXECUTION_INTERFACES.md).
They identify natlang capabilities and concrete delivery gates, rather than
treating application requirements as new language primitives.

## Shared capability vocabulary

| ID | Capability | Current status / intended change |
|---|---|---|
| C0 | Typed values, lexical functions, ordinary calls, finite Map/Fold/Iterate, validation, crisp code and declared effects | Existing mechanisms. Use them rather than new global tools. |
| C1 | Explicit crisp-engine bindings, proposed required `engine` argument, independent environment/isolation policy | Planned versioned surface and executor work. Existing JS/TS remains the compatibility implementation. |
| C2 | Explicit root seed and deterministic logical invocation seed derivation | Planned model/host API work. Record actual backend support; no blanket repeatability promise. |
| C3 | Checked programme definitions supplied in memory and isolated child-run invocation | Planned loader/embedding seam; meta-operations exposed through crisp environments. |
| C4 | Portable execution/reduction trace records and common inspection API | Planned observation interface. Diagnostic, reconstruction and replay coverage are distinct. |
| C5 | Stream Fold with typed input, waiting/closure/failure, bounded input admission | Partial open-list implementation exists; complete semantics and host integration are planned. |
| C6 | Bounded streaming Map and optional independent parallel reductions | Planned only where required. Output ordering, retained history and budgets need explicit contracts. |
| C7 | One TS-style natlang value/type boundary across engines | Existing structural subset retained; specify/test cross-engine conversions. Native host objects stay in eval. |

These labels organise dependencies; they are not model vocabulary or a runtime feature-negotiation framework. A project may start with existing compatibility execution while the new engine selector is being implemented. Do not block a finite merge or spell experiment on stream Map or durable infrastructure. C4 diagnostic recording can precede complete replay support.

## Common implementation rules

- `host.*` names below are **illustrative application APIs**, not a universal host protocol. Keep them small and extract shared interfaces only from real consumers.
- A `ts` engine with direct native sharing needs a JS host environment. It does not magically share Python objects from the current coordinator. Use a compatible host, a deliberately declared bridge, or another selected engine; record which. Isolation is a separate choice. Direct native sharing is supported and does not claim the old sandbox guarantees.
- Native jobs, images, tables, connections and DOM objects remain in the crisp environment. Cross the natlang boundary with ordinary typed summaries, IDs or immutable snapshots. IDs are local to a declared environment/revision; use after disposal or migration fails explicitly. No first slice requires new opaque host-type syntax.
- Eventful applications use stream Fold over application event records. Within a step, inputs stay fixed. Late observations/results become later events. Use IDs/revisions to reject stale results; do not introduce ambient interrupts or hidden instruction changes.
- Fold ordering is per state owner. Separate independent state owners may be scheduled independently. Multiple event sources must be merged by a declared host rule and their consumed order recorded.
- Map is for independent work. A finite Map may run serially; C6 is not a prerequisite for every programme that uses Map. A shared mutable environment must have deliberate ownership; it is not safe to parallelise all apparently independent calls automatically.
- Crisp code performs exact operations and host access. Natlang still chooses application steps, interprets observations and decides semantic outcomes. A host function must not hide the entire semantic workflow.
- Every plan states a first build, subsequent increments and contract tests. Teacher pilots execute the real `.nl` functions; supplied fake outcomes test the harness but do not prove teacher competence or real integration.
- Freeze source, engine/environment bindings, seed profile, inputs and expected contracts for collection. Shared native mutations may prevent replay; preserve the original observations and label the limitation. Product invariants need independent checks beyond successful tool execution.

## Plans and capability dependencies

“First” identifies the first useful slice, not every eventual product feature. C0/C7 apply throughout.

| ID | Plan | Natlang work for first supported slice | Later relevant work |
|---|---|---|---|
| P01 | [Media workbench](P01_MEDIA.md) | C1 host execution; C4 observations | C5 interactive progress; optional C6 batch work |
| P02 | [Semantic replication](P02_SEMANTIC_MERGE.md) | C2 reproducible configuration; C4 comparison | C5 live replication; C6 independent documents |
| P03 | [Build tool](P03_BUILD.md) | C1 process environment; C4 effects/results | C5 background operation; C6 independent ready tasks |
| P04 | [Semantic terminal](P04_TERMINAL.md) | C1 retained host environment; C5 session events | C3 installed programme execution |
| P05 | [Type tools](P05_TYPES.md) | C3 checked source/type APIs | C4 execution evidence; C5 edit stream |
| P06 | [IDE](P06_IDE.md) | C1 renderer bindings; C3 load/run; C4 inspector; C5 events | C2 experiments; deeper replay and training integration |
| P07 | [Package manager](P07_PACKAGES.md) | C3 source validation; C1 file environment | Additional engine compatibility checks |
| P08 | [Three game prototypes](P08_GAMES.md) | C2 controlled randomness; C5 world events | C6 independent policies; memory inspection through eval |
| P09 | [Spell game](P09_SPELLS.md) | C0/C7 suffice; C2/C4 for experiments | C5 interactive input; specialised checkpoint routing in host |
| P10 | [Notebook](P10_NOTEBOOK.md) | C1 TS/SQL selection; C3 cell runs; C4 lineage | C5 UI/background runs; C6 independent cells |
| P11 | [Executable wiki](P11_WIKI.md) | C2 merge profile; C3 cell loading; C5 revisions | Browser C1 implementation; C4 inspection; C6 pure cells |
| P12 | [Log investigator](P12_LOGS.md) | C1 search environment; C5 windows | C4 replayable incidents; source cursor persistence in host |
| P13 | [Data studio](P13_DATA.md) | C1 data engine; C7 conversions | C4 lineage; C5 live imports |
| P14 | [Evidence atlas](P14_EVIDENCE.md) | C1 search bindings; C7 source records | C4 run-evidence import; C5 updates |
| P15 | [API workflows](P15_WORKFLOWS.md) | C1 service bindings; C5 recovery events; C4 effects | Host-owned durability; no required new core lifecycle |
| P16 | [Experiment laboratory](P16_EXPERIMENTS.md) | C2 seeds; C3 child runs; C4 result evidence | C6 batch execution |
| P17 | [Test explorer](P17_TESTS.md) | C3 checked runs; C4 failure evidence; C2 reproduction | C6 independent scenarios |
| P18 | [Publisher](P18_PUBLISHER.md) | C1 renderers; C7 document values | C4 evidence links; C5 live refresh |
| P19 | [Scheduling engine](P19_SCHEDULING.md) | C0/C7 finite schedule; C5 interactive replanning | C1 live calendar environment |
| P20 | [Repository migration](P20_REPOSITORIES.md) | C1 workspace bindings; C3 validation/run; C4 change evidence | C5 interactive progress |

## Delivery order without a platform prerequisite

The [portfolio infrastructure plan](../INFRASTRUCTURE_IMPLEMENTATION.md) provides the concrete 17-patch sequence, source touchpoints and release gates. The following application order identifies consumers, rather than a competing infrastructure sequence.

1. Start finite P02 and P09 scenarios while extracting C1/C2/C3 and a minimal C4 sink. They test semantic interpretation with little host machinery.
2. Use P01 and P03 to validate actual engine/environment behavior, large native data and long calls. Extract a process helper only after both use it.
3. Complete C5 with P04 and P12. Establish waiting/closure, short steps and stale-result handling before introducing parallel streaming.
4. Build P05, P10 and the small P06 inspector using C3/C4. This makes tooling useful to subsequent development.
5. Grow the remaining plans as their listed dependencies become usable. Introduce C6 for measured throughput needs; complete P11 browser/collaboration integration after P02 semantics and C3 loading work.

A full portfolio does not require one host to implement everything. Each programme specifies its engine/library dependency closure. Add a new core feature only with a concrete failing scenario and a comparison against the smallest library/embedding alternative.
