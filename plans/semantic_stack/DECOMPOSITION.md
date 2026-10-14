# Semantic stack curriculum: natlang-native decomposition

## Unit and target

Two independent authored slates provide at least 200 distinct, substantial natural-language programs across at least 15 domains each. A source adapter makes runnable Program/2 cases with typed natural-language roots, callable folders, current-item scoped ad hoc `nl` literals, and typed child outputs. Source cases are generation inputs, not admitted training trajectories. Teacher trajectories pass exact program/oracle checks, source decontamination, semantic review, and corpus registration separately.

## Functions and boundaries

1. **Understand the task contract (root `.nl`).** Read the objective and named data layout. Decide which evidence is germane and what to ask each item. This is natural-language policy, varying by authored program.
2. **Inspect one scoped item (runtime-generated `nl` literal).** The root constructs a typed literal at the point where the item and current accumulated state are known. Its prose uses a short numbered plan: read the item, interpret the relation in the task rule, preserve a quote or path, and return a typed finding. Avoid brittle keyword matching. This is a distinct natural-language call per item.
3. **Revise accumulated state (`iterateOn`).** Carry the typed finding and existing state through an iterative step. The step may create further focused literals to reconcile conflict, fill a missing relation, or evaluate a candidate. The loop terminates on a structural measure: number of uninspected items or explicit finished predicate. No arbitrary global timeout is part of the language contract.
4. **Synthesize and verify (nested named `.nl` helpers).** The root calls the first named stage. Each later stage is in a callable child folder of the preceding stage and executes while that parent call is active. The final stage returns its grounded finding up the stack; the root assembles the declared result. Semantic synthesis is natural language; exact serialization, hashing, source snapshot and oracle checks are crisp. Sibling `.nl` files under the root do not implement a nested call chain: a sibling is outside the active child's callable scope.
5. **Build a case (crisp source adapter).** The adapter checks IDs, types, file paths, nonempty evidence, source identity, split and provenance, then emits Program/2 JSONL. It does not decide a semantic answer or fabricate a teacher trajectory.
6. **Review and generation (existing teacher pipeline).** A generated case goes through source validation, independent exact/semantic checks, held-source decontamination and teacher collection. A passing execution is not automatically admitted as training data.

For the typed ecology variant, an intermediate named stage owns and checks its
semantic contribution, then calls the next named `.nl` function and returns
the child's complete typed state in the same final eval action. The source
adapter renders the actual child name and typed input binding; it does not
decide the contribution or substitute a case-specific answer. This source
presentation follows a live staged-result recomputation and remains diagnostic
until its model-facing measurement is complete.

## Why the crisp pieces exist

JSONL construction, SHA-256 provenance, deterministic source grouping, type/schema validation, source-decontamination and exact file/oracle comparisons require exactness. They do not choose domain policy or replace the natural-language functions. The builder extends existing `curriculumCase` and collection contracts; it does not add a second runtime or a separate evaluation mechanism.

Several program questions may share one evidence world. The builder assigns those questions the same `source_group` and split group from `source.world_group` or `source.split_unit`; their program IDs remain distinct source IDs. This prevents a shared world's evidence from leaking across training and held splits.

## Quality requirements

A catalog entry counts as a distinct program only when its semantic relation, input structure, control flow, or output objective differs substantively. A domain name swap around the same classification loop does not count. Each runnable source must make the decisive evidence visible in inputs, keep oracle facts out of the opening, use scoped arguments instead of dumping the full folder into a child, and expose a result that can be graded. References replay only as source validity evidence; teacher quality and neuralese call topology require separate review.

The two early 3–4-record seed pilots exposed builder defects: the first used helper filenames that did not match the authored names; the second rendered stages as siblings, so a child could not call the next stage. Both pilots are rejected as builder diagnostics, not model negatives or DPO pairs. The corrected builder was preflighted with actual `read_code` and nested runtime calls through the full Astra and Sol chains, then all expanded cases passed exact scripted Program/2 replay. Live execution, semantic answer review, and training admission remain separate gates.

The first 24-record diagnostic exposed a further source rendering error: a parent-level step saying to call a named stage had been copied into that stage's own body. This made a self-call seem required. The adapter now renders such a step as the stage's imperative task, checks every generated helper for self-call directions, and preflights the model-visible `read_code` source at each depth. The affected attempt is rejected as a builder diagnostic, never a model failure or preference pair.

The next diagnostic showed that `item: string` lets a root serialize an entire directory as one item. The typed-record variant gives each named Astra helper `item: EvidenceRecord` (`path`, `text`) and `context: EvidenceContext` (objective, rule, relevant related records, prior and current stage findings); each returns `EvidenceFinding` with `sourcePath`, finding, evidence IDs and unresolved claims. The root's natural-language plan advances one source path per `iterateOn` step and uses unprocessed-path count as the structural measure. This is a source-presentation variant pending live teacher qualification; exact scripted replay and a typed nested-call preflight do not grant trajectory admission.

The typed-record live trace processed two records with about 35 model turns and 73 request starts, carrying grounded source IDs through the nested chain. One bad dynamic-literal scope reference recovered after the runtime's binding error. The source itself has 24 relevant records, so a broad relevance filter would not reduce this case. The v4 aggregate trace was stopped once its aggregate-as-string topology was established; neither diagnostic is a model-negative example.

The next source-presentation candidate groups records by their actual source folder. The root uses `iterateOn` over the sorted folder frontier, which decreases by one; a batch is `EvidenceBatch {folderPath,records: EvidenceRecord[]}`. Each named stage receives that typed batch and returns `EvidenceBatchFinding {folderPath,findings: EvidenceFinding[]}`. Its natural-language policy interprets within-folder relations and may create an ad hoc typed literal over the current batch; exact code only groups paths and checks coverage. The required invariant is exactly one distinct `sourcePath` finding per batch input path, with evidence IDs referring to real source records. The root synthesizes across batches afterward. This retains real directory iteration and a five-deep named call chain while amortizing that chain over coherent source groups. It is an unqualified live variant; exact reference replay is source validity only. If grouping conflates facts or loses per-record evidence, reject the presentation and retain the typed-record design.

## Path from reviewed sources to training

1. Freeze the two reviewed catalogs, including fixture/gold hashes, machine-readable per-field answer comparison, typed stage contracts and generation-ready flags. Rebuild the common Program/2 source once from those bytes and check every emitted helper type, source-group identity and reference replay. The 220 programs share 82 factual worlds; split by world, not program ID.
2. Run `materialize_semantic_stack_priority.py` against the pinned source and review hashes. Expand the authored priority order from the initial 18 only after the additional source reviews finish. The selector never grants training admission. Decontaminate source groups against the registered corpus before teacher collection.
3. Prepare immutable worker queues through the existing Luna campaign preparation and supervisor path. The current preparer's default 60 turns/384 requests/1800 seconds is inadequate for the observed 24-record typed diagnostic; qualify a case-size-aware resource ceiling from live traces and record it in the queue, without treating resource exhaustion as a language failure.
4. Review complete teacher outputs for exact source-path coverage, actual parent-child calls, stage-specific new information carried to children, consumed scoped literals, structural iteration, per-field answer equivalence and grounded alternatives. Source replay and final-answer equality alone are insufficient. Reject unusable rows as data-quality evidence; preserve a failure/DPO input only with a concrete recovery consumer.
5. Admit only independently reviewed correct native trajectories through the existing native-turn materializer, immutable SHA-256 corpus manifest and `training/neuralese_corpora.json`; then sync registered artifacts to DGX. Keep selected-but-unrun, failed, held and admitted counts visible in one lineage manifest so broad new source work does not disappear from the training mix.

The v5 and v6b partial live traces showed source-grounded per-record work, but v6b still made many per-record child calls and some named stages merely forwarded their input. Grouping reduced chain setup, not the full request cost. Both were stopped after their diagnostic questions were answered and remain outside training/DPO.

The v7 source renderer reads authored per-stage contribution contracts. Each stage receives a typed prior state containing every earlier owned field, must add its own source-backed field or a reasoned absence, and passes that state to its child. It rejects a missing prior field, a repeated owned field, and an Astra EvidenceRecord without a visible source ID. The renderer preserves Sol's authored passage-witness and work-frontier types. The declared answer comparator accepts only explicitly unordered array paths as sets, retaining multiplicity and ordered tuple endpoints; canonical witness alternatives remain pending independent review. Scripted replay validates source/runtime shape but does not prove any named stage actually contributed in a teacher trajectory. The v7 presentation and comparator need a small live pilot, then about 48 reviewed live cases per adopted model-facing variant. Until then, all v7 rows remain diagnostic candidates with training admission false.

The first v7 Sol008 diagnostic showed source-quoted contributions in all four
named stages, consumed scoped natural-language decisions, and a final answer
equal to the authored reference. Its collector then failed because the pinned
older runtime snapshot did not contain the new comparator. A retry on the
corrected frozen runtime completed with the same exact answer. Trace review
found that the early stages manually populated case-specific evidence and rule
arrays in `eval`; the scoped literal consumed those prior decisions, but it did
not do all the intended semantic work. The result is correct diagnostic
evidence, held from training until trajectory idiom is reviewed. Freeze the
runtime and bind its manifest hash to every queue before claiming an oracle
outcome.
The ecology01 diagnostic exposed a different issue: a rich scoped child
decision failed, after which the parent forwarded an empty owned field despite
applicable relations. The next presentation should use a narrow per-relation
typed literal and require its returned decision to enter the owned field.
Neither trace is an admitted training row or a model-negative example.

The first v8 ecology queue failed before any model call because its hand-built
single-case row used source index 1 against a one-row selected source. Worker
queue indices are zero based. A fresh immutable v8b queue uses index 0; the
six-case plain cohort was also resealed with indices 0–5 before launch. Both
queues were range-checked against their exact selected-source line counts.

## Relevance-first collection for wide evidence folders

The first source-driven folklore diagnostic spent about 379 turns on seven of
24 records. Its natural-language decisions were useful, but repeating the
whole nested chain for every record would exceed the frozen 600-turn collection
ceiling. This is a source-program decomposition issue, not a model failure.

1. **Screen the whole evidence world (new typed `.nl` callable).** Given the
   objective, task rule, all source records for context, and one source-folder
   focus group, return one source-linked `EvidenceRole` per focus record:
   directly queried, dependency or authority, background, or uncertain. The
   root combines those groups into one full-world ledger. A dependency names
   the record or relation it can change. Uncertain roles stay selected. The
   screen is a semantic judgment; no keyword or filename classifier chooses
   relevance. The first all-24-at-once live screen led the child to attempt
   repeated ad hoc calls and then hand-write a case-specific relation table.
   The smaller focus group is a source-presentation correction; it still sees
   all records so cross-folder dependencies remain available.
2. **Process coherent selected groups (existing named chain).** Group selected
   records by their actual folder and declared dependency links. Send each
   group through the named stages as a typed batch. Keep every selected source
   ID, path, and quote; stages can call focused ad hoc `nl` literals for the
   relations that need judgment. The active chain still returns a typed
   `EvidenceBatchFinding` with all selected inputs accounted for.
3. **Reconcile coverage and output (root `.nl`).** The root checks whether any
   background item is needed by a later finding and reopens it if so. It
   combines the selected findings with the role ledger, preserving unresolved
   questions. A background label is not evidence that its content is false.
4. **Exact plumbing.** Code checks that the role ledger covers every source
   path exactly once, selected groups contain only known paths, dependency
   links resolve, and returned findings cite visible records. These checks
   cannot decide which record matters semantically. The source builder emits
   this as an opt-in immutable variant of the existing Program/2 case; the
   usual teacher collector, oracle and admission path remain shared.

The frozen per-record diagnostic remains a baseline. The relevance-first
presentation needs matched live samples and a reviewed correctness result
before it replaces it. If screening selects all 24 records, the program may
need relation-level grouping or a higher collection resource ceiling; the
screen must not hide hard evidence to save requests.

The first source-disjoint Sol plain pilot exposed a different handoff problem:
the root and named stages carried the complete bundle, but an inline
`RelationDecision` child saw only a narrow prior state and called a later stage
with a fabricated empty bundle. The transport preserved the source exactly;
that empty value was model-authored. The opt-in
`--typed-literal-evidence` variant asks the owning stage to gather supporting
source-linked evidence from its original bundle, invoke the relation literal
on that evidence, append its returned typed decision to `StageState.relations`,
and pass the original bundle to the next named stage. It is a source variant
awaiting matched live measurement, not an accepted replacement for plain
generation.

## Application-scale semantic projects

The additive high-depth slate is a different source shape: a complete natlang project with roughly 10–30 authored `.nl` files, topic-specific callable folders, explicit typed interfaces, service declarations when outside-world work is needed, and real multi-step tasks. A project manifest selects an immutable source tree, task input snapshot, declared output/checker, provenance and split world. Its adapter snapshots the files into Program/2 without rewriting the author's call graph or forcing a generic chain. The root and modules remain natural-language functions; exact source hashing, path containment, type checking, replay and result checks are crisp. Full teacher trajectories, including child calls and tool outputs, are collected through the existing teacher pipeline. Existing 220 authored programs remain eligible when valid; their difficulty is a mix label, not an admission criterion. New project cases must show actual stage contributions, source visibility and correct outputs before training admission.

### One trusted exact verifier boundary

Application decisions and policy remain in the `.nl` call graph. A project may also declare one external verifier for exact properties of its returned value: source identities, quotations, arithmetic, state conservation and allowed effects. The package snapshots the verifier's repository path and SHA-256 outside the model-visible project. The existing teacher collector calls the pinned verifier with the returned value, authored oracle, and visible input/fixture snapshots; it does not ask the verifier to synthesize an answer. A verifier verdict may accept exact invariants while marking semantic entailment for independent review. The canonical reference replay bypasses this verifier only to prove source/type plumbing; separate canonical and mutation probes qualify the verifier itself. A missing or changed verifier is an infrastructure error and must not become a model-negative row. This crisp boundary is justified by exactness and uses the existing collector/oracle outcome, not a second scheduler or training admission path.

Independent review receipts are source facts, not model-visible policy. The
packager checks declared receipt paths and SHA-256 values exactly before
reporting a source as reviewed. The semantic judgment itself remains with the
independent reviewer; the package only records that the named frozen receipt
matches. Source-reviewed status permits a live diagnostic and never admits a
teacher trajectory or training row.

### Typed ecology evidence handoff (opt-in v12)

The folder reducer owns the finite folder frontier and exact source ID/path binding. Its first natural-language stage reads each `EvidenceRecord` and invokes a scoped interpretation literal, producing a typed `InteractionAssertion` for every record, including background or unresolved records. The second stage receives only these typed assertions and exact witness snippets, invokes a scoped relation literal, and returns one `EdgeDecision` per assertion plus established edges. Later natural-language stages assess causal support, edge-specific scope and the global signed graph using the accumulated typed evidence. Only the root appends completed-folder fields and checks quote substrings against the original records. These exact checks are plumbing; relevance, referents, direction, sign, scope and uncertainty stay in natural-language calls. The emitted source variant is separate from the catalog and preserves its fixture, gold and oracle. A source/reference replay proves wiring only; live stage contribution and final semantic correctness remain unqualified.

The first live typed-handoff trace showed that Stage1's global signed-effect rule caused a clear reported reduction to be marked unresolved. Stage2 also borrowed an edge from another assertion, and Stage3's aggregate child failed to return its own assessment. The v13 task boundary narrows Stage1 to extracting reported claims and perturbations, Stage2 to deciding the primary assertion's own directed relation with its own witness, and Stage3 to one edge at a time through a scoped causal-support decision. Every missing assessment is kept as a witnessed repair obligation. This changes natural-language stage policy and call granularity; exact code still only binds IDs, checks quoted evidence and carries returned fields.

### Folder cursor and cumulative graph (opt-in v15c)

The v15b live trace showed a real folder-selection bug: extra arguments to `iterateOn` stayed fixed, so the first evidence folder was processed again. The root now carries `FolderReductionState {nextFolderIndex,completedState,processedSourceIds}` as its sole changing iteration state. Its numbered plan binds the sorted folder collection once, selects `sortedDirs[state.nextFolderIndex]` each step, awaits the existing named natural-language stage chain, appends that chain's actual typed contributions, and advances the index once. The finite measure is the number of folders remaining. Exact code checks unique source IDs and paths, quoted spans, and final coverage of the visible ID universe; these are mechanism checks, not ecological judgments. The five named stages and scoped literals still decide claims, edges, causal support, scope and signed paths in natural language. The v15c renderer adds only the root plan and state type, preserving the v15b helper bodies, fixture and gold. Source replay proves wiring; a fresh teacher trajectory must establish that the cursor advances and that returned semantic decisions drive the final graph before any training admission.

The v15c trace confirmed the cursor advanced, then exposed an inconsistent return contract. Earlier-folder evidence belongs in `context.priorCompletedState`; `input.prior` and the returned Stage1–4 arrays carry only the current folder's new decisions. The final graph stage may read both to compute one global `effectAudit`, but returns the current-folder arrays with that audit. The root validates exact ownership and derivation of these arrays before appending them once, while allowing earlier source quotations as supporting witnesses for a current decision. This is exact state transfer and provenance checking. The ecological meaning of claims, edges, scope and signed paths remains in the named natural-language stages and their scoped literals. v15d changes this handoff presentation and must be measured live before any row is admitted.

### Non-gating live idiomaticity signal

The policy decision about whether a teacher trajectory expresses the authored
natural-language program remains an independent semantic review. An exact trace
scanner may count evaluation-code length, natural-language literal call sites,
and large authored object/array tables by invocation. It reports locations for
review and never rejects or admits a row. This crisp mechanism is justified by
exact counting of recorded tool calls; it cannot decide whether a typed object
literal is legitimate materialization of a natural-language judgment or a
hard-coded replacement for that judgment. The reviewer must inspect source
grounding and whether later natural-language stages consume the returned value.
