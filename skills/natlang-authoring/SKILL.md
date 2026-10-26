---
name: natlang-authoring
description: Author, refactor, review, evaluate, and adapt natlang code — inline `nl` calls in TypeScript, named `.nl` functions with their callable folders, and `natlang.d/` application context. Use for semantic algorithms, reducers, application logic, or measured instruction and program-guidance optimization, including programs intended for small interpreter models.
---

# Author natlang code

You are the capable program author; the interpreter may be a small model. Put the algorithm, decomposition, data contracts, and recovery decisions in source so execution does not depend on the interpreter inventing an architecture.

Natlang is TypeScript with natural-language functions. Ordinary TypeScript owns exact computation, I/O, and mechanical sequencing. Natural-language functions own interpretation, planning, semantic merging, choosing among options, explaining, and semantic decisions about which work to perform next. Choose the boundary deliberately. A host that computes everything and asks natlang to approve wastes it; making the model do arithmetic by hand wastes its context.

For a student served with `program-header` rendering, native natlang calls omit their generic tool schemas because that tool surface was trained into the student. A separate application calling the student directly should still supply schemas for its own tools; serving retains those schemas in this mode.

## Establish the contract

Locate the natlang checkout or installed `@natlang/node` / `@natlang/browser` version first. Use its declarations (`dist/index.d.ts`), tests, and `natlang check` as the authority; if prose here disagrees with executable behavior, flag the disagreement rather than inventing behavior. Outside a checkout, rely on these bundled references and the installed package.

Read [language and source contracts](references/language.md) before creating or restructuring code, [algorithm patterns](references/patterns.md) for loops, reducers, reconciliation, and generated methods, and [verification and diagnosis](references/verification.md) when testing or debugging. The [review example](assets/review/review.nl) shows a named function whose callable folder holds a semantic helper and an exact aggregation; copy the whole `assets/review/` directory.

Put a property of a value in its type rather than in a guard sentence: read [constraints belong in types](references/refinements.md) for `Is<T, "predicate">`, when to use it and how to phrase the predicate.

To let the runtime keep an intermediate value as Neuralese between stages, write the chain so its data flow is explicit: read [orchestrators that can fuse](references/fusion.md).

For evaluation suites, instruction optimization, or trainable program guidance, read [adaptation and evaluation](references/adaptation.md). It covers component selection, independent scoring, fresh fixtures, bounded search, report review, and artifact activation. Ordinary authoring does not require an optimization run.

## Build a real program

1. State the behavior as typed inputs, results, and observable scenarios, including ambiguity, empty inputs, and partial failure. Types check structure, not meaning.
2. Put semantic control flow in natural-language instructions: which hypothesis to test, which dependencies to reopen, when to change strategy, and what an unresolved result permits next. Use TypeScript for exact transport, arithmetic, and mechanical sequencing. Use inline `` nl`…` `` for a focused decision, or a named `.nl` function when it deserves its own file and typed children. Moving a TypeScript algorithm verbatim into an `.nl` body does not make its design NatLang-native.
3. Give each natural-language function one coherent responsibility, a precise signature, and the helpers it may call in its callable folder (`foo/` beside `foo.nl`, or `natlang.d/` for inline calls in application code). Put only the program's own code there; the systems it acts on (stores, boards, simulators, verifiers, APIs) are services with declarations, scoped to the functions that should use them.
   When one named function must call another during its own active call, place the second `.nl` inside the first function's callable folder. For a chain `root -> inspect -> decide`, use `root/inspect.nl` and `root/inspect/decide.nl`; two sibling files under `root/` do not give `inspect` access to `decide`.
4. Write each instruction so a small interpreter can take the next step: name the data and how to reach it, state the step a loop repeats, and keep the words for finishing statuses ("missing", "cannot") for real blockers. See [patterns](references/patterns.md).
5. Keep exact work exact: helpers in callable folders are ordinary TypeScript under a finite-iteration policy; application code outside them is unrestricted.
6. Run `natlang check` (types, `nl` signatures, callable-folder policy), then exercise the real source through the runtime. Distinguish checks, scripted wiring, live-model runs, and semantic evaluation.

## Design independent applications for a diverse corpus

Start each application from its own users, decisions, persistent domain records, events, and failure modes. Choose its public operations and typed intermediate structures before designing helper functions. A shared request envelope, a fixed number of stages, or identical prepare/decide/construct/audit children across unrelated programs creates repeated architecture even when domain nouns change. Do not use one generator or common semantic scaffold to stand in for those individual designs.

Depth must represent real work. A dependency ledger may need interval-scoped edge traversal; a scheduler may need a finite candidate frontier; a tutoring application may need competing diagnostic hypotheses and an interrupted-session checkpoint. Give each distinct judgment its own typed child when the parent needs that child's result. Reference actual calls with their supplied inputs, pass returned values to downstream decisions, and preserve unresolved outcomes. More files and unused helpers do not establish deeper execution.

Expose meaningful domain operations as the application's interface. Internal parsing or policy-selection helpers are not automatically public endpoints. Let the program own its request types, checkpoint state, result, and recovery operations; optional exact persistence or HTTP infrastructure must not prescribe those semantic contracts. A type-compatible saved checkpoint is necessary for resume: a version number cannot substitute for the domain state that the next call expects.

Validate the frozen program as well as the working directory. Native `uses:` paths need the package boundary (`natlang.json`) in the snapshot. Sibling functions require an explicit supported import or the correct callable-folder relationship. Native type grammar is narrower than TypeScript: use optional fields or `null`, not a type named `undefined`, and use `boolean` rather than boolean literal types. Run the native loader, callable-scope check, fixture type check, and interface-state compatibility check; a compiler-only pass cannot establish those contracts.

Preserve independent oracle answers and observed fixture evidence while changing API representations. Keep source ownership for every original domain invariant and operational scenario, without forcing those obligations into a common pipeline. Static potential-call graphs document source structure; only captured model execution demonstrates actual child calls and result consumption. Do not claim learned quality, training admission, or production deployment from structural checks.

## Improve instructions with measured adaptation

Use adaptation when the callable contracts and decomposition are sound and the task is to improve instructions or program guidance against explicit scenarios. Fix missing capabilities, types, capture names, or helper structure in source first; adaptation changes static instruction text, not those contracts. Label authored inline sites when stable selection matters, keep expected answers outside model-visible code, and use fresh fixtures with independent scoring. Review held-out gates, helper coverage, actual usage and uncertainty before adopting an artifact. A retained baseline is a valid result.

Keep search overlays separate from authored source. Load a compatible artifact explicitly, use `adaptation: null` for a baseline task, and revalidate after source or model changes. Read the [adaptation reference](references/adaptation.md) for SDK and CLI examples; use `natlang-integration` when wiring an application host or browser deployment.

## Preserve the execution model

When the host selects semantic execution, await callable-folder module entry
and inline native boundaries. Captured mutable bindings are shared cells:
complete the current step before consuming the next Fold event, and use bounded
Map only for independent work. A capture-conflict means another writer changed
the binding; reread current state and retry the owning computation deliberately.
Do not retry a failed native write merely because the surrounding eval failed.
After a realm-authority change, restart the task and reload its modules instead
of keeping old exported functions. Keep source and invocation identities in
controlled fixtures, and distinguish those fixtures from learned-model results.

- A natural-language call is an ordinary awaited function call returning a checked, typed value; a failure rejects with `NatlangCallError` (`outcome`, `detail`, `trace`). Catch it where the application can recover.
- When a stage invokes an inline literal for one relation, give the literal its source-backed item, supporting evidence and prior semantic context. Its typed result is the relation decision; the parent appends that returned decision to its carried state and passes the original complete input to the next named stage. A narrow literal cannot reconstruct a full source bundle from partial state.
- An inline `nl` reads the variables its instructions mention by exact name, live at call time. Mutable captures (`let`) are written back after a successful eval; a concurrent change makes that eval fail with `capture-conflict` so it retries against the current value. Rebinding a captured `const` is a compile error.
- Signatures are inferred before the model runs: from the contextual type, immediate call arguments, and later uses. When nothing determines the result type, `natlang check` reports `nl-unknown-return`; annotate the binding or use `nl<T>`.
- In eval, an inline result annotation can use only type names declared in that eval scope. If `nl-undeclared-type` reports that a name is missing, declare the type in scope or write the needed structure directly in the annotation; an alias mentioned only in earlier instruction text does not type the child.
- A call with formal parameters exposes those arguments by their declared names, with `inputs` and `read_inputs()` available when not shadowed. A zero-argument call has no caller-input bindings; declare local variables normally instead of expecting an implicit `inputs` object. Add a formal parameter when the function needs caller data.
- For `nl.with<ResultType>({ fixedContext })`, the single type argument is the child's result type and the object supplies captured context. Pass changing values to the returned child function, for example `const summarize = nl.with<string>({ rubric })`Summarize the supplied record against the rubric.`; `const summary = await summarize(record);`. If `finish:true` reports `missing-fresh-result`, omit `finish` or set it to false for inspection or staging; use a fresh typed final expression or explicit `return` when finishing.
- A natural-language function may not appear in its own chain of callers. A TypeScript function may call itself only on a smaller argument: a part of its input, a shorter array or string, or a smaller non-negative integer (walking a tree recursively is fine). Concurrent sibling calls (`Promise.all`) and repeated sequential calls are fine.
- Callable-folder TypeScript and eval code use finite iteration: `for...of`, counted `for (let i = 0; i < n; i++)` (the bound is read once, when the loop starts; `&&` may join it to an early exit such as `i < n && !found`), and array methods. `for...of` takes an array, string, Map or Set, plus `entries()`, `keys()` and `values()` of those and `string.matchAll(regex)`, which are finite views of a fixed collection; spread any other iterable first (`for (const x of [...iterable])`). `natlang check` reports a refused loop source with its location. `while`, `do`, `for...in`, open `for(;;)`, generators, hand-written iterators and `setInterval` are rejected. Open-ended refinement uses `iterateOn`, which records every step and reviews progress; a step can wait with `await new Promise(r => setTimeout(r, ms))`. Timers still pending when a call finishes are cleared. `for await` consumes async iterables that services, packages or `fetch` provide.
- Await the natural-language calls you start. A call that fails stops the calls it started; in eval, calls still running when the eval ends, or when its `timeout_ms` passes, are stopped (quietly for the losers of `Promise.race`). Service calls already made stay made.
- Host capabilities come from services: `import { wiki } from 'natlang:services'` in callable-folder code, or the same names as bindings inside eval. Every service call is traced as an effect and is not rolled back. A service's declaration is what the model knows of it; a scoped service is usable only in its functions' calls.
- Nothing the interpreter can reach is private: it reads callable items, service declarations, and packages with `read_code`, and edits callable items with `edit_code`. Never let correctness depend on it not reading source; limit what it can use (services, scopes) instead.
- Native input and scope listings show at least 4,096 characters per value (more when the call's tool-output size is larger). If a listing says a value is cut off, its named eval variable still holds the complete value; inspect the needed range with ordinary operations such as `state.slice(start, end)`. A tool output shows up to an eighth of the call's context budget (about 3.5 characters per token; never fewer than 2,000 characters: 7,168 at 16k tokens, 28,672 at 64k), so a long item usually fits in one read; only longer output is cut off, and its note names the `read_page(id, n)` call that shows the next page of the same size.
- When `read_page` says no tool output was cut off, inspect the value already in scope with `eval`; page IDs exist only for actual cut-off tool outputs.
- Directory reducers (`kind: directory-reducer`) receive a `Folder` as their first argument and file tools rooted there. A direct call returns the typed value and discards file changes; `folder.apply(reducer, ...args)` retains the committed changes.
- For a wide evidence directory, a typed natural-language child can screen source-folder groups for direct relevance, dependencies, background, and uncertainty before deeper per-relation calls. Give each call the complete source inventory for cross-record context and a smaller focus group whose roles it returns; combine the returned roles into exactly one per source record. Retain exact ID/path coverage; select uncertain records, close dependencies, and reopen a background record if a later finding needs it. Let NL judge relevance and authority. Use exact code only to check coverage, known IDs, quotation spans and grouping.
- If the result uses derived record IDs, map keys, case-folded entity names, or another source-specific encoding, put its derivation in a visible output identity convention in the root `.nl` contract. The oracle must never require an encoding that appears only in hidden metadata.
- In a directory reducer over item files plus a shared task contract, bind each value from the file that owns it before creating an inline `nl` call. Preserve each item's declared answer options; use task-level options, criteria, or levels only when that item does not supply its own value and the shared task file actually declares one. An explicit empty list remains empty. If no criteria are supplied, use the declared options as the answer domain without making a rubric from the labels.
- When semantic graph edges or directory findings join entities named across files, state the output identity rule in the program. Case-only spelling normalization can be exact; keep source quotations unchanged. Synonyms or ambiguous referents still need a source-grounded natural-language decision.
- A directory reducer whose later folders can change an earlier conclusion should carry typed prior findings into each `iterateOn` step. Each named function adds its own new source-backed contribution; the final judgment consumes that carried state. Keep exact array append and key normalization in code, and leave semantic relation judgments in natural-language functions.
- When a later named stage should depend on an earlier semantic extraction, give that later stage the typed extracted assertions and exact witness snippets as its input. The first stage reads raw records and binds source IDs/paths around the actual awaited literal result. A relation stage appends its own typed decisions from those assertions, including unresolved outcomes, and downstream stages consume those decisions. Keep quote substring checks and one-decision-per-record coverage exact in the parent.
- Separate extracting a reported claim from judging its causal sign or support. A scoped relation judgment owns only the relation asserted by its primary source record; related assertions supply referents and conditions, with their own relations decided separately. For a later causal-support pass, invoke one focused literal per established edge and append its returned assessment before calling the next named stage. A missing assessment stays a witnessed repair finding.
- Long productive runs are intended. Do not add arbitrary turn, token, or nesting limits to make a test finish; deployment budgets are runtime options.
- If a native call has no staged result and the provider proves two consecutive immediate empty EOS replies (the second after the ordinary missing-result recovery message), the runtime reports `model_incomplete: terminated_without_result`. Treat this as an incomplete model answer, not a provider outage; retry with a corrected task or model context. Empty text alone is not EOS proof.
- Report real blockers. Do not return invented success, manufacture receipts, weaken assertions, or treat an unknown effect outcome as safe to retry.

For embedding in an application, pair this skill with `natlang-integration` when available; this skill is usable on its own.

Persistent eval helpers support TypeScript `as const` assertions for literal and tuple results; the assertion needs no captured type binding. Other referenced named types must remain available in the call scope.

When a conversation carries a `native_state` content part, preserve its ID as an opaque exact-state reference through history and tool arguments. Same-model state can be spliced only when checkpoint, tokenizer, embedding and architecture identities match. A `neuralese-unknown-native-state` response means restore/upload that referenced artifact and retry the same request; an identity mismatch permits only explicit text retokenization, never reuse of its foreign tensors.

Tool-call argument JSON is decoded once before rendering, so a JSON string remains a string even when its contents resemble another JSON value. A native-state value nested in tool arguments is imported only when its ID is a complete canonical `nzs1_` content ID; keep the exact ID from the produced state receipt and restore that artifact before retrying.

## Recover from service argument-shape errors

A service declaration defines its positional arguments. When a service error gives the expected typed call, inspect that method with `read_code`, bind each declared argument separately, and retry the stated positional invocation. For the story witness service, use `storyEvidence.checkWitnesses(snapshot, witnesses)` with a `SourceSnapshot` and a `Witness[]`. Keep this recovery in the boundary error and skill guidance; the application prompt need not enumerate invalid call forms. Natural-language stages still decide what the witnesses mean; the service verifies their exact source identity and quotation.

## Carry exact provenance through semantic calls

Pass the original typed source record to a named evidence interpreter. Give context/declaration parsing and event interpretation separate typed responsibilities when they produce different data. A scoped NL judgment may return meaning arrays and exact quote bytes; its parent binds witness source ID and path from the original record, then checks quotation containment before consuming those findings. If exact witness checks fail, retain an unresolved evidence finding and inspect the original record rather than inventing provenance. Context canon remains input to authority interpretation even when it is excluded from an event frontier.

A nonempty source-grounded semantic contribution requires at least one nonempty supporting quote whose exact source check passes. An empty witness array cannot certify that contribution; retain the missing support as unresolved.

## Separate symbolic readings from operational bindings

When one semantic stage reads source terminology and its parent materializes an executable plan, give the child a typed symbolic reading and the parent a distinct typed binding contribution. A parent's scoped NL judgment joins that reading with visible code definitions and compares displayed wording with counted meaning. Preserve disagreement as a diagnosis without silently repairing an implementation. The executable uses the returned binding; comparison consumes the symbolic reading and meaning relation. Missing or conflicting definitions remain unresolved rather than selecting a code by label.

## Compile evolving prose as a semantic application

Separate source edition/authority selection, rule/exception interpretation, carried state inference, callable/literal lowering, NL module writing and scenario execution into typed transformations. Preserve historical event-time clauses alongside later replacements, and retain duties outside a scoped waiver. Generated guards return actual typed state effects consumed by scenario replay; a correct source model cannot certify an emitted body that dropped those rules. Diagnose earliest divergence and revise generated policy or its missing state handoff, keeping original authority/evidence unchanged. Missing authority or decisive evidence remains unresolved rather than becoming a fabricated default.

For each authority judgment, pass both the original act and the governing policy clauses into the scoped NL decision. In numbered pseudocode, inventory candidate issuers and their exact witnesses, test each against the policy's authentication and scope conditions, then return authorized true, false or null with the decisive quoted evidence. A parent's reputation guess must not choose among candidates before the child sees them. Keep authority distinct from the later question of which event an authorized act targets.

When checking generated policy, judge its actual emitted behavior independently from the source-required behavior. Keep separate typed state/effect channels so an ideal source rule cannot silently repair a faulty emitted guard or inherit its fabricated facts. Preserve the mismatch as a counterexample. Generated-code path/span evidence is distinct from original source witnesses. Label symbolic NL replay separately from actual generated callable invocation.

## Preserve conditional decision identity

When judging alternatives, store each actual child result with the original candidate IDs, context and hypothesis before deriving downstream bindings. Numbered NL steps then resolve accepted, contradictory and unresolved rows within that hypothesis. Distinguish consistency under a proposed hypothesis from knowledge of which hypothesis is actual. A source-local identity fallback needs its own naming evidence and cannot replace an existing candidate judgment. If a returned binding loses its candidate identity, inspect the owning table and rebuild from actual returned rows; retain missing conditional premises as unresolved.

In numbered pseudocode, perform all required evidence/state/comparison checks before the whole-function return. A decided emitted permission or effect needs at least one nonempty exact generated-code quote; an empty evidence array cannot certify its symbolic behavior. Keep conditional early returns local to their explicit branch and preserve typed unknown results.

## Interpret simultaneous events with owned snapshots

Separate announcement-captured references, resumed current state and departure last-known state in typed fields. Replacement judgments consume the actual current rewritten parts while sibling eligibility reads one immutable before-batch snapshot. A separate NL join reconciles simultaneous proposals before exact application. Trigger judgments consume resolved quantities and departures, returning a recipient and consequence that remain fixed after queueing. For finite closed turns, inventory source-authorized work capacities first: consuming an active slot decreases remaining capacity even when it activates another existing slot. Missing source choices remain unresolved rather than being resolved by storage order.

## Bind semantic entity names to source forms

For a casefold-only identity contract, let the NL reader select complete source noun phrases and return sourceForm/key pairs. The parent verifies reported lexical spans and lowercase equality; it does not infer nouns, singularize terms or resolve synonyms. Derive downstream entity keys from those verified forms and preserve grammatical number. Related assertions may resolve the primary claim's referents or conditions while retaining their own independent edge ownership. A narrow scoped judgment returns its own typed decision; its named parent owns subsequent full-state calls.

A symbolic generated-code interpreter needs the actual callee/type-file closure and actual runtime data referenced by the emitted body. It follows that body to select/read data; ideal branch guidance, expected results and reference-state atoms are excluded. A missing actual callee or input is unresolved, never reconstructed from what the source procedure ought to do.

## Separate object existence from directed permission

When a source describes a physical connection and travel restrictions, give existence an explicitly physical typed meaning. NL interpretation owns whether a connection exists, which ordered direction is allowed and which carried-state conditions apply. Attach reverse-prohibition witnesses to the present one-way connection. Compare physical identity separately from directed traversal; preserve all negative evidence while reconciling a candidate world. A failed comparison calls for a scope repair, not deletion of inconvenient claims.

For event certificates, distinguish generated work-slot identifiers from the original source identity that authorized the slot. A rejected one-use trigger still reports its source-owned capacity; consumption is a separate returned field. The parent copies these identities from the selected typed rule/slot before consuming the NL applicability result.

Procedure Compiler code-closure attribution: an emitted-behavior literal returns generated spans with the actual closure file path and exact nonempty quote. The parent checks closure membership and quote bytes before consuming a decided permission/effect. Keep original source witnesses in their separate domain. If a quoted callee is absent, retain unresolved behavior and recover by supplying the actual emitted callee file; source-required policy remains in the independent judgment.

For symbolic interpretation of generated modules, discover the actual transitive code closure with an NL file frontier. Preserve caller-path evidence and distinguish invoked calls, defining type files and explicit runtime source reads from intended plans or explanatory mentions. Activate only original unvisited files; report a missing referenced file by its caller/target identity and recover by supplying that actual file. Do not substitute ideal policy as emitted behavior.

For semantic order disputes, use the same typed NL transition function on isolated candidate state. Consume actual preview outputs in a separate NL equivalence/source-choice judgment. Keep source legality, semantic equivalence and submitted choice as distinct typed findings. Preserve a missing consequential choice as an uncommitted result, with its alternatives visible to the consumer.

Type-closure resolution needs actual defining declaration text, not file names alone. A typed dependency carries exact caller evidence separately from exact target-definition evidence. A missing definition reports the requested symbol and actual target path; recover by supplying the defining type file, preserving unresolved status until its original quote is verified.

For a semantic graph audit, make one NL function own a typed frontier transition and its named parent own iterateOn. Each progressing step consumes a fixed source-derived node identity and returns witnessed paths/findings; a blocked dependency returns an explicit unresolved terminal state. Narrow local condition judgments to their own returned type. Preserve source polarity at one declared boundary so root synthesis cannot apply it twice.

For survey extraction, an NL inventory returns original reading-unit IDs and scope questions, then directly awaited per-unit NL children return typed semantic meanings. Parent arrays consume those actual results; they do not precompute passage, landmark, gear or sensor fields. Bind clause-keyed exact quotations to immutable original source identity before downstream use. Frame-dependent readers receive the actual FrameSurvey; local origin and global anchor retain separate coordinates. Missing quote/endpoint errors identify the reading unit and original clause; recover by supplying its actual frame/supporting evidence and re-invoking that child. Internal absent/unknown passage costs may be null; executable edges require a known source-permitted cost.

Put a narrow scoped judgment's numbered steps directly in its NL literal when the child needs only one typed input. A captured instruction string is data and adds a separate interpretation boundary; prefer the direct template for a small fixed judgment. When a named stage's final action only forwards its already completed typed contribution, finish that first awaited child return in one eval action with `finish:true`. If a result has already been staged, finish the actual returned value rather than repeating its producing call. Preserve preceding semantic checks and owned contributions before forwarding.

For an experimental-design program, preserve joint hypotheses and their source prior masses across typed likelihood updates. Distinguish physical state uncertainty, observation resolution and the requested goal class; a solved goal can retain unresolved internal assignments. A finite policy closure may grow its pending queue, so terminate expansion by a finished predicate over a bounded original state universe, then value the fixed dependency graph with a decreasing unvalued-node measure. Keep semantic prediction, feasibility and candidate comparison in NL. If an observation likelihood or needed assay interpretation is missing, report unresolved weighting and recover the original scope evidence before claiming exact optimality.

When an experiment separates the focal subject from its physical participants, declare those roles explicitly in the visible typed/output contract. Reciprocal direction changes the parent roles without silently changing which accession the investigation concerns. Keep descriptive compartment wording semantic unless the source supplies a finite code.

A valid observation of another compartment can still update a joint hypothesis through shared latent variables. Apply its actual source-predicted likelihood before marginalizing to the requested goal. Use neutral likelihood only for established information absence or unusable measurement; unresolved scope/referents retain unknown likelihood and require their missing evidence.

In a folder reducer, carry the current folder cursor and accumulated semantic result in the iteration state. The step selects the current original folder from that cursor, consumes the actual child return, and advances once. Stable context and extra iterateOn arguments remain unchanged across steps; they do not themselves select a later collection item. A repeated-source-ID error identifies the processed record: recover from the carried cursor and preserve completed semantic state before the next call. Use remaining original folders as the decreasing measure and verify complete unique source coverage at the end.

### Perspective-scoped dialogue state and nonrecursive quotation frontiers

For conversation applications, separate original speaker/time, outer reporter, performed versus depicted
speech force, recipient availability, explicit endorsement and verified corroboration in named typed
fields. A quoted promise does not create a reporter promise, and a globally false proposition can remain
a participant's actual endorsed belief. Apply corrections only to the actor who actually adopts them.
Pass real historical views into condition/query children; preserve actual/issuer-knowledge/beneficiary-
belief channels without replacing unknowns by false. Use a typed original quote-span frontier with a
named NL step: disjoint proper interiors have smaller combined character mass. NL self/ancestor calls
remain forbidden. On an invalid span or quote, recover using the identified original turn/range and exact
substring; on a perspective mismatch, recover from the original recipient receipt and historical view,
not a fabricated empty bundle or global final ledger.

Dialogue promise refinement: type activation predicates separately from promised-consequent predicates and consume their distinct actual NL judgments. A false antecedent need not mean a failed unconditional promise. Preserve day-only due hours as null; source rules can still distinguish a strictly later day. Verified evidence must corroborate the endorsed value before it supplies knowledge support. A contrary log may be available while the actor retains a false belief.

For staged folder reducers, give current-batch contributions and completed-folder context separate typed ownership. Every intermediate prior contains only the current batch. A global audit may read both, but its returned appendable arrays remain current-batch deltas; the root performs the one cross-folder append. Supporting quotes may refer to earlier folders without transferring ownership. Derive owned edges from actual current source decisions and preserve their primary witness through later judgments. A terminal graph-frontier child returns graph state, so its parent constructs the batch result rather than forwarding an incompatible result shape.

For belief-policy applications, spell semantic comparison in numbered NL over actual typed alternatives. Separate complete positive-outcome coverage from attractive partial costs, observation equivalence from latent-state equality, and missing evidence from an explicit zero or losing world. Interpret source priority and tie conventions before exact arithmetic; retain equivalent optimal witnesses instead of claiming a reporting tie-break is necessary. Add depth to the consumer that owns this decision, rather than adding forwarding leaves or source-specific assertion tables.

Before asking a semantic child to interpret a readout, priority or scope, include that contract in its typed input. A prediction name or witness collection is not a substitute for the actual assay contract. A comparison that receives only candidate completeness consumes that upstream judgment instead of re-deriving an unavailable goal.

Dialogue typed handoff audit: use per-proposition claim/value rows for mixed conjunctions and corrections, with actual SpeechUnit focus passed into the NL illocution child. Carry attachment fact IDs from the actual receipt judgment. Keep private intention distinct from communicated retraction; a recipient adoption must join the real issuer ledger through a separately typed owner field. Preserve terminal retraction on later fact/deadline updates; a fresh promise has its own original-act lineage. Recovery uses actual scoped act IDs and pre-belief actor view, never an inferred adopter.

Preserve actual accessible basis assertion IDs when a private note adopts an earlier claim, so a later correction targets that belief through typed lineage. Composition parents can return scoped consistency findings: a declared mistaken private belief is a valid perspective difference, while an inconsistent child attribution is a defect. Report source-explicit ambiguity as preserved uncertainty, separately from missing contribution.

When heterogeneous source units require different output roles, make those roles explicit in a typed NL inventory before invoking specialized children. Identity, timing and capacity context need not invent an ability identity/condition/consequence. Pass actual context meanings to their downstream consumers. Reduce a finite original unit frontier sequentially, retaining accepted readings, semantic uncertainty and execution failures as distinct typed results with original source coverage. A later failed unit must not erase earlier actual child returns or provoke a replay of all completed units.

Give a scoped NL literal a concrete declared result shape before its direct awaited invocation, including semantic fields and exact provenance support. When a source reader sees the whole user request, state its local typed inventory responsibility and name the later stages that derive requested conclusions. Preserve those instructions as source context; inventory completion depends on source inputs, not on computing downstream answers in the reader.

Frame support refinement: a local frame origin can require a fixed global point definition and an explicit identity bridge from another file. Select those original support clauses in scoped NL and pass supportingClauses in the actual convention literal input. Support selection changes only support IDs, preserving every other unit field; complete alternatives retain local/global/bridge evidence and unresolved handedness. Missing origin recovery identifies the original named anchor and required source definition, rather than inventing a numeric point.

When a structured algorithm mixes exact graph mechanics with semantic consequence, pass the actual selected node's typed paths, uncertainty and source evidence to one focused NL judgment. Consume that awaited result directly; use exact arithmetic only for already interpreted signs and exact provenance checks. Preserve incident evidence for nodes with no incoming influence, and keep frontier iteration in its named parent.

When an audit recomputes a graph across folders, select source-role candidates from both the completed typed state and the current batch. Judge each original candidate through its scoped NL child; a previous perturbation must not disappear merely because the current folder contains no new seed. If an exact coverage check reports a missing source ID, revisit the joined typed inputs and perform that candidate judgment while preserving the completed decisions. The check verifies child lineage and source IDs, not which semantic role the child should choose.

For a program declaring actual child-result binding, finish only after the named child decisions have returned and their typed values have been carried into the result. A return rejection asking for declared child decisions is repaired by invoking the missing named stage on its original typed input and carrying its actual result forward; keep semantic judgment in the NL child.
An authored canonical reference can establish source/type validity without executing those named child decisions; it does not grant a live trajectory or training admission. Keep the actual child calls in the model-executed program.

For a large correlated source inventory, separate source-unit discovery from complete semantic tuple interpretation. Use a fixed original-ID frontier and one named NL reading step per tuple; pass the actual convention and referenced fixed-assignment clauses into its scoped child. Retain accepted, unresolved and failed readings separately and preserve every completed result. The parent projects actual returned tuples after exact coverage/provenance checks; posterior selection belongs to its later owner.

For clause readers, classify semantic roles in one typed scoped NL judgment per original unit. Make intervention permissions use the declared probe role. Carry completed accepted/unresolved/failed results through a finite unit reducer; a failed clause does not reset the index. Downstream question interpretation consumes the actual typed index; missing roles remain unknown until the reader repairs those original units.

Declare knowledge projection semantics in the visible contract: whether a knowledge criterion qualifies differs from the underlying proposition value. Keep actor endorsement, actually available corroboration and true/false predicate knowledge as separate typed fields. An authored full certificate is a private source witness; it does not implement source interpretation or replace actual NL stage execution.

Represent source-established deadline/cutoff closure as a typed boundary episode, retaining its original act/rule witnesses without inventing a physical occurrence or participant availability. The NL episode stage owns whether a closure applies; exact clock arithmetic only materializes the returned boundary.

Carry source statement roles explicitly when later graph stages distinguish an imposed perturbation from a relation predicting downstream change. Use a typed scoped seed judgment with original identity, role, signed change and evidence; consume only its actual exogenous-seed result. Retain complete causal judgment reasons and owner IDs through later scope/audit stages. A single-step reducer completes its actual updated state once; its parent owns the next iteration.

A named reducer step that owns one original source item should bind its actual updated typed state and finish that invocation immediately in its first completing return action. Its parent iterateOn supplies the returned state to a fresh next-item invocation. Explicitly assign this ownership when a long-lived child might otherwise revisit its immutable original argument after compaction.

For a finite scan, carry the advancing cursor in state and use `withLimit({ maxSteps: items.length })` or a decreasing `withMeasure` as the structural progress proof. An accumulator may stay unchanged across many items; that alone is not divergence. The default semantic progress judge is reserved for loops without either structural proof; an explicitly configured judge still runs.

For evidence-based reconstruction, give finite source-supported ambiguity a typed basis distinct from missing authority or an unknown attachment premise. Carry the actual target judgment into every alternative and global consistency consumer. Stateful performance or interpretation stages receive the selected source object and actual prior state, then return a scoped semantic delta; exact arithmetic applies that delta. Retain actual prior choices and incoming alternatives in rejected-extension proofs, with an explicit reducer traversal separate from presentation order.

Carry each act/reference's actual per-turn span identity, each nonroot's parent-child link and original physical fact IDs through structured outputs. Returned revision effects identify actual belief/issuer commitment targets; applying those effects is exact state plumbing, while original target and audience-adoption decisions remain NL.

When a source-local name is an exact output identity, carry it separately from inferred physical meaning in the typed inventory. Validate a complete case-preserved token in the original primary span, then require the scoped child to retain that name. Let semantic interpretation judge position, description and counterpart identity. Use one named item step for long source frontiers: store the actual raw typed child and checked projection, preserve earlier results, then finish the shorter state with `eval({code:"return next;",finish:true})`. Parent `iterateOn(initialState).withMeasure(next => next.pendingUnits.length).until(next => next.pendingUnits.length === 0)` owns advancement; aggregation consumes returned values without re-solving them.

For communicated revision tasks, carry adopting-act identity separately from the original revision it adopts and the original promise/propositions that revision targets. A named typed NL target-selection stage should resolve these joins before the parent applies owner-specific effects. Separate actual normative adoption from an issuer's available evidence/endorsement; if receipt without endorsement is tested, establish the report-only posture explicitly in the source.

For a semantic dependency join, expose each required evidence hop in the typed literal result: local premise, identity bridge, defining source clause and independent orientation/alternative constraints. Number the NL steps that select those actual originals. The parent unions all cited groups into child evidence, validates original quotes, and retains the raw returned closure. A known identity without its defining point/value is not a complete dependency; missing premises stay unresolved rather than prompting a crisp fallback.

For an application-level one-item source reader, bind the actual awaited child result into the smaller typed frontier, then finish that already bound frontier in the same action with `eval({code:"return next;",finish:true})`. The parent reducer alone selects the next original unit. If a diagnostic shows the same unit reinterpreted after a valid next state was staged, recover by returning that existing state once, preserving its source witnesses and earlier completed readings.

When a broad inline array judgment blocks after source inventory, move the semantic task into named typed readers over original units. One child judges one selected definition and returns its actual meaning, contextual status or unresolved premise; the parent retains every result and owns the shrinking original-ID frontier. Retain the returned typed indices in the downstream source state, project accepted meanings only from completed readings, and carry unresolved/failed reasons once. Keep domain and stem meanings separate from inventory role labels and later winning hypotheses. Use exact source identity and quote checks as plumbing, then pass actual returned meanings to downstream NL stages.

For a required original-source role reader, name the declared return alias in the direct awaited scoped literal and retain its actual result. Validate its exact original ID, declared role strings and separate exact-quote witnesses before removing the current ID from a measured frontier. If a child result is malformed or transport fails, keep that original item pending, give one concise recovery error and repair that same item; only a source-grounded uncertainty becomes a completed unresolved reading. This prevents an upstream shape error from silently removing a downstream definition.

Separate evidence discovery from interpretation when a scoped literal inherits unrelated helper capabilities or imports another item's policy. A substantive typed discovery callable receives original data only, classifies the selected unit's role/owned primary scope in numbered NL steps, and returns exact cited dependencies. Parent consumes the actual selection rather than automatically promoting all contextual input to ownership. A later typed child interprets those declared owners; intrinsic identity-map checks validate a returned reference-role contract without choosing that role or supplying a missing answer.

For one-item semantic reducers, preserve the actual typed child judgment as the new result slot and complete the updated frontier once. Declare any exact parent projection separately: original owner witnesses may be unioned with child witnesses, while applicability, status and reason remain the actual returned values. Keep earlier settled results unchanged; parent iteration owns the next original item.

For perspective-state programs, resolve adopting acts, original revisions and original target acts/propositions in a substantive typed NL child before applying effects. Retain its actual returned selection in the effect record. A delayed receipt changes current access, not original authorship; process the actual newly available participant/act pairs without replaying the old sender. Current matching corroboration can qualify an already endorsed value under a source-defined knowledge criterion; keep belief acquisition witnesses separately and do not require later evidence to have existed at acquisition. A retracted original remains terminal unless a source-established new promise/reinstatement supplies a distinct transition.

For a retained revision selection, validate its declared kind and original target lineage against the actual selected revision's typed meaning. Generated act IDs may rename, but original supersedes-turn links remain exact. Coalesced physical/receipt episodes can contain new utterances: use original turn membership and occurredAt to establish authorship, rather than requiring an utterance enum label.

When a linguistic hypothesis is tested, separate its actual segmentation/role prediction from the consultant's required judgment. A compatibility child may reject but must not repair that prediction. Scope comparisons to actually prompted/endorsed fields; depicted but unasked facts do not automatically constrain grammar. Retain effective correction versions and original notes separately, and preserve consequential surviving alternatives in held-out translation and gloss repair.

Separate original corpus-role inventory from grammatical interpretation. A one-item typed role reader discovers source scope only and retains its actual result; later typed children own candidate meanings, segmentation and role predictions. Represent an established impossible role assignment with a false licensing verdict and actual conflicting contributions, while missing role evidence retains null; the independent compatibility stage consumes this distinction without repairing the prediction.

Pass actual governing authority rules into note/correction judgments. Preserve a selected original primary separately from referenced context so evidence discovery cannot relabel another correction as the current one. An original correction and its explicit adoption can support one replacement version; group by original replacement identity while retaining both actual support links. Source-unasked prospective fields remain untested grammar differences rather than negative scene evidence.

Keep callable source discovery distinct from directory input data. `read_code(name)` shows named callable definitions and types; the directory reducer's `folder` contains supplied evidence files. Invoke the named typed stage with its declared input, then use its callable children within that active call. If no `.nl` files appear in `folder`, continue with the named callables already shown by `read_code`; that folder listing is evidence, not a callable registry.

When response scope depends on an original prompt, pass the actual prompt clauses into the named child and its scoped literal. A session containing only prompt IDs cannot supply the requested fields or exclusions. Give returned semantic readings an explicit generated identity when later groups reference them; retain actual IDs rather than inventing an undeclared compound encoding.

For a linguistic certificate, retain the actual stem/domain judgments and role contributions with nonempty original quoted support. Keep source-dependent elicitation order distinct from independent hypothesis order: a later contradiction cannot justify skipping its prerequisites retrospectively. Original correction IDs and adopted note versions remain separate from generated hypothesis/reading IDs. An exact certificate can validate declared fields and source joins; linguistic meanings and replacement wording still need semantic judgment. If feedback reports a missing original join, restore the actual returned child record and its original evidence rather than rewriting a preferred interpretation.

When a named reader returns per-item semantic entries, retain its complete typed index in the parent source alongside the accepted projection. Append actual missing-premise and transport/type failure reasons through frontier, index and parent summary. Project only accepted entries of the declared meaning kind; context/unresolved/failed entries remain represented without donating an accepted meaning. If exact feedback reports a projection mismatch, return the actual child meaning and original identity unchanged rather than reconstructing a replacement table.

Native positional-state artifacts are the existing typed `Neuralese<T>` values.
Pass the returned value directly to a child or parent; keep its content address
through captures and typed results. `neuralese-native-consumer-unavailable` means
the target must explicitly declare state adaptation or text reencoding before
consuming another model's native value. Preserve the source artifact and use a
configured native reader; do not paste its text to bypass the typed boundary.

Native computed Neuralese strings, including `""`, use the configured native
write port. Native `empty()` comes from a loaded native standard library's
model-bound zero-token state. For `neuralese-native-empty-unavailable`, rebuild
or load that library with its archived empty state. Contextual native body
initialization preserves its actual source capture and does not claim equality
with an ordinary-text prompt.

When hosting a server-backed native runtime, bind its real writer and reader
identity with `nativeStateRuntimeOptions({ endpoint, store, headers? })`; a store
alone cannot produce model-native state. If setup reports
`neuralese-native-text-write-unavailable`, use a server that exposes the forced
native text writer. Do not replace it with hash embeddings or projected token
values.

A configured learned semantic adapter may consume another trained model's
native value by decoding shared per-token codes into target-state deltas.
Keep the value reference and its actual consumption receipt. On
`neuralese-semantic-adapter-mismatch`, rewrite the payload with the target
model's native writer or restore the matching checkpoint and adapter export.

After shared adaptation, native writes may contain a compact semantic message
with text, source embeddings and learned codes. Keep its typed reference through
storage and calls. Its codes require the matching adapter before KV splicing;
`neuralese-semantic-adapter-mismatch` has the same recovery described above.

A learned compact value uses `semantic-state/<adapter SHA>` as its dialect.
Keep its reference intact. DefaultDialect may use a consumer's explicitly
advertised native adaptation/reencoding; a named semantic dialect still requires
that exact adapter. A consumer-unavailable error requires configuring the
matching adapter or an explicitly supported text reencoding path.

Native graph learning targets explicitly named model parameters or consumed raw
KV/hybrid-state tensors. Native optimizer updates can publish backbone and
interface changes as a new checkpoint; reload that checkpoint and restore its
archived optimizer state before continuing. A prior semantic adapter needs
refresh against changed weights. `neuralese-native-grad-order` means the service
supports first order only; remove derived computations or choose a service
advertising `grad.order2`. `neuralese-adapters-unavailable` means load the applied,
identity-bound checkpoint before requesting native graph gradients.
If a native gradient/update binding error occurs, restore the artifacts for the
exact current checkpoint and repeat the request; never relabel tensors or reuse
an adapter against changed weights.
Portable native checkpoints preserve manifest identities and all parameter,
optimizer and training-state dependencies. If bundle verification reports missing,
changed or escaped artifacts, restore the complete verified bundle and the pinned
base model files; do not rewrite identity metadata or substitute another adapter.

Native guided chat now uses the same call/line guidance checks as ordinary chat.
Guided streaming publishes only the final committed reply, since a rejected
prefix cannot be retracted from a client. A
`neuralese-guidance-template-conflict` error means the requested forced template
violates guidance; choose a compatible call template or remove that forced
constraint. Keep the actual sparse `banned_token_ids` in each accepted sampled
action when retaining native RL receipts; forced template inputs are not actions.

A native training `foundation_ready` receipt qualifies the complete declared
communication cohort for cross-model work. `completed` still means all declared
supervised rows and full final held evaluation finished. Preserve `sft_progress`
credits when continuing through semantic/RL stages, and consume its remaining
obligations rather than starting row counts again. A semantic-refresh boundary
is an incomplete continuation with actual optimizer updates and current checks.

An adaptive shared `refresh_ready` publication has executed every declared phase
quantum and actual current task/runtime/compact-wire checks. It is not full
curriculum completion or evidence of Neuralese lift. Preserve `semantic_obligations`,
per-pair exposure, held cursor, optimizers and RNG while continuing. A missing
phase-work, finite-held or compact learned-transport proof requires completing
that work against the exact current adapter, rather than changing receipt flags.
A native production foundation-source error requires restoring its declared
`foundation_native_graphs` dependency, rather than substituting a smaller shard.
A changed semantic objective-kind error requires a new declared phase ID so its
new obligations do not inherit earlier objective credits; keep the earlier phase
and its actual exposure while adding the new work.
If changed participant weights require adaptive refresh, declare positive
`adaptive_refresh.updates_per_phase` for actual VAE/joint/distill work before
merging; old exposure remains credited but never licenses a stale adapter.

A shared semantic controller without RL bootstrap automatically continues partial
publications until its semantic quotas finish. Retain its `last_semantic_publication`,
child launch record and per-cycle outputs when restarting. A continuation-parent
or wrong-cycle checkpoint error requires restoring the exact bound publication
and matching child checkpoint; do not substitute a different model or clear row
credits. A command-placeholder error requires `{participants}` and `{out}` in the
shared-stage command so each new cycle keeps its successful parent intact.
A parent change during an active cycle requires checkpointing and resuming with
the new declared parent. An exposure-counter underflow requires investigating
row quotas against the last valid checkpoint; never clear credits to bypass it.

Native pair conversion authenticates original provider/capture context and
selected targets before static role-span derivation. Missing earlier writer
evidence requires the complete original trajectory as context-only writer
input; it must not create extra targets. A source/value/writer binding error
requires the exact approved original record, typed block/name and payload hash.
Never edit old runtime receipts to match normalized markers, promote an
unapproved endpoint, or use a preview renderer to bypass those checks. Derived
receipts distinguish original source evidence from changed rendered bytes.


Existing reasoning boundaries keep their tokenizer IDs and spelling. Native
models may learn sparse input/readout overrides for `<think>` and `</think>`;
these are separate from the four Neuralese role markers. Old parents retain
exact behavior until the declared `--learn-reasoning-boundaries` upgrade. A
sparse-token checkpoint mismatch requires the exact checkpoint's token-ID
contract; do not rename markers or replace its tokenizer. An optimizer schema
mismatch requires the explicit sparse-row migration: added reasoning rows and
exact BF16/FP16-to-FP32 promotion of existing sparse masters are the only
permitted schema changes. Preserve moment values/RNG/exposure; never reset the
optimizer to bypass it. Sparse row shape/dtype errors require the exact saved
row contract, not broadcasting or silent casts. Reload actors from
the new actual checkpoint and rebind/requalify semantic communication after
updates. No new instruction text or reasoning-format workaround is needed.

Native functional adapters use exact-base named tensor coefficients and frozen
SVD bases. On a native service, `adapters.create({parameterNames, rank, kind})`
selects actual named layer matrices; retain the returned `Adapter` and bind it
with `withAdapters`. Native first-order graph learning uses
`nativeGraphGradient` with `target: {kind: 'adapter', id}` and the same active
adapter IDs, followed by `nativeAdapterOptimizerStep`; this does not enable the
legacy recorded-turn GradSession API or higher-order objectives. Keep the
coefficient, basis and optimizer artifacts together when saving or restoring.
`neuralese-native-adapter-invalid` requires restoring the exact coefficient and
basis artifacts for the current checkpoint and selecting its advertised matrix
names. `neuralese-native-tensor-as-state` requires binding adapters with
`withAdapters`, rather than passing named tensor artifacts as communication
state. Native adapter policy learning retains sampling/3 receipts and a durable
actor bundle containing coefficients and frozen bases. Bind the exact same
ordered IDs/scales when evaluating its admitted native-policy objective with
`nativeGraphGradient`. `neuralese-native-actor-changed` means finish or cancel
the current invocation before changing its adapter binding. Mixed adapter
bindings across one replay graph require explicit per-invocation declarations
and a versioned immutable behavior policy. The typed native learning service accepts `invocationAdapters` with a
`native-policy-objective/2` binding admitted from actual sampling receipts. Keep
each invocation's ordered scales and frozen bases; current coefficient IDs may
advance through `nativeAdapterOptimizerStep`. On
`neuralese-native-grad-invalid`, restore its behavior bundle and match the
admitted graph, invocation inventory and basis/scales. External observations
require the original consumer occurrence and explicit `same-base-adapter-state`
when actors differ. Production full-backbone stage admission still rejects
adapted actors until its separate coefficient-policy stage is implemented.

Native coefficient learning may explicitly select
`adapterGradientPrecision: 'fp32'` in `nativeGraphGradient`. This retains the
actor's original forward storage values and carries coefficient cotangents in
FP32 through selected native Linear and dense-expert matrices. The default is
`storage`; changing this setting does not replace coefficients, bases or
optimizer moments. For an invalid precision error, select one of those two
advertised values. It does not enable heterogeneous adapter-policy graphs or
higher-order objectives.

An admitted native adapter-policy objective retains its original behavior
coefficients, sampled log-probabilities and observations. You may continue
coefficient updates against that objective with current coefficients on the
same exact base and ordered scales/frozen bases. Pass the current coefficient
IDs and exact-parent optimizer state; retain the original behavior bundle.
For a behavior scope/basis mismatch, restore those bases/scales or collect a
new admitted objective for the deliberately changed scope. Do not rewrite old
sampling receipts to the current actor identity.

Under default strict transport, changing coefficients while reusing genuine
external behavior KV requires a fresh rollout. Internal producer state is
replayed under current coefficients; external KV cannot be silently relabelled.
An explicitly supported/requested same-base adapter-state mode is required to
retain external KV across actor changes.

Native adapter-state exchange: an explicit `same-base-adapter-state` declaration
may import raw state from another adapter actor only on the exact same base,
tokenizer, embedding and architecture. For `neuralese-native-state-mode`, restore
the source coefficient/basis artifacts and correct the exact input-ID declaration;
use learned semantic adaptation for different bases. Use this mode only when the server advertises it in native-state consumption
modes; matching tensor dimensions alone never establishes support.

An explicit `same-base-adapter-state` contextual import keeps distinct source
and target actor receipts. Preserve `native_state_modes` when recording or
replaying the call. For `neuralese-unknown-native-artifact` during restoration,
restore the source actor's exact coefficient artifacts and frozen bases alongside
its raw state; restoring only the state cannot authenticate the producing actor.

Native graph coefficient learning may declare `invocationAdapters` by exact
invocation name. Continue from the returned `invocation_actors` receipt when
updating several call actors; a heterogeneous graph has no single effective
model identity. `neuralese-native-grad-invalid` for an invocation declaration
requires actual graph node names and exact-base coefficient/basis artifacts.
Keep protected test rows out of gradient requests. Heterogeneous policy-objective
admission/replay remains unavailable until its per-turn behavior contract is
implemented and qualified; graph CE support does not imply that RL support.

A declared native coefficient production policy binds generation and decisions
to the same current actor. `neuralese-native-coefficient-policy` means restore
the exact declared policy and coefficient archive; retry with that binding.
Stable adapter handles are mapped by the declared policy, so retain those handles
in the program rather than substituting an old coefficient artifact yourself.

A produced Neuralese value remains an opaque typed reference. For a string-slot
`type-mismatch`, return actual answer text, or pass the reference to a declared
`Neuralese<T>` reader that returns the requested type. Do not return an artifact
ID or internal marker as answer text. Literal string contents remain literal.

Native services advertise direct `xs`/`tiny` adapters after integrated service
qualification. Supply explicit `parameterNames` selecting actual native layer
matrices and retain their exact coefficient/basis/optimizer archives. On an
invalid native adapter request, correct those names or restore the matching
checkpoint closure. Native `grad_order: 1` remains first order; advertised
adapters and same-base state exchange do not imply higher-order gradients,
legacy projection decoders or cross-base raw KV compatibility.


A native imported-state gradient can be applied with the typed host operation
`nativeStateOptimizerStep`. Its result is new KV/hybrid state, with the original
producer preserved as origin and exact gradient, update and optimizer-history
artifacts retained. Keep its full `.nz` dependency closure when saving or moving
it. For `neuralese-native-optim-invalid`, restore the exact parent state and
matching named optimizer history; explicitly omit history only for an intended
new optimizer. A compact inference view cannot construct optimizer updates;
restore its complete source parent before requesting an update.
Do not relabel an optimized state as a fresh producer capture or
replace missing state with decoded text. Public automatic `grad(...,{order:2})`
remains unavailable until its native recorded-turn/DAG path is qualified.

Optimized KV/hybrid states bind their exact scalar optimizer arithmetic runtime.
For a native optimizer arithmetic/runtime mismatch, restore the state with its
bound CPU Torch/runtime; do not relabel it for another executor or discard its
history. Archive byte integrity is distinct from numeric optimizer verification.


Native recorded learning needs the actual server input/write receipt and a durable
archive. For `neuralese-native-grad-invalid` reporting a missing receipt or
conflicting turn occurrence, keep each recorded call with its own execution
graph and use the matching native server; do not reconstruct producer edges
from equal text or artifact IDs. For `neuralese-native-archive-unavailable`,
configure a store supporting native artifacts before recording communication.
Greedy structural receipts have no behavior probability and cannot be used as
on-policy RL samples. This capture mechanism does not advertise public order-2
GradSession support.
Assistant-only producer blocks follow the same typed result checks as tool returns.
For an ordinary string result rejected with `type-mismatch`, read the imported
Neuralese through a typed natlang call and return its actual answer text; an
internal artifact ID is not the answer. Explicit Neuralese returns retain the
original block. Literal text that merely spells a marker remains literal.

When a server reports an actual native-state content part as the assistant's
final completion, the declared return type determines its meaning. A compatible
`Neuralese<T>` result keeps the typed state reference. An ordinary string or
JSON-compatible result reads the state's authenticated stored text and passes
that value through the usual exact result check. Mixed text and state parts keep
their reported order. This applies to inferred assistant completions; an
explicit `return_result` call still follows its existing value validation.

For a decision-objective `neuralese-grad-term` or native objective validation
error, provide finite numeric rewards or nonnegative finite probabilities with
positive total mass, exactly one entry per declared option. Keep the recorded
conditional token queries bound to their actual graph and avoid assigning a
likelihood to a forced input after an imported state.

For an explicit native query readout binding error, restore the complete immutable
readout snapshot for the matching base and layout, including final norm, tied
vocabulary table and sparse rows. A teacher hidden state alone does not select
a different teacher head; stop the declared teacher distribution. This does
not enable arbitrary heterogeneous full-backbone parameter overlays.

For a native derivative objective-origin error, restore the exact declared
program, graph and bound values. Older internal program gradients without an
exact graph receipt need a genuine reexecution; do not synthesize or relabel
provenance. This stronger program check does not invalidate ordinary prior
first-order state/gradient history.

Recorded native decisions need the exact rendered option receipt and native
input archive. For a missing decision receipt or archive error, use the matching
native scoring service and restore its original/decoded state dependencies;
never substitute a sampled text reply or infer producer links from equal IDs.

For a serving error `System message must be at the beginning`, preserve the
ordered system/recovery messages and use a qualified compatible server template.
Do not move recovery instructions into the initial prompt or change their roles.
The pinned Qwen compatibility renderer preserves originally valid renderings;
its changed recovery path needs live executor qualification before adoption.

For a whole-episode native gradient error, restore all original model turns and
their execution-graph write/read occurrences before re-admission. Correct plain
execution remains rewardable; do not add Neuralese traffic to pass a gate or
replace an omitted in-episode producer with detached archived state. Decoded
imports need their actual differentiable lowering, not relabeled provenance.

`stopGradient(value)` preserves native typed values and soft-function bodies as
constants for that use, including after context snapshots and `.nz` save/load.
An equal live value remains differentiable. Keep the typed value when passing
it to another call or `withAdapters`; do not extract its artifact ID to preserve
a constant boundary. No payload is decoded or rewritten by this operation.

Native Neuralese values retain their source-call occurrence through returns,
context snapshots and `.nz`; equal artifact IDs may come from different calls.
Pass the original typed value rather than rebuilding a reference from its ID.
If native graph verification rejects ambiguous origins, restore the complete
recorded episode and its typed source references; do not substitute a detached
external state for an in-episode producer.

Native marker failure recovery: when the native executor reports
`neuralese-native-state-open`, `neuralese-native-state-close`,
`neuralese-native-state-consume-marker` or `neuralese-native-state-nested`,
that sampled reply executes no tools and supplies no return value. Repair the
same call with a complete native value or its declared ordinary result. Any
completed writes from the failed reply retain their actual provenance; an
unfinished span is never a communication block. Do not copy an opaque state ID
into an ordinary string result. The runtime retains the failed draw for graph
learning instead of silently retrying the model transport.

Native state IDs are complete content IDs. A shortened ID copied into a tool
argument stays ordinary data for the declared result type; it cannot identify
a cached state. Recover from the normal type error by returning the required
ordinary value or by using the complete original state reference. Never guess
a full ID from a prefix. Actual complete references keep artifact restoration
and provenance checks.

Recorded callable Neuralese returns retain the sampled instruction body and
their typed runtime reference. If conversion reports an unlinked callable
return, restore the complete original raw tool call, invocation output and
block-write trace; do not replace the sampled body with an artifact ID or
invent a successful outcome. The converter authenticates the exact declared
callable type, body hash and producing turn before linking the return.

History reasoning follows the checkpoint's bound rendering profile. An
explicit `history_reasoning_policy: keep` makes earlier assistant reasoning
visible in complete conversation history; it does not change the original
source targets or authorize rewriting old graph receipts. If a saved checkpoint
or tokenizer reports a conflicting history policy, restore the matching
checkpoint/tokenizer binding rather than forcing a different template switch.
The opt-in profile remains unmeasured until its live executor comparison is
recorded.

TypeScript ambient callable declarations of the built-in `nl` supply types and are erased; they do not replace its runtime tag. Real runtime bindings named `nl` still hide the built-in. If that shadow diagnostic appears, preserve the built-in and rename the runtime binding; do not create a replacement implementation.

A scripted source oracle must answer a natural-language child with a value of that child's actual declared result type. Missing coverage is a harness repair, not evidence that the authored program is semantically wrong. Preserve the source task and independently execute all declared input cases; do not rewrite expected results or convert a string decision into a boolean to satisfy the checker.

For a function declared to return `Neuralese<string>`, a final assistant text reply is the exact text value. The runtime writes it through the configured Neuralese writer and returns the resulting typed reference. Intermediate text that leaves the invocation without a result is incomplete; finish the same invocation with a typed value instead of treating that text as success.


A checked inline callable with an explicit or contextual parameter signature
keeps that boundary when used as a JavaScript callback. `array.map(judge)`
passes only the parameters the callable declares; declaring an index or array
parameter preserves it. A saved open, untyped callable still receives every
argument supplied. To recover from an unexpected missing callback value, put
that value in the callable's declared signature rather than relying on an
undeclared array-method extra. Recompile source with the matching runtime;
do not relabel older compiled plans or saved actor provenance.

### Terminal policy children

When all facts and the decision rule are supplied to a typed child, compute its declared scalar result and return it directly. Apply explicit exceptions before ordinary inclusion conditions; keep parent aggregation and service/readback recovery separate. A delegation error calls for inspecting the supplied facts and completing the current decision, rather than delegating the same question again. The independently constructed terminal-policy recovery teacher packet exercises this pattern; its teacher measurement and training exposure are recorded separately.


For an outside-world paged service, declare the valid page range in its interface
and reject invalid indices with the exact range in one sentence. Recover from
that feedback by choosing an in-range call; a failed call does not establish that
the source has no records. Preserve reserved service bindings and return the
declared result type after completing the original task. The fresh TRAIN
`tool-contract-recovery` profile measures this behavior through the existing
self-hosted teacher; it does not change ordinary runtime or compiler semantics.

A completed reply cut off before any executable action consumes an explicitly configured repair budget. Recover with the existing instruction to take the next step with one tool call; return the actual typed result when ready. `model_incomplete: truncated_without_action` records exhausted repairs, separately from transport errors. Successful executed actions/results can establish progress; an emitted rejected proposal cannot. One still-pending long decode is not a failed continuation.
