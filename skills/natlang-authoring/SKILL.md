---
name: natlang-authoring
description: Author, refactor, review, evaluate, and adapt natlang code — inline `nl` calls in TypeScript, named `.nl` functions with their callable folders, and `natlang.d/` application context. Use for semantic algorithms, reducers, application logic, or measured instruction and program-guidance optimization, including programs intended for small interpreter models.
---

# Author natlang code

You are the capable program author; the interpreter may be a small model. Put the algorithm, decomposition, data contracts, and recovery decisions in source so execution does not depend on the interpreter inventing an architecture.

Natlang is TypeScript with natural-language functions. Ordinary TypeScript owns exact computation, I/O, and orchestration. A natural-language function owns a judgment: interpretation, planning, semantic merging, choosing among options, explaining. Choose the boundary deliberately. A host that computes everything and asks natlang to approve wastes it; making the model do arithmetic by hand wastes its context.

For a student served with `program-header` rendering, native natlang calls omit their generic tool schemas because that tool surface was trained into the student. A separate application calling the student directly should still supply schemas for its own tools; serving retains those schemas in this mode.

## Establish the contract

Locate the natlang checkout or installed `@natlang/node` / `@natlang/browser` version first. Use its declarations (`dist/index.d.ts`), tests, and `natlang check` as the authority; if prose here disagrees with executable behavior, flag the disagreement rather than inventing behavior. Outside a checkout, rely on these bundled references and the installed package.

Read [language and source contracts](references/language.md) before creating or restructuring code, [algorithm patterns](references/patterns.md) for loops, reducers, reconciliation, and generated methods, and [verification and diagnosis](references/verification.md) when testing or debugging. The [review example](assets/review/review.nl) shows a named function whose callable folder holds a semantic helper and an exact aggregation; copy the whole `assets/review/` directory.

Put a property of a value in its type rather than in a guard sentence: read [constraints belong in types](references/refinements.md) for `Is<T, "predicate">`, when to use it and how to phrase the predicate.

To let the runtime keep an intermediate value as Neuralese between stages, write the chain so its data flow is explicit: read [orchestrators that can fuse](references/fusion.md).

For evaluation suites, instruction optimization, or trainable program guidance, read [adaptation and evaluation](references/adaptation.md). It covers component selection, independent scoring, fresh fixtures, bounded search, report review, and artifact activation. Ordinary authoring does not require an optimization run.

## Build a real program

1. State the behavior as typed inputs, results, and observable scenarios, including ambiguity, empty inputs, and partial failure. Types check structure, not meaning.
2. Write the orchestration in TypeScript. Call natural language where judgment is needed: inline with `` nl`…` `` for a one-off decision, or a named `.nl` function when the instruction deserves its own file, its own helpers, or reuse.
3. Give each natural-language function one coherent responsibility, a precise signature, and the helpers it may call in its callable folder (`foo/` beside `foo.nl`, or `natlang.d/` for inline calls in application code). Put only the program's own code there; the systems it acts on (stores, boards, simulators, verifiers, APIs) are services with declarations, scoped to the functions that should use them.
   When one named function must call another during its own active call, place the second `.nl` inside the first function's callable folder. For a chain `root -> inspect -> decide`, use `root/inspect.nl` and `root/inspect/decide.nl`; two sibling files under `root/` do not give `inspect` access to `decide`.
4. Write each instruction so a small interpreter can take the next step: name the data and how to reach it, state the step a loop repeats, and keep the words for finishing statuses ("missing", "cannot") for real blockers. See [patterns](references/patterns.md).
5. Keep exact work exact: helpers in callable folders are ordinary TypeScript under a finite-iteration policy; application code outside them is unrestricted.
6. Run `natlang check` (types, `nl` signatures, callable-folder policy), then exercise the real source through the runtime. Distinguish checks, scripted wiring, live-model runs, and semantic evaluation.

## Improve instructions with measured adaptation

Use adaptation when the callable contracts and decomposition are sound and the task is to improve instructions or program guidance against explicit scenarios. Fix missing capabilities, types, capture names, or helper structure in source first; adaptation changes static instruction text, not those contracts. Label authored inline sites when stable selection matters, keep expected answers outside model-visible code, and use fresh fixtures with independent scoring. Review held-out gates, helper coverage, actual usage and uncertainty before adopting an artifact. A retained baseline is a valid result.

Keep search overlays separate from authored source. Load a compatible artifact explicitly, use `adaptation: null` for a baseline task, and revalidate after source or model changes. Read the [adaptation reference](references/adaptation.md) for SDK and CLI examples; use `natlang-integration` when wiring an application host or browser deployment.

## Preserve the execution model

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

## Recover from service argument-shape errors

A service declaration defines its positional arguments. When a service error gives the expected typed call, inspect that method with `read_code`, bind each declared argument separately, and retry the stated positional invocation. For the story witness service, use `storyEvidence.checkWitnesses(snapshot, witnesses)` with a `SourceSnapshot` and a `Witness[]`. Keep this recovery in the boundary error and skill guidance; the application prompt need not enumerate invalid call forms. Natural-language stages still decide what the witnesses mean; the service verifies their exact source identity and quotation.

## Carry exact provenance through semantic calls

Pass the original typed source record to a named evidence interpreter. Give context/declaration parsing and event interpretation separate typed responsibilities when they produce different data. A scoped NL judgment may return meaning arrays and exact quote bytes; its parent binds witness source ID and path from the original record, then checks quotation containment before consuming those findings. If exact witness checks fail, retain an unresolved evidence finding and inspect the original record rather than inventing provenance. Context canon remains input to authority interpretation even when it is excluded from an event frontier.

A nonempty source-grounded semantic contribution requires at least one nonempty supporting quote whose exact source check passes. An empty witness array cannot certify that contribution; retain the missing support as unresolved.

## Separate symbolic readings from operational bindings

When one semantic stage reads source terminology and its parent materializes an executable plan, give the child a typed symbolic reading and the parent a distinct typed binding contribution. A parent's scoped NL judgment joins that reading with visible code definitions and compares displayed wording with counted meaning. Preserve disagreement as a diagnosis without silently repairing an implementation. The executable uses the returned binding; comparison consumes the symbolic reading and meaning relation. Missing or conflicting definitions remain unresolved rather than selecting a code by label.

## Compile evolving prose as a semantic application

Separate source edition/authority selection, rule/exception interpretation, carried state inference, callable/literal lowering, NL module writing and scenario execution into typed transformations. Preserve historical event-time clauses alongside later replacements, and retain duties outside a scoped waiver. Generated guards return actual typed state effects consumed by scenario replay; a correct source model cannot certify an emitted body that dropped those rules. Diagnose earliest divergence and revise generated policy or its missing state handoff, keeping original authority/evidence unchanged. Missing authority or decisive evidence remains unresolved rather than becoming a fabricated default.

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
