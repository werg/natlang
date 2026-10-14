---
name: natlang-integration
description: Embed natlang in Node/TypeScript and browser applications. Use when wiring the runtime, model drivers, host services, compiled `nl` calls, event loops, terminal or browser interfaces, directory reducers, packages, persistence, traces, call records and compilations, or application evaluations.
---

# Integrate natlang applications

Build an application in which natlang makes the judgments and ordinary TypeScript does the rest: storage, transport, exact operations, and presentation. Natural-language functions are ordinary async functions in the application; there is no separate program runner.

Start by locating the installed `@natlang/node` / `@natlang/browser` version or the checkout and any existing embedding. Reuse its runtime and model lifecycle. Repository paths in the references are lookup hints relative to a natlang checkout.

## Choose the smallest adequate boundary

Read [hosts and model drivers](references/hosts.md) for the runtime, compilation, services, and model transport; [frontend applications](references/frontend.md) for browser models, event loops, and generated interfaces; [terminal applications](references/terminal.md) for CLIs, packages, sessions, and event streams; [effects and recovery](references/recovery.md) for native objects, concurrency, and durability; [call records and compilations](references/records.md) for the machine's record of calls, opting values out of it, and compiled cases. Read [refinements](references/refinements.md) for `Is<T, P>` settings, crisp checkers, the `refinement-*` error codes and their recovery. Read [fused pipelines](references/fusion.md) for the `fusion` setting, the qualification certificate and the `fusion_*` trace events. Read [delivery scenarios](references/delivery.md) before claiming an integration complete.

1. Decide who owns state, native objects, and effects. Host capabilities become typed services passed to the runtime; natlang code reaches them only through `natlang:services` or eval bindings.
2. Write the application in TypeScript and compile it with `natlang build` (or `buildProject` / `compileVirtualProject`) so `nl` calls are planned and typed. Named `.nl` functions import like modules.
3. Create one runtime per application (`createNatlangRuntime({ model, services })`) and run natlang work inside `runtime.run(...)`. Callbacks from uncompiled code (DOM events, timers, libraries) use `runtime.bind(fn)`.
4. Define event ordering and commit points with ordinary code or `EventLoop` (`KeyedEventLoop` for one loop per user or document). Persist operation identities and observations for effects that must not repeat.
5. Test the actual engine and transport, success and partial failure. Make general fixes in the shared runtime, not as app-local workarounds.

## Operational commitments

- Omit arbitrary run budgets unless deployment requires them; `limits` and model options exist for that. A call longer than the model's context (`contextTokens`) is compacted: the model writes a note, and its full history stays searchable from eval as `transcript`.
- Set budgets (`maxTurns`, `maxTokens`, `maxSeconds`, `maxFailureRepairs`) explicitly for evaluations; none apply by default.
- Services and live values are trusted native authority passed by reference. Eval is trusted code in the application's process, not a sandbox.
- Service calls and live-object writes happen immediately and are traced as effects; they are not rolled back when a call fails. Captured `let` variables are written back only after a successful eval.
- Every call is recorded in the machine's call store (exact values within bounds) and may be served by a compiled crisp case; a case that fails hands the call back to the function. Exclude sensitive values with `"recording": { "exclude": [...] }` in `natlang.json`, and give embedded runtimes `programRoot`. See [records](references/records.md).
- Seeds and pinned sources support reproducibility; they do not make native state replayable or decisions identical across inference backends.
- For a folder reducer, keep item-file and task-file contracts distinct: preserve each item's options and use shared options, criteria, or levels only as declared fallbacks. Bind values from those files before constructing per-item inline calls, keep explicit empty lists empty, and use declared options as the answer domain when no criteria are supplied.
- Native input and scope listings show at least 4,096 characters per value. A cut-off listing names the complete live eval variable; use ordinary code such as `value.slice(start, end)` to inspect omitted ranges. Tool-result text is sized from the call's context budget (`contextTokens`; `shownCharsFor` in `ts-host/src/native/cutoff.ts`: an eighth of the budget at 3.5 characters per token, at least 2,000), and only output longer than that is paged with `read_page`. A host that sets a larger `contextTokens` therefore shows longer outputs whole; `contextTokens: null` keeps the 2,000-character floor.
- Eval exposes `inputs` and `read_inputs()` only for calls with formal input fields, unless those names are shadowed. A zero-argument call has no implicit caller-input object; ordinary local declarations may use those names.
- Verify live-model quality separately from fixture wiring and rendering, and name the remaining empirical gates.
- For source-grounded graph outputs, make any exact entity-key normalization visible to the natlang stages and keep the original evidence text intact. A source/type replay does not prove that live stages joined cross-file identities correctly.
- For staged semantic extraction, pass raw source text to the owning first stage and pass its returned typed assertions, source IDs and exact witness snippets to later named stages. Each later stage owns a new semantic decision and hands its actual returned value forward; an unresolved decision remains typed. The host may check ID coverage and quote substrings, but it does not construct domain edges or repair missing semantic findings.
- Keep claim extraction, relation interpretation and causal-support assessment as separate typed interfaces when each can disagree. A relation decision cites its primary assertion's own witness, and a focused causal assessment consumes one established edge plus the needed typed premises. The parent binds original IDs and checks exact witness membership; it passes the actual returned decision into the next stage. A failed or absent assessment remains explicit for the audit.
- When a directory reducer processes related folders, pass its previous typed result into the next call and merge only new stage-owned contributions. Check live traces for a final stage that re-derives past semantic findings from raw files; that indicates a missing carried input contract.
- In native collection, an absent result after two consecutive provider-confirmed immediate empty EOS replies is reported as `model_incomplete: terminated_without_result` after one normal recovery prompt. Keep this distinct from transport failures and resource timeouts; if a provider adapter does not expose exact EOS metadata, it must not claim this classification from an empty string alone.

Pair with `natlang-authoring` for substantial natural-language algorithms when that skill is available; this skill is independently usable.

Persistent eval helpers support TypeScript `as const` assertions for literal and tuple results; the assertion needs no captured type binding. Other referenced named types must remain available in the call scope.

## Student prompt rendering

Use `promptRendering: "program-header"` in the model profile (or runtime `model`
settings) for students trained beyond the full-prompt warm-up. Match the checkpoint's
`prompt_rendering` declaration. Other models default to `"full"`.
Program-header requests retain the signature, types, instructions, inputs, skills,
call-specific declarations and trajectory. Native natlang calls omit generic interpreter
prose and native tool schemas; the same native tool calls still execute. When another
client calls the student directly with its own tool schemas, serving retains those
schemas so the student can discover and call them. Custom application instructions
remain. Training and serving use the scope table exported by
`node ts-host/scripts/neuralese-prompt-pieces.mjs` after building ts-host.

## Validate exact service arguments at the boundary

Export trusted service methods with explicit named parameter and return types so their declarations show the callable shape. Validate required argument structure before dereferencing fields, and reject invalid input with one sentence stating the correct positional call and its declared types. For example, the story witness boundary teaches `storyEvidence.checkWitnesses(snapshot, witnesses)` with `SourceSnapshot` and `Witness[]`; it then performs exact quote checks. Preserve semantic judgment in NL stages. Record changed declarations or error text as model-facing debt and sample the corrected boundary live before adoption; a recovered call or a source/type replay alone does not qualify its presentation.

## Preserve source-record authority at semantic boundaries

Package and pass the complete original typed source record, including its ID, path, text and hash, through evidence call boundaries. Bind witness identity structurally in the parent from that record, while the NL child supplies interpretation and exact quote bytes. Use the existing exact witness service early; a rejected quotation yields an unresolved evidence finding before it can support downstream decisions. Distinguish context declarations from scene/event input frontiers without discarding controlling canon. Pin source revisions separately from a running diagnostic, and measure the revised signatures/instructions live before claiming the provenance repair is adopted.

For wide evidence inputs, qualify a relevance-first callable as a typed semantic stage before using it to reduce deeper calls. It can receive the complete inventory for context plus one smaller focus group, returning roles only for that group; the host joins the groups into a ledger covering every original source ID/path. Include uncertain and dependency records, and allow later findings to reopen background records. The exact host validates coverage and quote spans, while the callable judges relevance and authority. Compare live completeness, source-supported outcomes, and request cost against the frozen per-record source before adopting its model-facing wording.

For a scoped relation literal inside a named stage, carry the original complete source bundle through the named stages. Give the literal the current relation and exact supporting evidence; its result is a typed relation decision returned to the parent. The parent merges that decision into the typed state before calling the next named stage. Review live traces for dropped evidence or a child inventing a replacement bundle; these are source handoff defects until localized.

When an authored oracle relies on a derived output identity or serialization convention, render that convention in the model-visible root instructions and verify its exact presence in the frozen source. Source/reference replay can pass even when this wording is absent, so check visibility separately before interpreting live rejections. Treat an absent convention as a source bug, not a model failure.

A nonempty source-grounded semantic contribution requires at least one nonempty supporting quote whose exact source check passes. An empty witness array cannot certify that contribution; retain the missing support as unresolved.

## Observe semantic materialization at both stages

Keep child symbolic interpretation separate from parent operational binding in report types and traces. Exact services receive the parent's resolved code or plan fields, while semantic comparison receives the retained reading and any display-meaning relation. An exact checker can verify owner/clause identities, code/arithmetic consistency and declared invariant fields; it cannot establish source entailment or actual scoped-literal execution. A new typed witness is a canonical form, and its numeric equivalence to an older witness does not qualify revised model-facing stages live.

## Keep generated policy distinct from exact compiler services

Use existing parser/type/callable-folder mechanisms for generated NL source, immutable source identity and exact file serialization. Semantic edition/rule selection, exception composition, lowering boundaries and guard effects stay in NL. Package the actual generated module bytes and their carried state/guard traces separately from the original governed model. A typed generated-folder witness or static contribution audit is not live execution proof; independent scenario entailment and literal-to-state consumption remain separate gates. Generated-source revisions are private proposals with explicit unresolved failures.

A generated-program audit must preserve actual emitted decisions/effects and independent source-required decisions/effects in separate typed channels. Exact source quotation checks use original record IDs; generated-code evidence uses its generated path and exact span. A symbolic interpreter can produce useful semantic certificates, but must not be labeled as generated-runtime execution, and cannot silently correct the candidate before comparing it to source obligations.

## Inspect conditional result projection

Keep candidate-pair identity attached to actual semantic child returns in a typed parent-owned table. Exact checks can verify quotation provenance and that projected IDs/positions match those rows; NL functions decide conditional compatibility and whether a place lacks an independently named counterpart. Review live parent outputs as well as child correctness: a true conditional judgment can still be dropped during projection. Repair the source handoff in a separately pinned variant, preserve the frozen diagnostic, and do not credit canonical source replay as live contribution use.

When reviewing generated/source pseudocode, check that evidence and comparison obligations precede the whole-function return. Validate nonempty generated-code spans separately from original source witnesses before certifying symbolic emitted decisions; missing support keeps that decision unresolved. Preserve a typed nullable result at an unresolved authority boundary rather than fabricating a clause.

## Verify event interpreters without choosing their effects

Keep timing, replacement applicability, simultaneous merging and trigger recipient selection in typed NL stages. Exact commits validate before-snapshot identity and apply the returned complete deltas; external checks replay those supplied deltas and source-defined token capacities. Trace capture/current/last-known references independently and allow harmless sibling-set permutations while preserving window chronology. A source-bounded work-slot universe proves termination only when newly queued work activates existing capacity; new source rules require a revised closure proof. Canonical traces and counts do not establish actual literal-result consumption.

For source-form entity keys, exact checks validate a returned complete lexical span and key equality to its lowercased source form. A substring check alone can accept an inflected word's prefix; check declared lexical boundaries too. This is span validation, not semantic noun recognition. Validate downstream endpoint membership in supplied forms and preserve exact keys through typed projections. Trace actual parent consumption of narrow child decisions before deeper named calls; field declarations and static replay alone do not prove those calls occurred.

Supply actual generated child bodies and referenced runtime definition data to symbolic code interpretation, preserving that closure's provenance. Keep expected branch guidance, normative schema invariants and the reference-state channel out of the emitted-behavior context. Shared source definitions can be runtime data only when the emitted body reads them; they cannot override a wrong emitted branch. Actual generated callable execution is separately qualified.

For connection models, inspect typed physical presence, ordered direction and state-dependent permission separately. Exact graph projection may create only the directions selected by source-grounded NL interpretation and must preserve negative witnesses. A retry that drops absence or prohibition evidence does not qualify the repaired model. Pin representation changes separately from their stopped diagnostics and check source/type replay independently from live evidence consumption.

Verify event certificates with typed source identities and separately tracked generated IDs. Positive controls should alpha-rename generated slots, revisions and rule IDs without rewriting source quotations or original clause IDs. Keep unordered witness/board rows independent from ordered window and rewrite histories. A malformed identity error should name the missing source or generated reference domain.

Procedure Compiler code-closure attribution: an emitted-behavior literal returns generated spans with the actual closure file path and exact nonempty quote. The parent checks closure membership and quote bytes before consuming a decided permission/effect. Keep original source witnesses in their separate domain. If a quoted callee is absent, retain unresolved behavior and recover by supplying the actual emitted callee file; source-required policy remains in the independent judgment.

For symbolic interpretation of generated modules, discover the actual transitive code closure with an NL file frontier. Preserve caller-path evidence and distinguish invoked calls, defining type files and explicit runtime source reads from intended plans or explanatory mentions. Activate only original unvisited files; report a missing referenced file by its caller/target identity and recover by supplying that actual file. Do not substitute ideal policy as emitted behavior.

An ordering-preview verifier checks candidate identity/permutations, source precedence, common input revision, branch-local reservations and the exact selected preview's later consumption. Add negative controls for leaked preview tokens and a selection with no source basis. Such replay does not prove semantic equivalence; retain independent NL comparison and source-choice review.

Type-closure resolution needs actual defining declaration text, not file names alone. A typed dependency carries exact caller evidence separately from exact target-definition evidence. A missing definition reports the requested symbol and actual target path; recover by supplying the defining type file, preserving unresolved status until its original quote is verified.

When rendering a new iteration child, audit inherited footers as well as its body: the child performs one step and returns its declared state; the parent invokes iterateOn on that state. A parent's prior stage record is not interchangeable with its initialized frontier. Verify callable path/type visibility, a decreasing fixed-universe measure, explicit blocked termination and consumption of returned findings. Object literals may materialize a semantic result; classify policy displacement from actual algorithm/ownership bypass, not object syntax alone.

For survey extraction, an NL inventory returns original reading-unit IDs and scope questions, then directly awaited per-unit NL children return typed semantic meanings. Parent arrays consume those actual results; they do not precompute passage, landmark, gear or sensor fields. Bind clause-keyed exact quotations to immutable original source identity before downstream use. Frame-dependent readers receive the actual FrameSurvey; local origin and global anchor retain separate coordinates. Missing quote/endpoint errors identify the reading unit and original clause; recover by supplying its actual frame/supporting evidence and re-invoking that child. Internal absent/unknown passage costs may be null; executable edges require a known source-permitted cost.

Trace final forwarding by invocation identity: a returned typed child value should be consumed once, and its parent should finish that first return. Render a positive final `eval({code:"return await child(nextStageInput);",finish:true})` contract with the actual child name after binding the complete typed input. A staged-result recovery finishes the available value instead of re-executing the child chain. For narrow judgments, direct literal instructions avoid confusing captured instruction data with callable bindings; measure this presentation separately from source/type replay.

An adaptive experiment certificate carries actual per-world predictions, answer likelihoods, branch-local stock and posterior updates. Exact verification may normalize rational masses, validate source capacities and recompute costs, while source interpretation and experiment selection stay in NL stages. Check that hypothetical siblings inherit separate parent snapshots and original experiment slots cannot be regenerated under new IDs. Record goal-class completion separately from full-state identification. A canonical policy witness and exact finite enumerator are external review tools, never model-visible substitutes for the application's semantic decisions.

Audit exact graders for hidden field-role conventions: focal subject, mother/father and assayed tissue can be different identities. Make identity roles source-visible before exact checking, and accept equivalent free-prose descriptions separately from fixed source IDs and answer codes. A failed exact description comparison is a verifier defect when the source never fixed its wording.

Review posterior updates against complete joint state, not only the named target trait. A different measured quantity is not automatically uninformative. Verify that the actual compartment-specific prediction reaches the likelihood updater; neutral and unknown likelihoods have distinct source conditions and downstream readiness.

Check collection iteration separately from semantic stage correctness. Bind fixed folder/source inventories outside a one-state step, carry its index and processed original IDs, and validate one visit per source record before merging returned contributions. Exact append/index/coverage mechanics may be crisp; interpretation remains in the called NL stages. A successful first folder is not evidence that later folders received distinct inputs or the actual accumulated state. Trace the cursor, original IDs and final completed-state return in the next pinned diagnostic.

### Dialogue perspective application integration

Preserve the original document IDs/paths/text and full actor/time view handoff for dialogue reducers.
Keep performed and depicted speech distinct, and retain each scoped quotation/reference/endorsement/
condition result in its actual parent-owned typed contribution. A proof challenge must reach the final
answer; no wrapper may erase it or silently rebuild a preferred proof. Exact checks validate source
span containment, original quotation bytes, clock arithmetic and actor-owned delta application; speech
meaning, availability/adoption and commitment scope remain NL. Invalid source spans identify their
original turn/range for repair; recover from that exact input. Inspect live nested literals and before/after
view deltas before qualification. A projected query oracle is not a full typed report or executor replay.

Preserve Dialogue commitment activation and fulfillment arrays independently through typed stage handoffs. Do not reinterpret a performed report as a performed promise: the query force determines that observable. Validate nested quoted provenance using original bounds and parent IDs; node collection order may vary, while sibling source order and episode chronology remain meaningful.

When rendering batch-delta pipelines, keep generic footers consistent with the per-stage ownership contract. Distinguish current input.prior from context.priorCompletedState, and terminal batch arrays from a global audit field. Validate source-ID coverage and actual decision→edge→assessment→scope derivation before the single root merge; permit older exact supporting witnesses. On an ownership violation, identify the stage/source and instruct it to return only the current batch contribution while retaining earlier evidence in context. Check terminal child result types before adding generic forwarding instructions.

When auditing semantic policy stages, distinguish typed result materialization from crisp policy displacement. Trace actual prediction/child-value fields into observable partitions, completeness decisions and selected values. Exact arithmetic/provenance is a seam, while assay equivalence, source objective interpretation and uncertainty remain NL-owned. On a missing dependency, identify the original world/answer/node and repair that typed contribution; do not substitute zero probability or a losing state. A source prose refinement requires its own pinned peer/live receipt even when fixture and canonical outputs are unchanged.

Audit instruction nouns against the actual signature and caller fields. If an assay grammar or goal is referenced without an input, pass the actual typed contract or leave that judgment with its upstream owner. On an unavailable-input failure, repair the caller/signature rather than supplying guessed fixture facts.
