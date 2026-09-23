# TypeScript-native natlang: implementation status

Working record for the cutover specified in [TS_INLINE_HOST_INTEGRATION_PLAN.md](TS_INLINE_HOST_INTEGRATION_PLAN.md),
[inline-natlang-lambdas.md](inline-natlang-lambdas.md), and [ITERATE_ON_PLAN.md](ITERATE_ON_PLAN.md).
It records the concrete module layout and the decisions taken where the plans leave room.

## Module layout (ts-host/src)

| Module | Role |
|---|---|
| `compiler/intrinsics.ts` | Declarations for `nl`, `iterateOn`, `NatlangFunction`, `Iteration`, `Folder` (global and module forms) |
| `compiler/host.ts` | Compiler host over real or virtual files; shared library `SourceFile` cache; browser lib provider |
| `compiler/targets.ts` | `ts.Type` to target descriptors (portable natlang type text, aliases, live host contracts) |
| `compiler/inline.ts` | `nl` discovery by resolved symbol, captures by exact mention, signature inference, `InlineLambdaPlan` |
| `compiler/eval-check.ts` | Checked analysis of `nl` in eval snippets against the session's typed scope |
| `compiler/policy.ts` | Constrained-source policy: finite iteration, no dynamic code, authored-call graph recursion |
| `compiler/lower.ts` | Lowering: `nl` to `nl.__inline(...)`, `iterateOn` site IDs, entry guards, finite iteration, browser await restoration, `.nl` import specifiers |
| `compiler/project.ts` | `compileProject` over a `ProjectFiles` interface: `.d.nl.ts` generation, callable-folder checks, emit, manifest |
| `compiler/node-project.ts` | Node adapter: `buildProject`, `checkProject` |
| `scope-compiler.ts` | Eval snippet compiler (REPL semantics, bindings, loop/recursion policy, `nl` lowering) |
| `native/runtime.ts` | The interpreter: one natlang invocation, its session tools, eval over the live channel |
| `native/agent.ts` | Model tool loop and the scope opening (captures, services, imports tree) |
| `runtime/context.ts` | Frames, context stores (Node `AsyncLocalStorage`, browser slot + await restoration), recursion guard |
| `runtime/runtime.ts` | `createNatlangRuntime`, tasks, `bind`, services recording |
| `runtime/kernel.ts` | `invokeDefinition`: the single invocation path |
| `runtime/hooks.ts` | What the kernel gives each interpreter run (callables, inline, iterateOn, guards, analysis) |
| `runtime/loader.ts` | Records for `.nl`, TS modules, folders; `natlang.d/` discovery |
| `runtime/modules.ts` | Callable-folder module instances (host realm, lowered, one per revision) |
| `runtime/callable.ts` | Callable objects with child attributes, `iterateOn`, and the folder `apply` protocol |
| `runtime/iterate.ts` | `iterateOn`, statistics, progress judge, errors |
| `runtime/lowered.ts` | Runtime support targeted by compiled code |
| `runtime/virtual-project.ts` | Compile and run in-memory projects (browser playground, workers, tests) |
| `app/event-loop.ts` | `EventLoop`, `EventQueue` |
| `teacher/program.ts` | Program IR: `natlang.program/2` records, root nodes, `definitionProject` |
| `app/playground.ts` | Playground projects: validation, `projectEntry`, `projectSignature`, pinned runs, trace frames, admission |
| `native/node-files.ts` | `openFolder` (lazy disk-backed `Folder`) and `saveFolder` |

## Decisions

The plan documents now record the decisions (recursion by caller chain, version-checked capture write-back,
`natlang.d/` application context, typed host services, Node `AsyncLocalStorage` plus browser await-restoration,
removal of Map/Fold/Iterate, and a two-verdict progress judge). Only implementation details live here.

- **Live values in the Node evaluator.** The `vm` evaluator runs in the same isolate, so live objects, functions and capture
  cells enter an eval by reference through a private channel beside the portable scope snapshot.

## Progress

- [x] Compiler: intrinsics, virtual host, targets, inline analysis, diagnostics, eval analysis
- [x] Compiler: policy (loops, dynamic code, call graph), lowering, project check/build, `.d.nl.ts`
- [x] Runtime: context, kernel, callables, live values and captures with version-checked write-back, recursion guard, services
- [x] Runtime: `iterateOn`, statistics, progress judge; Map/Fold/Iterate removed
- [x] Eval: `nl` in eval, callable hierarchy injection and listing, TS module instances, loop policy
- [x] Public API: Node entry, `.nl` loader, `defineNatlang`, CLI `check`/`build`/`run`/`call`/`ask`, package manifest v2
- [x] Browser: runtime binding, await restoration, virtual projects, local model loader, playground, Studio (22 apps and the research lab), board, browser-local pilot; Chromium smokes pass
- [x] Folders: one `Folder` abstraction; `openFolder`/`saveFolder` replace the lazy file-tree inputs
- [x] Applications: one TS project in `applications/` (wiki, scheduling, evidence, publisher, logs, notebook, build, media, terminal, games, ide, workflow, migration); console packages use `natlang.package/v2`
- [x] Examples (`examples/triage`, `examples/npm_app`) and the corpus `codebases/` on services, `iterateOn`, and the loop policy
- [x] Program IR `natlang.program/2` (projects); collector, synthetic generator, source cases, failure corpus, code corpus, playground import; tracked IR migrated
- [x] Conformance programs as projects with reference agents
- [x] Python runtime, Python applications, and runtime-dependent scripts and tests removed; training tools kept runtime-free
- [x] Skills, spec, README, setup, packages, and application guides rewritten
- [ ] Open: `natlang.program/2` adapters for the retired external datasets (decisions, SCONE, SGD, FinQA, CLEVR) and the semantic-merge scenario set
- [ ] Open: a live-model smoke of each shipped application (fixture and Chromium smokes pass)

## Model surface revision (2026-09-23)

Reworked against live Bonsai 27B runs (`ts-host/scripts/live-probe/`):

- The opening is the call's signature and instructions, then a runtime-run eval that reads
  the arguments with `read_inputs()` into typed consts (their values shown in the eval's
  result), declares functions and services, and for directory reducers a prefilled
  `list_files`.
- Completion: the `return_result(value)` tool finishes; a top-level eval `return` or
  `return_result(...)` inside eval stages a computed value, and a reply without a tool call
  returns it (or, for a string-typed call, the reply text is the result). `blocked(missing)`
  and `failed(message)` end without a result, as tools or eval functions. `mark_lines`, line
  listings, the `result` variable, the final-expression result rule, and `commit` are removed.
- `read_page(id, page)` reads cut-off output by a short word ID; `read_value` is removed.
- Parameters are `const` and frozen. The `debug` binding and the failure paragraph in the
  system prompt are removed; the system prompt is fixed for the whole call.
- No default limits: repairs, nudges, conversation segmentation, eval time, turn tokens, and
  sampling temperature apply only when configured. Eval accepts an optional `timeout_ms`.
- Eval allows `try`/`catch`, classes, and Node host globals; package imports resolve from the
  workspace like any module (the npm install layer is removed).

### Training-data pipeline: state after the revision

Done: collector (`maxTurns`/`--max-turns`, run ID from program and seed only), materializer
(the durable opening includes the prefilled scope exchanges), code-corpus replay (eval then
done), playground import (no tool calls in program IR), SFT selection (current tool names;
`return_result` and replies are terminal turns), improvement pipeline (`--max-turns`),
teacher and synthetic-data docs; the orphan `scripts/prompts/no_fudging.md` is removed.

Still to do:

- [ ] Regenerate every derived trace, turn, and SFT file under the new surface; the tracked
      `data/direct-code-2026-09-23/` replays and turns still show `mark_lines` and the old
      prompt. Program IR does not change.
- [ ] Rewrite `var` to `let` in code that becomes eval bodies (see PROGRAM_IR_PIPELINE.md).
- [ ] Re-run teacher collections with an explicit `--max-turns`.
- [ ] Re-audit the probe families whose scoring assumed line marks (read-before-code) and the
      student-improvement evidence kinds (`premature_reply`) against the new completion rule.
