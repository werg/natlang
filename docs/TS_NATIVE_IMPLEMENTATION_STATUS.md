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
