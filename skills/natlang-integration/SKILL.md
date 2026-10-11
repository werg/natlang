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

Pair with `natlang-authoring` for substantial natural-language algorithms when that skill is available; this skill is independently usable.

Persistent eval helpers support TypeScript `as const` assertions for literal and tuple results; the assertion needs no captured type binding. Other referenced named types must remain available in the call scope.

## Student prompt rendering

Use `promptRendering: "program-header"` in the model profile (or runtime `model`
settings) for students trained beyond the full-prompt warm-up. Match the checkpoint's
`prompt_rendering` declaration. Other models default to `"full"`.
Program-header requests retain the signature, types, instructions, inputs, skills,
call-specific declarations and trajectory. They omit generic interpreter prose and
wire tool schemas; the same tool calls still execute. Custom application instructions
remain. Training and serving use the scope table exported by
`node ts-host/scripts/neuralese-prompt-pieces.mjs` after building ts-host.
