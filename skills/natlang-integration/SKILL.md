---
name: natlang-integration
description: Embed natlang in Node/TypeScript and browser applications. Use when wiring the runtime, model drivers, host services, compiled `nl` calls, event loops, terminal or browser interfaces, directory reducers, packages, persistence, traces, or application evaluations.
---

# Integrate natlang applications

Build an application in which natlang makes the judgments and ordinary TypeScript does the rest: storage, transport, exact operations, and presentation. Natural-language functions are ordinary async functions in the application; there is no separate program runner.

Start by locating the installed `@natlang/node` / `@natlang/browser` version or the checkout and any existing embedding. Reuse its runtime and model lifecycle. Repository paths in the references are lookup hints relative to a natlang checkout.

## Choose the smallest adequate boundary

Read [hosts and model drivers](references/hosts.md) for the runtime, compilation, services, and model transport; [frontend applications](references/frontend.md) for browser models, event loops, and generated interfaces; [terminal applications](references/terminal.md) for CLIs, packages, sessions, and event streams; [effects and recovery](references/recovery.md) for native objects, concurrency, and durability. Read [delivery scenarios](references/delivery.md) before claiming an integration complete.

1. Decide who owns state, native objects, and effects. Host capabilities become typed services passed to the runtime; natlang code reaches them only through `natlang:services` or eval bindings.
2. Write the application in TypeScript and compile it with `natlang build` (or `buildProject` / `compileVirtualProject`) so `nl` calls are planned and typed. Named `.nl` functions import like modules.
3. Create one runtime per application (`createNatlangRuntime({ model, services })`) and run natlang work inside `runtime.run(...)`. Callbacks from uncompiled code (DOM events, timers, libraries) use `runtime.bind(fn)`.
4. Define event ordering and commit points with ordinary code or `EventLoop`. Persist operation identities and observations for effects that must not repeat.
5. Test the actual engine and transport, success and partial failure. Make general fixes in the shared runtime, not as app-local workarounds.

## Operational commitments

- Omit arbitrary run budgets unless deployment requires them; `limits` and model options exist for that. Conversation segmentation and backend context are separate controls.
- Set budgets (`maxTurns`, `maxTokens`, `maxSeconds`, `maxFailureRepairs`) explicitly for evaluations; none apply by default.
- Services and live values are trusted native authority passed by reference. Eval is trusted code in the application's process, not a sandbox.
- Service calls and live-object writes happen immediately and are traced as effects; they are not rolled back when a call fails. Captured `let` variables are written back only after a successful eval.
- Seeds and pinned sources support reproducibility; they do not make native state replayable or decisions identical across inference backends.
- Verify live-model quality separately from fixture wiring and rendering, and name the remaining empirical gates.

Pair with `natlang-authoring` for substantial natural-language algorithms when that skill is available; this skill is independently usable.
