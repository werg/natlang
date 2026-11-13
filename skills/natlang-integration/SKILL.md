---
name: natlang-integration
description: Embed natlang in Node/TypeScript and browser applications. Use when wiring the runtime, model drivers, host services, compiled `nl` calls, event loops, terminal or browser interfaces, directory reducers, packages, persistence, traces, call records and compilations, native Neuralese model servers, tuned weight releases, or application evaluations.
---

# Integrate natlang applications

Build an application in which natlang makes the judgments and ordinary TypeScript does the rest: storage, transport, exact operations, and presentation. Natural-language functions are ordinary async functions in the application; there is no separate program runner.

Start by locating the installed `@natlang/node` / `@natlang/browser` version or the checkout and any existing embedding. Reuse its runtime and model lifecycle. Repository paths in the references are lookup hints relative to a natlang checkout.

## Choose the smallest adequate boundary

Read [hosts and model drivers](references/hosts.md) for the runtime, compilation, services, and model transport; [frontend applications](references/frontend.md) for browser models, event loops, and generated interfaces; [terminal applications](references/terminal.md) for CLIs, packages, sessions, and event streams; [effects and recovery](references/recovery.md) for native objects, concurrency, and durability; [call records and compilations](references/records.md) for the machine's record of calls, opting values out of it, and compiled cases. Read [refinements](references/refinements.md) for `Is<T, P>` settings, crisp checkers, the `refinement-*` error codes and their recovery. Read [fused pipelines](references/fusion.md) for the `fusion` setting, the qualification certificate and the `fusion_*` trace events. Read [native Neuralese state](references/native-state.md) for native model servers, `Neuralese<T>` values and student prompt rendering. Read [delivery scenarios](references/delivery.md) before claiming an integration complete.

1. Decide who owns state, native objects, and effects. Host capabilities become typed services passed to the runtime; natlang code reaches them only through `natlang:services` or eval bindings.
2. Write the application in TypeScript and compile it with `natlang build` (or `buildProject` / `compileVirtualProject`) so `nl` calls are planned and typed. Named `.nl` functions import like modules.
3. Create one runtime per application (`createNatlangRuntime({ model, services })`) and run natlang work inside `runtime.run(...)`. Callbacks from uncompiled code (DOM events, timers, libraries) use `runtime.bind(fn)`.
4. Define event ordering and commit points with ordinary code or `EventLoop` (`KeyedEventLoop` for one loop per user or document). Persist operation identities and observations for effects that must not repeat.
5. Test the actual engine and transport, success and partial failure. Make general fixes in the shared runtime, not as app-local workarounds.

## Operational commitments

- Omit arbitrary run budgets unless deployment requires them; `limits` and model options exist for that. A call longer than the model's context (`contextTokens`) is compacted: the model writes a note, and its full history stays searchable from eval as `transcript`.
- Set budgets (`maxTurns`, `maxTokens`, `maxSeconds`, `maxFailureRepairs`) explicitly for evaluations; none apply by default.
  A `.nl` function's `budget:` frontmatter replaces the model's bounds for its own calls.
- Every failure of a call rejects with `NatlangCallError` naming the call (`callId`) with a typed `reason`
  (`callFailureReason`); a failure that is not the model's outcome (driver, arguments, host) has reason `error` and keeps
  the original error as `cause`, and a cancelled call has reason `cancelled`. Classify infrastructure failures by walking
  `cause`, not by the class of the rejection.
- Services and live values are trusted native authority passed by reference. Eval is trusted code in the application's process, not a sandbox.
- Service calls and live-object writes happen immediately and are traced as effects; they are not rolled back when a call fails. Captured `let` variables are written back only after a successful eval.
- Every call is recorded in the machine's call store (exact values within bounds) and may be served by a compiled crisp case; a case that fails hands the call back to the function. Exclude sensitive values with `"recording": { "exclude": [...] }` in `natlang.json`, and give embedded runtimes `programRoot`. See [records](references/records.md).
- A program declares its model in `natlang.json` (`"model": { "stock": "lfm3-1.7b-q8_0" }`, or `profile`, `provider`+`model`, `endpoint`, `request`, `contextTokens`); a command's `--profile`/`--provider`/`--model` or `NATLANG_PROFILE`/`SERVER`/`PROVIDER` still win. Give it `"warmup": [{ "function": "classify.nl", "inputs": { ... } }]` and ship it built with `natlang build --warm`: the lock (`.natlang/build/natlang.lock.json`) binds the model file, llama.cpp build, tuning release and functions, the warm-up calls are traced, and their prompt openings (the long system prompt and tools every call starts with) are prefilled and saved, so `natlang run` starts warm; the managed server also saves any opening it meets for later runs (`NATLANG_WARM_PREFIXES=off` disables it). `natlang dashboard` runs, traces and configures all of this in a browser.
- A call whose model repeats the same failing action four times in a row ends as `model_incomplete: repeated_failed_action` instead of running on silently; give it a better model, clearer instructions or an escalation profile.
- Seeds and pinned sources support reproducibility; they do not make native state replayable or decisions identical across inference backends.
- Native input and scope listings show at least 4,096 characters per value. A cut-off listing names the complete live eval variable; use ordinary code such as `value.slice(start, end)` to inspect omitted ranges. Tool-result text is sized from the call's context budget (`contextTokens`; `shownCharsFor` in `ts-host/src/native/cutoff.ts`: an eighth of the budget at 3.5 characters per token, at least 2,000), and only output longer than that is paged with `read_page`. A host that sets a larger `contextTokens` therefore shows longer outputs whole; `contextTokens: null` keeps the 2,000-character floor.
- If no tool output was cut off in the call, `read_page` directs the agent to inspect the in-scope value with `eval`; only an actual cut-off tool output creates a page ID.
- Eval exposes `inputs` and `read_inputs()` only for calls with formal input fields, unless those names are shadowed. A zero-argument call has no implicit caller-input object; ordinary local declarations may use those names.
- Verify live-model quality separately from fixture wiring and rendering, and name the remaining empirical gates.
- A call that ends `model_incomplete: terminated_without_result` (two empty replies after the recovery prompt) or `truncated_without_action` (replies cut off before any action) is an incomplete model answer, not a transport failure; keep the two apart in retries and reports.
- For a folder reducer, keep item-file and task-file contracts distinct: bind each value from the file that owns it before constructing per-item calls, and keep explicit empty lists empty.
- Service methods that can reject arguments should say why in one sentence with the correct positional call and its declared types; the model recovers from that sentence. Error text a model reads is model-facing: measure a change to it on the live executor.

Pair with `natlang-authoring` for substantial natural-language algorithms when that skill is available; this skill is independently usable.

## Tuned weight releases

`natlangRuntime({ programRoot })` serves a program's active release on the checkpoint it was tuned on and loads it into the server; prefer it in hosts (see [hosts](references/hosts.md)). Hosts that build the runtime themselves pass `weights: activeWeights(programRoot, { program })` to `createNatlangRuntime`; each new task pins the release active at its creation, so activation never changes a running task. The model driver must serve the release: the managed llama.cpp session takes the release's GGUF exports through `loraAdapters` and every request carries its own call's stack; other backends report the release `degraded` and run the base. Do not pass adapter IDs or wrap calls to select a stack; resolution is per callee. Use strict `WeightBinding` resolution in evaluation so missing artifacts throw (`weights-unavailable`, `weights-incompatible`, `weights-stale`) instead of producing base results labelled as tuned. See [docs/TUNING.md](../../docs/TUNING.md).

## Trained native models

`natlang setup --native SOURCE [--gguf q8_0]` pins a trained checkpoint and writes its profile; the native profile
starts its reference server on the first model call (`natlang serve` keeps it loaded), and `NAME-gguf` runs the same
weights on managed llama.cpp for ordinary tokens. A profile's `escalate: PROFILE` (or `--escalate`) reruns a call the
student fails on another profile through the tiered engine and says so on stderr; the escalated root call's input
joins the program's `tune.examples.jsonl` without an expected result. `natlang traces good|bad|fix CALL` turns tested
calls into roots; `good` refuses outputs of models that are neither a tuning teacher nor one of our native models
(recovery: `natlang traces fix CALL --expected JSON` with the answer you checked). `natlang tune DIR` needs no
configuration with a native profile: student, trainer and teacher (`escalate`) come from it. `natlang doctor
[--program DIR]` shows the pin, newer checkpoints, the server, the tuning setup and pending examples. See
[docs/NATIVE_MODELS.md](../../docs/NATIVE_MODELS.md).

## Decision readouts in tuning

Decision readouts (`readout: decision`, `decide(fn, ...)`) are trainable model outputs: SFT supervises the choice over
the option set, and tuning rollouts (`natlang tune`) sample the answer from the scored distribution at the rollout
temperature for exploration (recorded with the capture) instead of taking the most probable option. Ordinary runs keep
the most probable option. A tuning rollout needs a native server for decisions (`rollout-capture-unavailable`
otherwise: serve the student with `natlang serve` or a native profile).

Internal notes on the repository's semantic training applications and their hosts are in `docs/SEMANTIC_APPLICATION_NOTES.md`, and native learning and training-capture contracts in `docs/NATIVE_RUNTIME_NOTES.md` (both in the natlang repository); applications built with this skill do not need them.
