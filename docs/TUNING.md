# Tuning a program

`natlang tune` specializes a student model to one natlang program. The program and its call sites stay unchanged:
after a successful tune, ordinary `natlang run`, `natlang call` and SDK runtimes use the tuned release on their own.
The design is [plans/HIERARCHICAL_LORA_TUNING.md](../plans/HIERARCHICAL_LORA_TUNING.md); the implementation
decomposition is [applications/tuner/DECOMPOSITION.md](../applications/tuner/DECOMPOSITION.md).

```sh
natlang setup                 # one-time model/runtime configuration (includes the tune block)
natlang tune ./my-program     # collect/reuse data, train, validate and activate
natlang run ./my-program      # ordinary execution uses the result
```

## What the author supplies

- The ordinary typed program.
- Root examples in `tune.examples.jsonl` beside it, one JSON object per line:
  `{"function": "triage/label-ticket.nl", "input": {"ticket": "..."}, "expected": "billing", "group": "customer-17"}`.
  `expected` is optional; with it the root has a direct oracle. `group` names the source/world component: examples
  that share a world stay on one side of every split. `"protected": true` keeps an example out of all tuning; it is
  for final reporting only.

No adapter manifest, adapter IDs, export, conversion or server-loading step is part of the author's work.

## What a tune does

1. Discovers the program's bound definitions and compiles the lexical scope tree: program root, one folder node per
   source directory (an author-declared tree root becomes its own tree level), one lambda node per named `.nl`
   definition or persistent inline `nl` site.
2. Splits root examples by component into training, development and confirmation before anything is expanded.
3. Runs the program with the configured authorized teacher and records every inner model turn with the binding of
   the invocation that made it, its exact request and reply, and separate evidence (root oracle, enclosing success,
   call outcome).
4. Trains one joint fit: every training turn updates its own binding's path (root, folders, lambda), against one
   parameter snapshot per update.
5. Exports each trained node for the serving backend, stores a content-addressed release under
   `.natlang/tune/releases/<digest>/`, loads it, runs development roots through the normal engine with strict
   resolution, and only then activates it by atomically replacing `.natlang/tune/active.json`.

Rerunning the same command reuses collected data and training receipts. The report prints the time from invoking
`tune` to the first tuned call, split by stage.

## How calls use a release

Each call runs its **own** definition's stack: its families, then its scope ancestry from the program root to its own
lambda. A callee never inherits its caller's adapters, and two aliases of one definition share one stack. A rebound
definition (`fn.in(context)`) has no tuned binding of its own and runs the base. Explicit
`withNativeAdapters(...)` scopes are appended after the lexical stack.

A task pins the release that was active when it started; inner calls use the same release. A newly activated release
serves new tasks; running tasks finish on theirs.

## Status and recovery

`natlang tune ./my-program --report` prints the status, the active release and the latest report.

| Status | Meaning | What to do |
| --- | --- | --- |
| `untuned` | No release | Run `natlang tune` |
| `tuning` | A tune job is running | Calls use the current release or the base until it activates |
| `ready` | The active release serves | Nothing |
| `stale` | Definitions changed since tuning; changed ones run untuned | `natlang tune --incremental` |
| `degraded` | The backend cannot serve the release (another base model, missing export, unloaded adapter) | Rerun `natlang tune` for the current backend and model |

Errors from strict resolution name the cause:

- `weights-unavailable`: an adapter in the call's stack is not loaded by the serving backend. Rerun `natlang tune`, or
  restart the process so the managed server loads the active release.
- `weights-incompatible`: the release was trained or exported for another exact base model. Retune for this model.
- `weights-stale`: a definition changed after tuning. Run `natlang tune --incremental`.

`natlang tune ./my-program --rollback` reactivates the previous coherent release. `--no-weights` on `run`/`call`
runs the base model.

## SDK hosts

```ts
import { activeWeights, createNatlangRuntime } from '@natlang/node';

const runtime = createNatlangRuntime({ model, program, weights: activeWeights(programRoot, { program }) });
```

`activeWeights` rereads the active pointer at each task creation. The model driver must serve the release: the
managed llama.cpp session loads GGUF exports (`loraAdapters`) and sets each request's own stack; the native
reference server applies the stack through its exact native adapter path.

## Migration

`activeNativeAdapters()` (what model drivers read) now returns the invocation's lexical stack followed by the
explicit `withNativeAdapters` scopes. Programs without a release are unaffected. Code that relied on
`withNativeAdapters` being the only adapter source inside a tuned program now sees the program's stack first; pass
`weights: []` to a task to run it without its release.

## Current coverage

Implemented: scope compilation, per-call resolution with requested/effective identities and coherent fallback,
release store with compare-before-activate and rollback, `natlang tune` (collect, split, train, export, validate,
activate, report, watch), managed llama.cpp serving with per-request stacks, the joint trainer with nested shared
bases and free LoRA factors, and per-node GGUF export. Not yet implemented: cross-program families
(`tune --families`), placement/ablation, rank sweeps, base distillation, vLLM and wllama serving, on-policy repair
rounds and dependency-closure incremental training. These remain required work in the plan.
