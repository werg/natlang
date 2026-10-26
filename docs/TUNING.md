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

Rerunning the same command reuses collected data and training receipts. Splits are persisted
(`.natlang/tune/splits.json`), so later edits never move an example between partitions. Collected data is kept per
root and recollected only when a definition that root reached changed, or when non-instruction sources (helpers,
services, types) changed; `--incremental` prints which bindings changed and which shared nodes and dependents must
be refit. The report prints the time from invoking
`tune` to the first tuned call, split by stage.

## Edits and renames

Named `.nl` definitions keep their identity by path. Unlabeled inline `nl` sites are identified by their file
revision and position, so the tuner keeps its own persistent site registry (`.natlang/tune/sites.json`): an edit
elsewhere in the file keeps a site (same text and contract), and an edited instruction keeps its site when it is the
only candidate with that contract in its file. Ambiguous edits become new sites; nothing inherits another site's
adapters. A renamed file, or an ambiguous edit you want carried over, goes in `tune.migrations.json` beside the
program: `{ "<site ID from sites.json>": "<current component key>" }`. A site that only moved reuses its data and
training; `natlang tune --incremental` reseals and revalidates the release for the new keys. Labeling a site
(`/* @natlangSite name */`) makes its identity independent of edits.

## Collecting through the generation pool

When teacher data comes from the operators' reviewed generation pool instead of a configured teacher profile:

```sh
natlang tune ./my-program --export-cases runs/tune-cases/my-program-1   # roots without valid data
# operators review and dispatch source.jsonl / cases.json through the existing pool
natlang tune ./my-program --import-cases runs/<campaign>/               # collector results → tune data
natlang tune ./my-program                                               # reuses the imported data
```

The export holds only training and development roots that have an `expected` result and no valid data
(`--include-confirmation` adds confirmation roots; protected roots never leave the program). Each case binds the
program build, the scope tree with its site IDs and the split receipt; the import rejects rows collected under any
other. Turns map to their invocation's binding exactly as at serving time and keep the pool actor's identity as
`teacher`; roots the teacher got wrong are kept as failures, never as targets. The pool's collector adds a block of
collection guidance to the system prompt; imported rows present the ordinary serving prompt with exactly that block
removed. Programs with program-level guidance, imported programs or services cannot use the pool yet.

## Families across programs

`natlang tune --families ./prog-a ./prog-b ...` looks for behavior several tuned programs share. All programs must
have an active release on the same exact base model.

1. Discovery compares every tuned definition's effective weight change across programs (a gauge-invariant
   similarity, never raw adapter factors) and clusters those above `tune.families.threshold` (default 0.5) that
   span at least `tune.families.minPrograms` programs (default 2). Author tags in `tune.families.json` beside a
   program, `{ "<family>": ["<definition source path>", ...] }`, are always proposed.
2. The proposal policy (`tune.families.mode`: `crisp` by default, or `nl`/`shadow` with
   `applications/tuner/proposeFamilies.nl`) chooses the families to fit.
3. Each family is fit once on its members' training rows only, with every program-private adapter absent and each
   program weighted equally, then normalized to a fixed amplitude and published as an immutable version in
   `<natlang config>/tune/families/<base digest>/`. Refitting a published name keeps its amplitude.
4. Each member program is refit with the family frozen and a gate per member definition (starting at 1, trained
   with an L1 penalty, 0 removes it), measured against its previous release on development roots, and activated only
   within `tune.placement.maxSuccessLoss`. The report flags negative transfer; `--rollback` restores the previous
   release. Programs you do not name never contribute data.

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

`natlang tune ./my-program --evaluate development|confirmation|protected [--with-teacher]` measures the untuned base,
the active release and optionally the teacher reference on that partition, paired by root, with intervals clustered by
example group. `--placement` refits candidate removals deepest-first on development roots and activates the cheapest
candidate whose development success stays within the configured envelope (`tune.placement.maxSuccessLoss`, cost
weights in `tune.placement.costWeights`). `--sweep spec.json` runs a resumable rank/penalty/data-size sweep. The
report (`--report`, `--partition`) shows every evaluated arm.

With several authorized teachers in `tune.teachers`, `natlang tune ./my-program --calibrate` runs each on the
development roots without weights and records success, failures, tokens and time per root under its exact executor
identity. Teachers below `tune.teacherRouting.minSuccess` (default 0.5) are never chosen; among the rest the
cheapest teacher within 0.05 of the best success rate collects the data. A changed teacher identity requires a new
`--calibrate`; `--teacher <profile>` overrides the choice.

`natlang tune ./my-program --rounds` runs SFT and then RL rounds (`tune.rounds`: `schedule`, default
`["sft","rl","rl","rl"]`; `patience`; `samples` per training root, default 4; `temperature`, default 0.8;
`rlRecipe`). Each round continues the best build so far; an RL round samples the training roots from that release
with exact sampling receipts and fits on those rollouts together with the teacher rows. Rounds are compared on
development roots only, and the best release is validated and activated as usual. RL rounds need the native
reference server (`tune.backend: "native"`); an SFT-only schedule works on every backend.

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

## Backends

| Backend | Weight residuals | Soft instruction/skill states | Mixed stacks in one batch |
| --- | --- | --- | --- |
| Managed llama.cpp (upstream runtime) | Per-node GGUF LoRA, per-request scales | No: soft-state bindings run degraded | No (serialized per stack) |
| llama.cpp fork (`--lora-per-sequence`) | Same | Through its Neuralese server (not yet wired here) | Yes, per-sequence scales, verified bitwise isolation |
| Native reference server (`neuraleseServer: true` profile) | Exact native coefficient artifacts | Yes | Per request |
| vLLM (`vllm: true` profile) | One whole-stack PEFT LoRA per distinct effective stack, loaded at activation | No: soft-state bindings run degraded | Yes (multi-LoRA batching) |
| wllama in the page (vendored build, `loadBrowserRelease`) | Per-node GGUF LoRA, per-request scales | No: soft-state bindings run degraded | No (serialized per stack) |

In a page, `loadBrowserRelease(releaseUrl, { baseGgufSha256 })` fetches `natlang.adapters.json` and its GGUF exports,
checks each file's sha256, and returns the adapters for `loadBrowserLocalModel(source, { loraAdapters })` and the
binding for `createNatlangRuntime({ weights })`. Adapters load at scale 0; each request lists every adapter with its
own call's scales. A release exported for another base GGUF runs untuned. This needs the patched wllama vendored in
`ts-host/vendor/wllama` (LoRA blobs, and a server that never reuses a prompt computed with other adapters; see its
`provenance.json`); a stock wllama build refuses the adapters with `weights-unavailable`.

The managed runtime accepts the fork's server only once the fork is rebased onto a build the runtime check accepts
(currently `>= b10964`).

## Migration

`activeNativeAdapters()` (what model drivers read) now returns the invocation's lexical stack followed by the
explicit `withNativeAdapters` scopes. Programs without a release are unaffected. Code that relied on
`withNativeAdapters` being the only adapter source inside a tuned program now sees the program's stack first; pass
`weights: []` to a task to run it without its release.

## Current coverage

Implemented: scope compilation, per-call resolution with requested/effective identities and coherent fallback,
release store with compare-before-activate and rollback, `natlang tune` (collect, split, train, export, validate,
activate, report, watch), managed llama.cpp and in-page wllama serving with per-request stacks, the joint trainer with nested shared
bases and free LoRA factors, per-node GGUF export, whole-stack PEFT export for vLLM, and soft instruction/skill states. Placement, sweeps, paired evaluation reports and cross-program families are implemented; large sweeps are run by operations. SFT/RL rounds are implemented as library stages: student rollouts on training roots scored by the root
oracle (`tune/rollouts.ts`, `natlang.tune-rollout/v1`), the trainer's whole-graph policy objective over them, and a
resumable round scheduler that keeps the best release by development success (`tune/rounds.ts`); the CLI does not run
them yet. Not yet
implemented: full per-lambda fine-tuning arm, base distillation, a CLI
command for the RL rounds and refitting only dirty nodes (data and training receipts are reused; a semantic change still refits the whole tree). These remain required work in the plan.
