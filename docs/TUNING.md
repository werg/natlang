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

For long prompts with short action replies, set `tune.recipe.prompt_weight` to `0` to train only the reply.
The default `0.1` also supervises prompt tokens at one-tenth weight; repeated instructions and schemas can then
dominate the objective. Adjust this on development outcomes. Recipe changes in `--rounds` create a fresh journal;
set `tune.rounds.initialFitParent` to the saved build directory to continue its adapters and optimizer histories.

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
services, types) changed.

`natlang tune ./my-program --incremental` refits only what an edit dirtied, against the active release:

- A **candidate** trains the edited lambda leaves (and new bindings) against every other node pinned at its exact
  released residual, warm-started from the released leaf. Only rows whose stack contains a trained node are fitted.
  A dirty shared ancestor (a folder or the program root trained on the old semantics) stays pinned at its older
  version, and the release receipts record which older binding semantics it was trained on (`stalePins`).
- A **shared refit** then trains the dirty shared ancestors and every residual fit against them, the rest pinned;
  it is planned against the release the candidate pinned and seeded with the candidate's leaves. If it does not
  complete, the next `--incremental` run finishes it rather than reporting the program fresh.
  `tune.incremental.sharedEdits: "refit"` skips the candidate.
- A **full** joint refit runs after `tune.incremental.fullRefitEvery` incremental builds (default 5), on `--full`,
  or when the recipe changed (pinned residuals of another rank cannot compose).

Warm-started nodes continue their optimizer moments only when the released optimizer state binds the exact artifact
and algorithm; otherwise their history starts new and the receipt says so. Families with their learned gates and the
soft states of untouched bindings carry over unchanged. `--watch` makes the same incremental builds. `--report`
shows the active release's build kind and any stale pinned parents. The report prints the time from invoking
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
5. Transfer: every published family is also applied zero-shot (gate 1 on every definition, nothing refit) to the
   named programs that are not its members and to held programs (`--held ./prog-c,./prog-d`), and measured paired
   on their development roots. Held programs never enter discovery or fitting, and transfer releases are stored but
   never activated.

Every activated program release is recorded per base (`<store>/programs.json`). With `tune.families.every: N`, a
successful `natlang tune` runs families over all recorded programs of the base after N new releases since the last
family run; `natlang tune --families` without program arguments uses the recorded programs.

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

A `repair` round (on-policy correction, plan §4.4; for example `"schedule": ["sft", "repair", "rl"]`) runs the best
release so far on the training roots that have an `expected` result. For each root it gets wrong, the teacher takes
over the student's own run at a turn boundary: the student's earlier turns are replayed call by call, so the program
re-executes in the same world, and the teacher answers from that turn on. Boundaries are tried from the latest
backwards (`repairsPerRoot`, default 1; `repairBoundaries`, default all). A repair is admitted only when every replayed
turn and the boundary observed exactly what the student observed and the repaired run meets the root oracle. Only the
teacher's turns train, together with the earlier teacher rows; the student's mistaken prefix is context at zero weight.
Each correction row records the student checkpoint and release, the boundary call's requested/effective stack, the
boundary, the round, the teacher and the verification evidence; the whole repaired trajectory is kept in
`.natlang/tune/rounds/<data>/round-<n>.repairs.jsonl`. Repair rounds work on every backend.

`natlang tune ./my-program --rollback` reactivates the previous coherent release. `--no-weights` on `run`/`call`
runs the base model.

## Programs that use host services

`natlang tune` from the CLI runs programs without host services. An SDK host that tunes a service-using program
passes `services: example => ({ ... })` in the tune options: fresh services per root invocation, a test or sandbox
world rather than production systems, because collection, evaluation and rollouts call them. Every call made during
collection and on-policy repair runs is journaled exactly (service, method, arguments, result or error). A repair
replays the student's journal: it is admitted only when every service call of the repaired run was answered from
that journal, so a repair never acts on the world a second time; a continuation that needs a call the student's
run never made is rejected as leaving the captured world.

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
| llama.cpp fork (`--lora-per-sequence`, `--native-state`) | Same | Yes with `tune.softStates: true` (LFM2): exact native state spans | Yes, per-sequence scales, verified bitwise isolation |
| Native reference server (`neuraleseServer: true` profile) | Exact native coefficient artifacts | Yes | Per request |
| vLLM (`vllm: true` profile) | One whole-stack PEFT LoRA per distinct effective stack, loaded at activation | No: soft-state bindings run degraded | Yes (multi-LoRA batching) |
| wllama in the page (vendored build, `loadBrowserRelease`) | Per-node GGUF LoRA, per-request scales | No: soft-state bindings run degraded | No (serialized per stack) |

**Soft states on the managed llama.cpp fork.** A soft state is native KV/hybrid state, not text: per position, the
attention keys (after the key norm, before RoPE) and values, and LFM2's short-convolution gated inputs. With
`tune.softStates: true` on the `managed-local` backend, the build trains soft states and exports each as a llama.cpp
native state span (`exports/<state>.state.gguf`, bound to the base GGUF). The managed session passes them with
`--native-state`, and requests carry `{"type": "native_state", "id"}` content parts where the instruction or skill
text was. The fork imports the stored state exactly for virtual tokens between the consume markers. On the real
LFM2.5-350M it matches the reference engine as closely as plain text does (max |Δlogit| 0.043 vs 0.075, top-1 1.0;
`scripts/llama_state_span_parity.py`). A server without `--native-state` (the upstream runtime) refuses to start with
soft states (`weights-unavailable`); in `run`/`call` such releases run their soft-state bindings degraded. Mellum
soft states are not exportable yet.

In a page, `loadBrowserRelease(releaseUrl, { baseGgufSha256 })` fetches `natlang.adapters.json` and its GGUF exports,
checks each file's sha256 (program residuals and family versions; files are cached in the origin-private file system by content hash and re-verified on every load, `removeCachedModelFiles` drops replaced ones), and returns the adapters for `loadBrowserLocalModel(source, { loraAdapters })` and the
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
implemented: full per-lambda fine-tuning arm. Incremental builds (candidate, shared refit, periodic full refit) are
implemented; the native-backend soft-state carry-over of untouched bindings has no test yet.

Invocation budgets can be set with `tune.execution`, for example
`{ "maxTurns": 12, "maxTokens": 4096, "maxSeconds": 120, "maxFailureRepairs": 3 }`.
They apply consistently to collection, evaluation, repairs, distillation and RL. Evaluation and RL receipts record
them; a different budget gets a separate rounds journal and rollout groups. Each limit bounds an invocation, so a
program with several calls may use more turns or time overall. Choose budgets for the program's workload and
compare base and tuned execution under the same settings.
