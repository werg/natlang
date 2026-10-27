# Running natlang on our trained models

How to run natlang programs on a trained native checkpoint (LFM2.5, Mellum), for user testing and everyday use.
Everything below is ordinary `natlang` CLI use; nothing needs a hand-written server command or profile.

## One-time setup

```sh
natlang setup --native lfm-latest --gguf q8_0      # pin the newest checkpoint; also export it for llama.cpp
natlang serve --profile lfm-latest                 # optional: keep the model loaded between commands
natlang run applications/evidence --profile lfm-latest
natlang call triage/classify.nl --profile lfm-latest-gguf --inputs ticket.json
```

`SOURCE` for `--native` is a checkpoint file, a JSON pointer to one, or a name from `nativeSources` in
`~/.config/natlang/config.json`:

```json
{ "nativeSources": {
    "lfm-latest": "/home/werg/data/lfm-local-recovery-20261024/current-recovery.json",
    "mellum-latest": "/home/werg/data/mellum-cloud-resume-20261025/current-local-recovery.json" } }
```

A pointer names its file as `local_full_state`, `checkpoint` or `path` (with `sha256`, `bytes`, `global_step`), the
format the training supervisors already maintain. Setup:

1. verifies the source against its pointer's SHA-256 and pins an exact copy (the trainers' verified copy,
   `train/checkpoint_offload.py`) as `<data dir>/native-models/<sha256>.pt` (`NATLANG_NATIVE_MODELS` moves it), so a
   learner rewriting its own file never changes what runs, and the pin keeps the training checkpoint's identity, which
   tuned releases and RL receipts bind;
2. prepares the pin's compact actor view once (`python -m natlang_neuralese.model.native_bundle actor-view`: the exact
   actor tensors without optimizer, RNG and exposure state, verified by a safe reread) under
   `native-models/actor-views/<sha256>/`; the server loads it under the complete checkpoint's identity (measured on
   the LFM tuning base: 13 s and 2.4 GiB host peak instead of about 2 minutes and 8 GiB). The complete pin stays the
   source for tuning and optimizer endpoints; a missing view falls back to it;
3. writes the profile (named after the source, or `--profile NAME`): `native.checkpoint`, `sha256`, `actorView`, the
   source's training checkpoint hash and step, and the checkout's `.venv-neuralese` interpreter;
4. with `--gguf q8_0|bf16|f32` (or when a `NAME-gguf` profile already exists, in its format), exports the same
   weights for llama.cpp (below) as profile `NAME-gguf`;
5. removes the superseded pin (with its view and GGUF export) once no profile names it. `--default` makes the profile (the GGUF one with `--gguf`)
   the default.

Rerun the same command to move to a newer checkpoint. Every run says when the source has a newer one:
`natlang: lfm-latest has a newer checkpoint (step 40500; profile lfm-latest pins 40128); run natlang setup --native lfm-latest to use it`.

## The default model of a fresh install

Without a configured profile, natlang downloads and runs `DEFAULT_MODEL_RELEASE` (`ts-host/src/model-default.ts`) on
managed llama.cpp. It is a GGUF export of one of our native checkpoints, published as a GitHub release asset with its
chat template under `models/templates/` and its serving settings (template arguments, the banned producer marker,
greedy decoding), which apply whenever a profile runs the default model. Publish a newer export with

```sh
node scripts/publish_browser_model.mjs --native-gguf ~/.local/share/natlang/native-models/gguf-<sha12> --release natlang-lfm-step<STEP>
```

which verifies the file against its export manifest, uploads it, and regenerates `model-default.ts`; commit that and
the template. Tuned programs still run on the checkpoint they were tuned on.

## From a TypeScript host

`natlangRuntime()` from `@natlang/node` gives a host the runtime `natlang run` configures, from the same profiles:

```ts
import { natlangRuntime } from '@natlang/node';
import { route } from './triage/route.nl';

const natlang = await natlangRuntime({ programRoot: import.meta.dirname });   // profile: NATLANG_PROFILE or the default
const lane = await natlang.runtime.run(() => route(ticket));
await natlang.close();                                                          // stops the servers it started
```

It runs a tuned program's release on the checkpoint it was tuned on (pinning it when needed), uploads the release to
the native server, starts and stops natlang-owned servers, wires the profile's `escalate` teacher and its examples,
and prints the same one-line notices as the CLI on `log`. Options: `profile`, `provider`/`model`, `weights: false`
(untuned), `escalate: PROFILE | false`, `environment`, `log`, and `runtime` for further runtime options (services,
trace, refinements). `prepare()` starts the model server ahead of the first call. `createNatlangRuntime` remains for
hosts that wire the model themselves.

## The two backends

| Profile | Backend | Use |
| --- | --- | --- |
| `NAME` | native reference server (`natlang_neuralese.serve`), started on demand | Exact native execution: Neuralese production and import, tuned adapters and soft states, RL sampling receipts. |
| `NAME-gguf` | managed llama.cpp with the exported GGUF | Faster ordinary-token execution: no Neuralese production. |

**Native server.** The first model call starts the server for the pinned checkpoint on a port derived from it, or
reuses a server already serving exactly that checkpoint (identity from `/v1/neuralese/info`); the command that
started it stops it. Loading takes about two minutes for LFM, so `natlang serve` keeps it running for later commands
until Ctrl-C. On a shared machine, `native.launcher` and `native.stop` start it through a service manager, e.g. the
DGX memory ledger:

```json
"launcher": ["/usr/bin/python3", "/home/werg/natlang/scripts/memory_ledger.py", "run", "--unit", "{unit}",
             "--budget-gb", "14", "--wait", "900", "--"],
"stop": ["systemctl", "--user", "stop", "{unit}.service"]
```

Other `native` settings: `device`, `dtype`, `memoryGb`, `contextTokens`, `port`, `storeDir`, `args`,
`startTimeoutMs`, and `prefixCacheGb` (below).

**GGUF.** `export/native_gguf.py` merges the learned sparse vocabulary rows (role markers and reasoning boundaries)
into the backbone's embedding and readout, which is exact for ordinary tokens, and exports the tokenizer's own chat
template with the backbone's pinned history-reasoning arguments. The low-rank producer adapter only acts inside
producing spans, which llama.cpp does not run, so the profile bans the producer's opening marker (`logit_bias`) and
stays in ordinary tokens. Its `manifest.json` records the checkpoint, the merged rows and what is not represented.
The profile also sets `temperature: 0`: the native server decodes greedily by default, llama-server samples at 0.8.
`local.server` selects the llama-server executable (for example a CUDA build; the managed default may be a CPU
build, which takes about 16 s per 4.5k-token natlang turn on the DGX).

Measured on LFM step 40128, 16 recorded ticket-routing requests (median 5,586 prompt tokens), greedy:

| Backend | Replies identical to native | Median seconds per call |
| --- | --- | --- |
| native, prefix cache on | (reference) | 4.5 |
| native, prefix cache off | 16/16 | 6.8 |
| llama.cpp f32 GGUF | 16/16 | 3.5 |
| llama.cpp bf16 GGUF | 14/16 | 2.7 |
| llama.cpp q8_0 GGUF | 15/16 | 2.0 |

`scripts/native_replay.py TURNS.jsonl --endpoint A --endpoint B` reproduces such a comparison from recorded natlang
requests (tune collection writes `*.turns.jsonl`).

## Prompt-prefix reuse

Calls of one function share long rendered instructions and differ in a short tail. With `--prefix-cache-gb` (the
managed native profile's default is `prefixCacheGb: 2`), the native server prefills fresh prompts on a fixed token
grid and keeps the state after each block, keyed by the exact prefix and the active adapters; a later call resumes
from its longest stored prefix. The resumed computation is bitwise the one a fresh prefill on the same grid performs
(`tests/neuralese/test_serve_prefix_cache.py`). The grid differs from the server's default single-chunk prefill, so
it is opt-in for servers whose numerics are bound elsewhere (training actors, RL rollouts). `/metrics` reports hits
and reused tokens. llama.cpp keeps its own per-slot prompt cache.

## When the student fails: escalation

```json
"lfm-latest": { "native": { ... }, "escalate": "dgx-qwen" }
```

With `escalate` (or `--escalate PROFILE`; `--escalate none` turns it off), every function runs on the profile's model
as the student tier, and a call it fails (an error, a budget or turn limit) runs again on the named profile, through
the tiered engine (`calls/tiers.ts`, tiers 1 and 0). Each escalation prints one line and is recorded with the call:

```
natlang: classify failed on lfm-latest (model turn limit reached); answered by dgx-qwen. Record the right answer with natlang traces good|fix
```

A wrong but well-typed answer is not a failure; escalation does not catch it. Use only an owner-authorized
teacher profile (AGENTS.md); escalated answers are teacher outputs with their own actor identity.

When the run ends, the input of each escalated root call of a program function joins that program's
`tune.examples.jsonl`, without an expected result (the teacher's answer is not a verified oracle), in a group
`escalations-<day>-<tester>`; an input already listed is left alone (`NATLANG_ESCALATION_EXAMPLES=off`, or
`escalationExamples: false` in `natlangRuntime`, turns this off). The next `natlang tune` collects and verifies them
like any other root.

## Turning test sessions into training data

`natlang traces good CALL`, `natlang traces bad CALL [--expected JSON]` and `natlang traces fix CALL --expected JSON`
annotate a recorded call and add it as a root example to its program's `tune.examples.jsonl`, grouped per tester
and day so one session's calls stay on one side of every split (docs/TUNING.md "Examples from testing"). Protected
examples are never rewritten. `natlang tune --incremental` then refits on them.

Each example records its `producer`: the model that produced the recorded output (`good`), or `tester` (`fix`). Once
tuning is configured, `good` refuses an output from a model that is neither a tuning teacher (`tune.teacher`,
`tune.teachers`, or the student profile's `escalate`) nor one of our native profiles' models, so an unauthorized
model's answers never become training labels; give the answer yourself with `fix --expected JSON`.

## Tuning without configuration

`natlang tune DIR` needs no `tune` block: without a configured student, the student is the native profile natlang
runs (`NATLANG_PROFILE`, else the default; a `-gguf` export stands for its native profile, else the only native
profile). Its pinned complete checkpoint is trained with the checkout's `.venv-neuralese` interpreter, releases are
served through that profile (native backend), and its `escalate` profile is the teacher. The exact base identity is
what the profile's server reports: `natlang tune` starts the server once and the pin index remembers it. A partial
`tune` block (recipe, rounds, execution limits, an explicit teacher) still applies.

When the release a program runs is older than its profile's checkpoint, the run says so:

```
natlang: ticket-routing runs its tuned release b0a4faf8b5e7 on the checkpoint it was tuned on (fe6740d1cda3, step 39495) instead of lfm-latest; --no-weights runs lfm-latest untuned; lfm-latest is at step 40593: natlang tune DIR retunes on it
```

## Status: natlang doctor

`natlang doctor [--profile P] [--program DIR]` adds to the installation check: the profile's native pin (step, a
newer checkpoint behind its source, actor view, known identity, whether its server is running), the escalation
teacher, the tuning setup (or why it is unavailable), and for a program (`--program`, or the current directory when it
has tuning state) its active release, the base it runs on, examples waiting for `natlang tune` (with expected results,
from escalations) and when to retune.

## Tuned programs run on the model they were tuned on

A program's active tune release (`.natlang/tune/active.json`) binds its adapters and soft states to the exact
checkpoint it was tuned on (the release's `base`). `natlang run` and `natlang call` on such a program serve that
checkpoint, whatever the default profile names:

```
natlang: ticket-routing runs its tuned release b0a4faf8b5e7 on the checkpoint it was tuned on (fe6740d1cda3) instead of dgx-qwen; --no-weights runs dgx-qwen untuned
```

The checkpoint is found as a pin, through the program's own tune job plans (which record the student checkpoint),
or behind a `nativeSources` pointer, and pinned on first use; `natlang setup --program DIR` pins it ahead of time.
The native server takes its launch settings (interpreter, launcher, device) from the selected native profile, or the
first configured one. A native profile on another checkpoint is rebound the same way. An explicitly selected
non-native backend (`--profile`, `--provider`, `NATLANG_SERVER`) is kept and runs the program untuned with a notice,
and `--no-weights` runs the selected model untuned for comparison. A release whose base is nowhere available runs
untuned, with a notice naming the missing checkpoint.

Tuned turns cost about the same as untuned ones: forward-only adapted execution builds each adapter stack's effective
weights once and reuses them for every token and call (`model/native_adapters.py` `InferenceReplacementCache`,
bitwise the uncached actor). Rebuilding them per token had made LFM decode 35x slower (2.5 s instead of 72 ms per
token with three stacked adapters). The cache holds about 1 GiB per stack for LFM2.5; `NATLANG_ADAPTER_CACHE_GB`
(default 4, 0 off) bounds it.

## Mellum

The same commands work for Mellum (`mellum-latest`); the GGUF export supports HF backbones only, so Mellum runs on
the native server. Its 12B weights need far more memory than LFM; on the DGX, give the launcher a budget measured
from an actual load before testing alongside training.
