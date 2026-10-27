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

1. verifies the source against its pointer's SHA-256 and pins a weights-only copy (the optimizer is dropped by the
   trainers' own verified pruning, `train/checkpoint_offload.py`) as `<data dir>/native-models/<sha256>.pt`, so a
   learner rewriting its own file never changes what runs;
2. writes the profile (named after the source, or `--profile NAME`): `native.checkpoint`, `sha256`, the source's
   training checkpoint hash and step, and the checkout's `.venv-neuralese` interpreter;
3. with `--gguf q8_0|bf16|f32`, exports the same weights for llama.cpp (below) as profile `NAME-gguf`;
4. removes the superseded pin once no profile names it. `--default` makes the profile (the GGUF one with `--gguf`)
   the default.

Rerun the same command to move to a newer checkpoint. Every run says when the source has a newer one:
`natlang: lfm-latest has a newer checkpoint (step 40500; profile lfm-latest pins 40128); run natlang setup --native lfm-latest to use it`.

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

## Turning test sessions into training data

`natlang traces good CALL`, `natlang traces bad CALL [--expected JSON]` and `natlang traces fix CALL --expected JSON`
annotate a recorded call and add it as a root example to its program's `tune.examples.jsonl`, grouped per tester
and day so one session's calls stay on one side of every split (docs/TUNING.md "Examples from testing"). Protected
examples are never rewritten. `natlang tune --incremental` then refits on them.

## Tuned programs

A program with an active tune release (`.natlang/tune/active.json`) is served tuned only by the exact base it was
trained on. A release fitted on another checkpoint runs untuned, with a notice saying why; pin that checkpoint as
its own profile to test the tuned program.

## Mellum

The same commands work for Mellum (`mellum-latest`); the GGUF export supports HF backbones only, so Mellum runs on
the native server. Its 12B weights need far more memory than LFM; on the DGX, give the launcher a budget measured
from an actual load before testing alongside training.
