# Native memetic learning for repeated program invocations

`natlang_neuralese.rl.memetic_program` is the current task-success search and
training-data bridge for one program. It replaces the old decision-readout
memetic experiment and its flat soft-skill/weight-adapter individuals in this
workflow. The policy that rewrites guidance is a natural-language function run
by the self-hosted author model; the rest is exact collection and verification.

Prepare two disjoint JSONL packets with at least two invocations apiece. Every
row must have `split: "train"`, the same `program_id`, a distinct `id`, and the
normal authored root task contract. Confirmation invocations are **never**
shown to the author or exported as SFT graphs. If a program has protected test
rows, keep them in a third packet and use the existing held evaluator after
the candidate is bound. Development and confirmation use the same student,
sampler and seeds for seed/candidate comparison. Outcomes come from the normal
task oracle; infrastructure and incomplete scores stop acceptance.

For existing production packets that lack `program_id`, use the batch selector
and worker. It derives a stable ID from the exact natlang root, files and
declared service contract. Private service implementation data remains an
invocation world binding. Rows retain the source file SHA-256, original row
SHA-256 and index. The selector makes global source/world connected components
before splitting, so related cases cannot appear in development for one
program and confirmation for another. It writes a `batch.json` manifest with
ready jobs and explicit pending groups. The worker iterates every ready
program, journals successes and failures, and resumes each program from its
own candidate/fitness state on restart. With a base recipe, it verifies that
all winning graphs share the exact student, combines them, and emits one
`training.json` and one executable shared-trainer continuation recipe.

```sh
PYTHONPATH=training/neuralese .venv-neuralese/bin/python -m natlang_neuralese.rl.memetic_batch \
  --source /path/to/train-cases.jsonl --out /work/memetic-batch \
  --select-only
PYTHONPATH=training/neuralese .venv-neuralese/bin/python -m natlang_neuralese.rl.memetic_batch \
  --source /path/to/train-cases.jsonl --out /work/memetic-batch \
  --server http://STUDENT:PORT --model-id lfm \
  --author http://SELF-HOSTED-QWEN:PORT --author-model Qwen/Qwen3.6-35B-A3B \
  --author-provenance verified-qwen36-author.json \
  --recipe-base /work/current-native-production-recipe.json --generations 3
```

The current 36-case failed-training packet has five ready programs spanning
30 invocations: one 8-case program, three 6-case programs and one 4-case
program. Remaining groups are recorded as pending because they lack enough
independent cases for two training and two confirmation invocations.

```sh
PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python -m natlang_neuralese.rl.memetic_program \
  --program-id my-program --development dev.jsonl --confirmation confirm.jsonl \
  --recipe-base training/neuralese/recipes/native-state-production-base-20261015-v1.json \
  --foundation-id native_full_sft \
  --recipe-graph-path /work/memetic/native-guidance-graphs.jsonl \
  --server http://STUDENT:PORT --model-id lfm \
  --author http://SELF-HOSTED-QWEN:PORT --author-model Qwen/Qwen3.6-35B-A3B \
  --author-provenance verified-qwen36-author.json \
  --out RUN-DIR --generations 3
```

The author receipt has schema `natlang.memetic-author-provenance/1`, the exact
`model_id`, `hosting: "self-hosted-coreweave"`, the exact `endpoint` and
`sandbox_id`, and SHA-256 fields `checkpoint_sha256`,
`tokenizer_fingerprint`, `runtime_sha256`, and `source_receipt_sha256`. Prepare it
from the sandbox's verified teacher serving records. The loop binds its bytes
and never treats a model-card name or an endpoint as checkpoint proof.

Use the same command and output directory to resume with a larger generation
count. Its input receipt binds the program, split file hashes, exact student
identity and sampling. Each proposed guide is saved before evaluation, so an
interrupted cycle resumes the same candidate. The student writes its guide
through `/v1/neuralese/native-states/write-text`; the returned token IDs,
producer markers, state bytes, tokenizer and checkpoint are retained. The
collector must record the native-state import in every evaluated invocation.
The candidate text is retained as provenance and for re-encoding, but the
scored executor receives only the native state; text alone cannot explain an
accepted gain.
The winner's `binding.json` includes the text, state ID, artifact bytes binding,
student identity and measured development/confirmation outcomes.

Supply `binding.json` to the native program-guidance runtime API. The runtime
uses the exact stored state with the same model identity, and writes the stored
text again when the checkpoint changes. The user-facing binding applies to the
program's calls, not to unrelated programs. The state is imported as positional
KV/hybrid state, with ordinary token embeddings and the learned low-rank
transformation; it is never flattened into the old projected-value channel.

When `--recipe-base` is supplied, successful development rollouts with exact
whole-episode native captures become `native-guidance-graphs.jsonl`, a direct
input for the shared `native_state_sft` trainer. Each graph retains every
captured model turn and the guidance import's recorded position. The external
guidance state is resolved to the actual `write-text` producer span; episodes
with other unresolved external states are rejected. `--recipe-base` also
emits and validates `native-guidance-recipe.json`, with a new SFT stage that
is scheduled by RAW supervision before the declared foundation stage while
retaining all original recipe stages. It follows the normal earned-checkpoint
continuation mechanism; supply a trainer-side graph path after the exact graph
bytes are transferred. Each row puts the guide's actual producer and captured
consumers in one forward graph, preserving gradients. The emitted recipe
retains the disjoint `held_native_graphs` binding. The
normal native graph source validator checks the resulting rows before updates.
The loop does not count graph export as learner exposure: use the trainer's
actual exposure receipt to establish that.

This mechanism is shared by LFM and Mellum (`--model-id`). It has local oracle,
identity, split and tiny-model gradient tests. A live student and self-hosted
teacher evaluation is still needed before claiming a task-quality lift or
publishing a winner for users.
