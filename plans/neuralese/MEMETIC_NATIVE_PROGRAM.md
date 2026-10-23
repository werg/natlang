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

```sh
PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python -m natlang_neuralese.rl.memetic_program \
  --program-id my-program --development dev.jsonl --confirmation confirm.jsonl \
  --supervised-graphs verified-train-native-graphs.jsonl \
  --recipe-base training/neuralese/recipes/native-state-production-base-20261015-v1.json \
  --recipe-graph-path /work/memetic/native-guidance-graphs.jsonl \
  --server http://STUDENT:PORT --model-id lfm \
  --author http://SELF-HOSTED-QWEN:PORT --author-model Qwen/Qwen3.6-35B-A3B \
  --out RUN-DIR --generations 3
```

Use the same command and output directory to resume with a larger generation
count. Its input receipt binds the program, split file hashes, exact student
identity and sampling. Each proposed guide is saved before evaluation, so an
interrupted cycle resumes the same candidate. The student writes its guide
through `/v1/neuralese/native-states/write-text`; the returned token IDs,
producer markers, state bytes, tokenizer and checkpoint are retained. The
collector must record the native-state import in an actual executing model turn.
The winner's `binding.json` includes the text, state ID, artifact bytes binding,
student identity and measured development/confirmation outcomes.

Supply `binding.json` to the native program-guidance runtime API. The runtime
uses the exact stored state with the same model identity, and writes the stored
text again when the checkpoint changes. The user-facing binding applies to the
program's calls, not to unrelated programs. The state is imported as positional
KV/hybrid state, with ordinary token embeddings and the learned low-rank
transformation; it is never flattened into the old projected-value channel.

When `--supervised-graphs` is supplied, `native-guidance-graphs.jsonl` is a
direct input for the shared `native_state_sft` trainer. `--recipe-base` also
emits and validates `native-guidance-recipe.json`, with a new SFT stage that
continues the predecessor's earned checkpoint and optimizer/RNG/exposure state
through the normal recipe runner. Supply a trainer-side graph path after the
exact graph bytes are transferred. It puts the guide's
actual producer span and each source invocation's consumer in one graph, with
an explicit import occurrence and one-pass producer-to-consumer gradient path.
The graph source must be verified native train data for the same program and
student tokenizer. The emitted continuation retains the recipe's disjoint
`held_native_graphs` binding. The
normal native graph source validator checks the resulting rows before updates.
The loop does not count graph export as learner exposure: use the trainer's
actual exposure receipt to establish that.

This mechanism is shared by LFM and Mellum (`--model-id`). It has local oracle,
identity, split and tiny-model gradient tests. A live student and self-hosted
teacher evaluation is still needed before claiming a task-quality lift or
publishing a winner for users.
