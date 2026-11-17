# Vibestudio fixtures through canonical program IR and backbone SFT

Vibestudio suites are a frontend for `natlang.program/2`, not a separate training
format. The full model source, real entrypoint, JSON inputs, host fixture source,
selected case, expected requirements and source lineage remain durable in the IR.
The standalone Vibestudio episode-to-chains implementation was removed; Git is its
history. No measured corpus had used that path.

## Shared mechanism inventory

| Component | Responsibility |
| --- | --- |
| Vibestudio `scripts/natlang-teacher.mjs` | Export original source and complete split inventory to ProgramRecords, launch the ordinary collector, perform isolated semantic review and bind its evidence. |
| `ts-host/src/teacher/program.ts`, `evaluation-fixture.ts` | Neutral canonical fixture descriptor and execution frontend; real NL entry or authored TS entry with explicitly identified NL semantic anchor. Host fixtures/scorers stay outside model-owned files. |
| `scripts/neuralese_data/teacher_gen.py` | Explicit repeatable source pools, held exclusion via `grpo.load_tasks`, root mode, resumable teacher collection. |
| `ts-host/src/teacher/collector.ts` | Ordinary hosted/native provider execution, request/reply journals, `programRow` and `natlang.teacher_trajectory.native/1` containing `task.program_ir`. Hosted teachers need no native sampling probabilities. |
| `rl/task_rollouts.py` | Ordinary collector scoring, source-custody binding and `natlang.rollout/1`; real invocation metadata follows the canonical turns. |
| `scripts/neuralese_data/teacher_admission.py` | Validate supplemental source/review evidence and invoke the existing collector replay. It does not render tokens or produce a second trajectory protocol. |
| `scripts/neuralese_data/sft_sources.py` | Shared extraction, exact collector replay, held/protected exclusions, rendering, loss masking, selection and function target-mass census. |
| `scripts/neuralese_data/exp0_sft.py` | Existing FP32-master/BF16 learner consuming the shared builder's `chains.pkl`. |
| `register_corpora.py`, `sync_training_corpora.py` | Existing immutable registry/manifests and data transport. Collection, admission, staging and learner exposure remain separate. |

No engine protocol, scoring, sampling, native-state or cache change is introduced.
The source adapter emits ordinary canonical teacher/rollout records. Tokenization
uses `sft_sources.TemplateRenderer` (tokenizer-only, no model); training uses the
existing `exp0_sft.chains_of` representation. Managed llama-server, wllama, Python
native serving and SGLang continue to consume their existing serving protocols
(`docs/ENGINE_PARITY.md`).

## Canonical collection

The host first exports the **full** source inventory (train, validation and test)
and a selected training pool. Each ProgramRecord records explicit split, stable
`source_namespace`, source ids/groups/revisions and the real fixture entry.
Source inputs include the selected fixture case/world as well as function JSON
arguments: service-only functions legitimately have empty argument records.

Run from the experiment pin through the memory ledger on the DGX:

```
python scripts/neuralese_data/teacher_gen.py --work POOL_WORK \
  --pool SOURCE_IR.jsonl --held-manifest HELD_V3/manifest.json --root-mode authored \
  --teacher '{"provider":"openai-codex","model_id":"gpt-6-luna","reasoning_effort":"low","slots":2}'
```

`--pool` is repeatable and replaces the default GRPO pools. Default old campaigns
retain `--root-mode student`. An evaluation fixture's authored mode preserves its
real source entry and records `authored_fixture_entry`; it never fabricates a
reference eval or an NL forwarding wrapper. Tasks and source-pool receipts bind
resumes; changing explicit pool content, held selection or root mode needs a new
work directory. Existing default student campaigns predate this receipt and can
still resume.

Ordinary durable output is `manifest.jsonl`, per-task `cases.jsonl` and queues,
canonical native teacher results/journals, and scored `rollouts.jsonl`. The native
teacher result contains the original ProgramRecord, actual requests/replies,
traces, execution graph and fixture outcome evidence.

## Supplemental admission

For a fixture task the teacher-pool manifest entry declares
`admission: {path, sha256}` for a `natlang.teacher-admission/1` sidecar. The generic
checker is shared by host diagnostics and SFT extraction:

```
python scripts/neuralese_data/teacher_admission.py --pool POOL_WORK
```

The sidecar binds canonical cases, rollouts, full original ProgramRecord split
inventory, exact teacher identity, and per-sample native result file hashes and
collector program-IR digests. Each sample retains the actual suite score/gates,
full fixture outcome evidence, collector code provenance, exact replay verdict
and an isolated semantic review with its full request/reply and reviewer identity.
Semantic review is fallible admission evidence, never labelled exact gold and
never supervised as an executor target. It may use the same model in an isolated
conversation; this does not alter the stock collector's distinct-model judge
policy.

The generic replay helper uses `sft_sources.ReplayServer/replay_episode` and emits
receipt drafts, not a new trajectory representation:

```
python scripts/neuralese_data/teacher_admission.py replay --pool POOL_WORK \
  --runtime-repo PIN --work FRESH_REPLAY_WORK --out admission-drafts.json
```

The host adds full-inventory bindings and retained semantic reviews to these
drafts, records rejected samples separately and attaches the sidecar hash to the
ordinary teacher-pool manifest. Replay resume identity excludes these admission
links and uses git provenance rather than ephemeral checkout paths.

Admission excludes nontraining splits, protected source-group siblings and
identical input/world scenarios crossing splits, including renamed groups and
new program revisions. It requires full fixture success, all gates, no failed
tool actions, a bound exact replay and accepted semantic review. Review failures
remain collected evidence, not training targets. File hashes and replay receipts
are trusted collector-code attestations, not signatures or adversarial proof.

## Shared SFT and the next experiment

```
python scripts/neuralese_data/sft_sources.py extract --work SFT_WORK --teacher POOL_WORK
python scripts/neuralese_data/sft_sources.py replay --work SFT_WORK --runtime-repo PIN
python scripts/neuralese_data/sft_sources.py build --work SFT_WORK --out BUILD \
  --tokenizer TOKENIZER --held-manifest HELD_V3/manifest.json
```

Use the normal builder's source-mix, base-rate and budget flags for the experiment
recipe; there is one renderer, loss objective and trainer. Fixture sources without
validated sidecars remain unadmitted. Admission evidence is checked again before
build, and current-runtime exact replay must pass. Held-v3/v2 exclusions apply via
the existing shared builder. The teacher's own actions and answers carry loss;
prompts, earlier history, tool observations and an inserted empty think prefix
carry zero weight under the existing masking mechanism.

`build.json.target_mass` reports actual target and weighted target tokens by
suite/function/root-or-child and totals for root, child and unknown. Names,
definition ids and parentage come from actual invocation ledgers. Missing identity
is explicit unknown; forwarding-wrapper mass is measured, not silently reweighted.

Register the actual source collection and built data through the existing corpus
registry and immutable manifests, record project-generated/provider/template
license provenance, then transfer to the actual DGX/Pop owner. Registration is not
learner exposure. Adopt the corpus at the next experiment pin; never change data
inside a measured live run. Judge task success on paired held-v3 evaluation and
separate unseen Vibestudio scenarios, not training loss or fixture case counts.
