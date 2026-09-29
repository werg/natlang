# Retired WorkflowEvals import — 2026-09-29

User explicitly released the former evaluation collection into training and confirmed
Apache-2.0 for all four repositories. Collection:
https://huggingface.co/collections/typesafe/workflowevals-6abaf8dcb1e283d9e3d57b6b
Upstream workflow code: https://github.com/typesafe-ai/WorkflowEvals

## Published supply

705 source scenarios, 13,105 question instances. Pinned source revisions and SHA-256
checksums are in `data/external/workflowevals/manifest.json`. Invoice's license is
upstream; the other three are recorded as `user_confirmation_2026-09-29`, not a
claim about their current cards. Original `test` provenance remains intact. Scenario tables are also checksum-verified
and joined before assigning groups. Current teacher holdout pools share zero
source IDs/groups with the new bundle.

| Source | Typed judgments | Directory batches | Total cases |
|---|---:|---:|---:|
| Invoice processing | 1,624 | 0 | 1,624 |
| Customer service | 1,927 | 426 | 2,353 |
| Security incidents | 228 | 50 | 278 |
| Agent trace observability | 415 | 135 | 550 |
| Total | 4,194 | 611 | 4,805 |

All 4,805 reference replays admit. Published native bundle:
`data/teacher/workflowevals/{train.ir,static.results,static.turns}.jsonl` and
`static.manifest.json`: 8,453 approved decisions, zero held/unlinked decisions.
Directory references show actual input files, including all cut-off pages; the
builder reconstructs visible text and checks every source input was exposed.

Both `build_lora_sft.sh` and `create_training_pipeline.py` automatically discover
this manifest. Default pipeline materialization was independently repeated and
matches the published turns byte for byte. Recipe discovery includes its validation
stage. No student training or model calls were used to create this bundle.

## Quality and scope

The source labels are model-generated, not human adjudications. Require both
OpenAI and Anthropic references, answered status, complete finite normalized
probability distributions, unique matching modal labels, probability >=0.95 for
each reference, and exact recomputation of the two-contributor consensus.
Validate typed question schemas and bound original question+state to 22,000
characters. Confidence/agreement remain imperfect evidence of correctness.

8,911 source questions held, retaining original labels and reasons:

- 4,871 context too large;
- 2,723 confidence below threshold;
- 683 missing reference answers;
- 529 reference disagreements;
- 105 tied modal answers.

Question audit: `trajectory-audit.jsonl` (all 13,105 instances); holds:
`rejections.jsonl`. Dedupe exact question/state while preserving alias identities
and scenario groups; conflicting duplicate targets invalidate both candidates.
Customer-service successive `/tN` and security activity variants share upstream activity groups.
Directory batches contain selected questions from one scenario, not unrelated
records. Invoice packets do not fit multiple-question batches under the chosen
bound, so no artificial invoice reducers were created.

Noul becomes boolean, choice returns the original criterion key, score returns
the original **string ordinal index**. Never train a weighted expected score as
an exact categorical answer. Original instructions/state and label distributions
remain attributed. Source gold and reference scripts are outside visible inputs.

These are typed judgments and bounded directory review tasks, not complete policy
workflows. Embedded agent conversations/tool spans are evidence being reviewed,
not successful native trajectories. No complete external workflow or agent
trajectory conversion is claimed. Customer-service refined final decisions in
particular must not be reconstructed from raw modal questions without porting and
checking the actual upstream policy.

The code's retired-evaluation exception names only these four exact revisions and
checks generator, repository, source identity, release authorization, split and
license. They are now training sources and must not be used as held-out evaluation.
Other source holdouts retain their existing protection. Release-boundary audit
rejects changed revisions, validation splits, another repository, missing release
and recovered-source promotion.

**Answer policy change:** verified static replays from these exact released sources
can train their typed answers without an invented chain of thought. Native
materialization requires conversion evidence and `action-notes/1` synthetic
reasoning; rendering masks that reasoning. This is a scoped exception to the
default scripted-direct-answer hold, not a global `--direct-answers` switch.

Independent spot inspection of eight retained customer-service judgments and
selected invoice/agent-trace examples found no clear mismatch; this is not an
exhaustive semantic adjudication. Label distribution is recorded in the pipeline
audit; conservative confidence filtering can bias class coverage. Security retention is 227 false booleans plus one choice, with no positive boolean;
customer/invoice booleans are also negative-heavy. Avoid interpreting this supply
as a balanced classifier benchmark. The 64-case teacher pilot emphasizes question
diversity rather than treating all negatives as equally valuable. Final student
token limits, rendered-pair deduplication, concentration and holdout auditing still
run at the training build. Native replay alone does not establish those properties.

## Operations and further generation

Frozen reference runtime: `runs/workflowevals-20260929/runtime-v2`, v39 baseline with
selective source-admission/materialization/reference overlays. Bonsai runtime-v39
and its active queue-v31 were unchanged. MiniCPM's evaluator watches the exact old
Bonsai PID; check its state before any migration. New teacher collection can use
the published IR or the prepared 64-case pilot, after a safe queue/runtime migration.

Reproduce acquisition in the existing `natlang-train` Python environment with
`huggingface_hub` and `pyarrow`:

```sh
python scripts/acquire_workflowevals.py --out data/external/workflowevals
node ts-host/scripts/inline-curriculum/build-source-backed.mjs \
  --workflow-cache data/external/workflowevals \
  --out NEW_IMMUTABLE_OUTPUT --limit 100000
```

Acquisition now processes Parquet in 16-row batches, sequentially across sources,
inside 1GiB. The initial parallel whole-table conversion was OOM-killed before
publishing a manifest. Completed acquisition has all four sources and checksums.

Published workflow bundles refuse overwrite; select a new output for revisions.
Superseded hidden-evidence drafts live only under `runs/workflowevals-20260929/`,
losslessly gzipped after verifying decompressed hashes; never training inputs.
Audit/replay logs and recipe discovery are in the same run directory.
