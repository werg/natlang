# Program-adaptation pilot on the full67 parent (2026-11-03)

Review item 8 (`RECIPE_REVIEW_20261103/TUNING_METALEARNING_RL.md`). A bounded machinery qualification of the new
`program_adaptation` recipe stage (`tune.stage --pilot`) on two one-GPU CoreWeave sandboxes. It measures per-root
time, self-teach row yield, untuned headroom and first paired development lift on LFM3, and shakes out lane bugs
before S3 runs at scale. It is **not** a production adapter run and never completes the stage (`--pilot` reports are
`scope: pilot`, `completed: false`); program tunes never advance backbone exposure. Launch status: section "Launch"
below.

## Structural diversity: none exists yet

`tune.structure` over all 54 available tune programs (51 supply programs in 18 nominal families plus the operational
ticket/invoice/inventory programs) finds **one** structural shape (key `950aa7445931e5f6`: one operation, 2-3 NL
functions, a category result, an ordered rule list, no iteration, no services, no Neuralese values; a crisp helper or
not). The stage's exit criterion needs gains over **6** shapes, so S3 cannot complete on the current supply whatever the
tuning results; review item 5 (codebase applications with several operations, iteration, services, effect oracles) is
its prerequisite. The pilot therefore picks the most different programs that exist, varying only secondary features:

| Program | Roots | Why |
| --- | ---: | --- |
| ticket-routing | 128 | reused Luna-verified teacher roots; LFM2.5 reached 17/20 dev (repeatability of the earlier lift) |
| invoice-review | 256 | exact integer helper; largest root set (60 dev roots) |
| inventory-routing | 128 | reservation-aware helper, prose "if ... otherwise" decision rather than an arrow list |
| refund-decision | 128 | catalog supply program with privileged self-teaching references (helper) |
| access-request | 128 | catalog supply program, no helper |

## Inputs (all exist on DGX)

- Parent (supervisor decision 2026-11-03): **full67**, the behavioural parent,
  `/home/werg/data/lfm3-rl-dgx-20261029/ops/bf95-quality-source-zero-sgd-v2-1102/checkpoint.pt`, SHA-256
  `71c3eb45594dd866d2c3d935a5577188e7e0750bf39cb2484ebbe8e4c7d9a83b`, 12,496,833,447 bytes, global step 67. Step 179
  scored 11/16 on dev16 against full67's 12/16, so full67 stays the selected parent. Its private exact recovery is
  `private-recovery-20261101/quality-parent-71c3eb45594dd866-exact-delta-capsule-1102.json` (89.6 MB ZIP delta over
  `bf95…`, itself an 81 MB delta over the raw private base `0c8e75f6…`); sandboxes reconstruct it from those blobs and
  nothing is uploaded from the DGX. The pilot still runs without `--parent-selection`.
- Program set: `/home/werg/data/program-adaptation-20261103/program-set-v1.json`
  (SHA-256 `85d285339c4da3589f5ccc084411b808e0ec264dc0ce379e2b95efa0c5864563`, 54 programs).
- Protection: `training/protected-worlds.json` (SHA-256 `d024a3ed…ec8e` at prep time) and the protected packets'
  `runs/<packet>/cases.jsonl` it names (the RL lineages' custody-packet bundle already ships these).
- Runtime: a code bundle at main including this change (tune.stage, tune concurrency), ts-host built, Node 24.

## Launch

Lineage kind "tune shard" in `scripts/coreweave/rl_cloud.py` (run file
`/mnt/external/natlang-lfm3-cloud-20261103/run.json`, lineages `tune-full67-s0`, `tune-full67-s1`). `rl_cloud.py seed`
stages the five programs once on the DGX into the stage directory with `tune.stage.stage_programs` (both shards and the
DGX merge share `/home/werg/data/program-adaptation-20261103/pilot-full67`), ships the staged programs, the program set
and the protection custody files the code snapshot lacks as exact inputs, and binds the parent as a declared exact base.
The sandbox restores the parent and stage directory at their DGX paths and runs, with the bundle's code snapshot
(ts-host built, root `npm ci`):

    python -m natlang_neuralese.tune.stage --programs .../program-set-v1.json \
      --protection /home/werg/natlang/training/protected-worlds.json \
      --continue-from /home/werg/data/lfm3-rl-dgx-20261029/ops/bf95-quality-source-zero-sgd-v2-1102/checkpoint.pt \
      --out /home/werg/data/program-adaptation-20261103/pilot-full67 --device cuda --shard I/2 \
      --program-names "$PROGRAMS" --pilot --program-hours 24 --concurrency 4

`--shard` takes every second program **in manifest order** (refund-decision, access-request, ticket-routing,
invoice-review, inventory-routing): shard 0 = refund-decision, ticket-routing, inventory-routing (cw3); shard 1 =
access-request, invoice-review (cw4). One GPU, 8 CPU, 64Gi each; sandbox lifetime set to the shard's GPU-hour ceiling
(79 h, 53 h). The stage directory (without `hash-cache.jsonl`, `natlang-config`) is exported every 30 minutes and at
the end; the exporter's hub copy is `ROOT/<lineage>/stage`, and each export manifest carries `tune_progress`
(per-program evaluation arms, tune/timing/failure records, entry, log sizes). The exporter shuts a shard's sandbox
down once its final export (status done or failed) is verified on the DGX.

Merge on DGX by directory union of the two hub copies into the stage directory (disjoint program work; identical
`operation.json`), then the unsharded run writes `report.json` without GPU work (same arguments, `--device cpu`, no
`--shard`).

## Expected cost

Per program: entry arm (untuned development) ~0.3-1 GPU-h at 4 roots in flight; `--rounds` capped at 24 h wall
(`--program-hours`; the joint lane's estimate is 5-12 h/program before batching); release development arm, one
confirmation pass and the distillation rerun ~1-2 GPU-h. Bootstrap (bundle, 5.35 GB delta plus raw base, build)
~1 h per sandbox.

| | Expected | Ceiling |
| --- | ---: | ---: |
| shard 0 (3 programs) | 20-40 GPU-h | 3 x 26 + 1 = 79 GPU-h |
| shard 1 (2 programs, invoice is the largest) | 15-30 GPU-h | 2 x 26 + 1 = 53 GPU-h |
| **total** | **35-70 GPU-h** | **132 GPU-h** |

About 24 GPU-h per program at the ceiling, as the review sized it. Rates and remaining credit are opaque
(AGENTS.md "Recovery and limited compute credit"): track actual sandbox hours, stop each sandbox when its shard
exits.

## What the pilot measures and what it does not

Measures: untuned development headroom per program (the S3 entry signal), self-teach row yield, rounds and their
development numbers, paired tuned-vs-untuned gain, one confirmation per release with gain, wall time per root and per
round (`logs/`, `tunes/<name>.timing.json`), distilled training-world rows. Does not: complete S3 (one shape; pilot
scope), select a production parent, or establish capability lift beyond these paired development roots.
