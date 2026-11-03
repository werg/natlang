# Program-adaptation pilot on the step-179 parent (prepared 2026-11-03, not launched)

Review item 8 (`RECIPE_REVIEW_20261103/TUNING_METALEARNING_RL.md`). A bounded machinery qualification of the new
`program_adaptation` recipe stage (`tune.stage --pilot`) on two one-GPU CoreWeave sandboxes. It measures per-root
time, self-teach row yield, untuned headroom and first paired development lift on LFM3, and shakes out lane bugs
before S3 runs at scale. It is **not** a production adapter run and never completes the stage (`--pilot` reports are
`scope: pilot`, `completed: false`); program tunes never advance backbone exposure. No sandbox has been created.

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

- Parent: step 179, `/home/werg/data/lfm3-rl-dgx-20261029/earned84-joint-r38-a4-window64-1103/learner/checkpoint.pt`,
  SHA-256 `239c8ca0d4f49a3c4414580c587890fd036c07bf9076df87fbf354ef487ca6dd`, 15,403,153,798 bytes. Its ZIP delta over
  `bf95…` (5.35 GB) already exists in the cloud run store
  (`/mnt/external/natlang-lfm3-cloud-20261103/.cloud/deltas/bf95bec416a1bb06-239c8ca0…`); sandboxes reconstruct it
  from the private raw base `0c8e75f6…` exactly as the joint-SFT lineage does. No behaviour selection exists for step
  179 (no matched dev16 run); the pilot runs without `--parent-selection`, which is why it is a pilot.
- Program set: `/home/werg/data/program-adaptation-20261103/program-set-v1.json`
  (SHA-256 `85d285339c4da3589f5ccc084411b808e0ec264dc0ce379e2b95efa0c5864563`, 54 programs).
- Protection: `training/protected-worlds.json` (SHA-256 `d024a3ed…ec8e` at prep time) and the protected packets'
  `runs/<packet>/cases.jsonl` it names (the RL lineages' custody-packet bundle already ships these).
- Runtime: a code bundle at main including this change (tune.stage, tune concurrency), ts-host built, Node 24.

## Launch plan

Accounts: **cw3** (shard 0) and **cw4** (shard 1), as the review assigns (cw1/cw2 go to RL actors). Resources: exactly
one GPU, 8 CPU, 64Gi RAM (the native server plus four concurrent roots; learner-class host memory for the adapter
fits), 720 h lifetime, explicit stop after the shard finishes. Both accounts last refused GPU sandboxes
(`CWSANDBOX_GPU_NOT_ALLOWED`, `LFM3_RL_CLOUD_20261031.md`); the one-time entitlement check is not to be re-polled, so
the first launch attempt is the entitlement check.

Each sandbox, after the standard bootstrap (bundle, HF base, parent reconstruction, ts-host build):

    PROGRAMS='["ticket-routing","invoice-review","inventory-routing","refund-decision","access-request"]'
    python -m natlang_neuralese.tune.stage --pilot --shard I/2 \
      --programs /home/werg/data/program-adaptation-20261103/program-set-v1.json \
      --protection /home/werg/natlang/training/protected-worlds.json \
      --continue-from /work/parent/checkpoint.pt --out /home/werg/data/program-adaptation-20261103/pilot-179 \
      --device cuda --program-names "$PROGRAMS" --program-hours 24 --concurrency 4

Shard 0 (cw3): ticket-routing, inventory-routing, access-request. Shard 1 (cw4): invoice-review, refund-decision.
Export: the stage directory without `hash-cache.jsonl` and `natlang-config` (each shard's `programs/<name>`, `tunes/`,
`distill/programs/`, `logs/`, `structure.json`, `entry.json`, `operation.json`). Merge on DGX by directory union (the
shards' program sets are disjoint and `operation.json` is identical), then the unsharded run writes `report.json`
without GPU work:

    python -m natlang_neuralese.tune.stage --pilot --programs ... --protection ... --continue-from <DGX parent> \
      --out /home/werg/data/program-adaptation-20261103/pilot-179 --device cpu --program-names "$PROGRAMS" \
      --program-hours 24 --concurrency 4

(The DGX parent path differs from the sandbox's: bind the same absolute path on the sandbox, or the operation binding
refuses the merge. `rl_cloud` already restores the parent at its DGX path.)

Launch mechanics: `scripts/coreweave/rl_cloud.py` has joint-SFT and RL lineage kinds only. The pilot needs a third,
small lineage kind ("tune shard": parent from `checkpoint_bases`, the program set and protection as `cloud_inputs`,
command above, export of the stage directory). It is left to the strand currently fixing the cloud launch path, to
avoid conflicting edits there; until it exists the pilot is prepared, not launchable.

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
