# LFM3 whole-graph RL on CoreWeave GPU sandboxes (2026-10-31)

Whole-graph RL (JOLT+ and rejection-sampled self-distillation) moves from the shared DGX to one-GPU CoreWeave
sandboxes spread over the four new accounts (cw1..cw4). The DGX is the hub: seeds, the newest verified state of every
lineage, decisions. Code: `scripts/coreweave/rl_cloud.py` (DGX), `scripts/coreweave/lfm3_rl_lineage.sh` and
`scripts/coreweave/rl_cloud_sandbox.py` (sandbox), job template `scripts/coreweave/jobs/lfm3/rl-lineage.job.json`.
Run file (machine paths, not in the repo): `/home/werg/data/lfm3-rl-cloud/run.json`. Operations log:
`/home/werg/data/lfm3-rl-dgx-20261029/ops/decisions.log` (lines `cloud lfm3-rl-cloud-20261031: ...`).

## Shape

One lineage per GPU sandbox. The actor (natlang SGLang fork behind `serve_native_checkpoint.py --engine sglang`) and
the learner (`natlang_neuralese.rl.native_stage`) alternate in the sandbox exactly as `alt_par.sh` runs one lineage
on DGX: `natlang_neuralese.rl.production run --rounds 1`, then the next round. Nothing about the objective, prompt,
custody or controller changes: the lineage config is the DGX lineage config with paths moved.

Sandbox environments (no image build; the public NGC PyTorch image digest the learners already use):

- learner: the image's Python (NGC torch 2.14 nv26.9, CUDA 13.4) plus the registered exact wheelhouse, exposed at the
  DGX path the configs name (`/home/werg/natlang/.venv-neuralese`, a `--system-site-packages` venv).
- actor: a uv venv at the path `training/neuralese/sglang-fork.json` names, installed from an x86_64 lock that
  `rl_cloud.py bundle` compiles on DGX for the pinned fork commit (`uv pip compile`, public wheels: torch 2.14.1+cu130,
  sglang-kernel 0.4.10 prebuilt for sm90/sm100/sm120, flashinfer 0.7.2 JIT with the image's nvcc), plus the fork's
  Python source (`git archive` of the commit) editable with `--no-deps`. Our fork changes are Python only (CuTe DSL
  conv kernel included), so no kernel build is needed for any plausible GPU (H100, H200, B200, RTX PRO 6000).
- runtime `ROOT/runtime`: `build_runtime.sh` semantics in the sandbox: the bundle's code snapshot, root `npm ci`
  (native addons for x86_64; the DGX copy has an aarch64 zstd addon), ts-host build, and the held/dev task packets
  under `runs/` from the template runtime (custody: `ProtectedWorlds.load`). The actor is rebound to this runtime
  (`rebind_actor.sh` semantics) whenever the lineage's current round is unbound.
- base model from Hugging Face at the pinned revision (bundle URL entries, sha256 verified); small config
  dependencies (recipe, CE policy, geometry, held graphs, protection bundle, task source) pushed through the file API.

The input closure was taken from an strace of `ProtectedWorlds.load` + the controller preflight on DGX
(`/home/werg/data/lfm3-rl-dgx-20261029/cloud/closure/`) plus the learner's explicit arguments.

## State, exports and resume

The DGX link is the constraint: uplink measured 2-6 MB/s, sandbox->DGX 9-11 MB/s, file-API pushes ~1 MB/s, while a
full-state checkpoint is 15.3 GB. So the cloud tier of the hub is the private W&B project `werg/natlang` (storage
account key on DGX only):

- After every round that advanced `trained_rounds`, the sandbox stages an immutable export: `top` (state, config,
  selections, actor tokenizers; tar.zst), one archive per completed round (rollouts, receipts, learner metadata;
  without checkpoints and the regenerable actor export; made once and carried forward) and the parent full-state
  checkpoint (weights, optimizer, RNG, exposure; raw). Blobs are hard links or files under
  `ROOT/.exports/<lineage>/cas/<sha256>`, so later rounds cannot change what an export names.
- `rl_cloud.py export` (DGX ledger service) records the export's intent, issues signed PUT URLs for the missing blobs
  (W&B `createRunFiles` on the blob run), the sandbox PUTs them at cloud bandwidth, the DGX commits them, marks the
  export uploaded in the sandbox (prune may then drop superseded checkpoints there), then GETs every blob to the DGX
  CAS (`ROOT/.cas`), verifying each byte, and materializes `ROOT/<lineage>` on DGX: the hub copy is a resumable
  lineage directory at the same absolute path as in the sandbox.
- `rl_cloud.py launch` always starts from the newest DGX-verified export (or the seed): signed GET URLs for its blobs
  reach the sandbox as the one-time secret `rl-parent`; `rl_cloud_sandbox.py fetch` verifies every byte. A cloud
  PUT or a sandbox-side GET is never counted as the workstation GET.
- Seeds: `rl_cloud.py seed` uploads the parent checkpoint once (content-addressed; all lineages from the same parent
  share the blob) with the generated cloud config and actor tokenizers.

## How the four GPUs are used (decision 2026-10-31)

Four lineages, one per account: `jolt-c1`, `sd-c1`, `jolt-c2`, `sd-c2`, all from the genuine zero-update fresh-init
(`5ff07638...`, the same parent as the DGX lineages), runtime at the current main commit, 32k context.

- Fresh rather than continuing `jolt-r1`/`sd-r1`: those two moved very little (relative parameter delta ~3e-5 after
  five updates, then a few rounds at 5e-6), are stopped mid-round with bound collections on older runtimes, and moving
  their 15 GB state plus in-flight rounds costs 1-3 h of DGX uplink each. The cloud lineages start with every fix
  (32k context, child guidance by identifier with type check) from round 0. `jolt-r1`/`sd-r1` stay intact and
  resumable on DGX (not discarded). The seed mechanism also takes a DGX lineage boundary if continuing one becomes
  worth its upload.
- Two lineages per method rather than one lineage across two GPUs: an actor and a learner in different sandboxes
  would have to move the policy every round across accounts; one lineage per sandbox keeps each lineage
  self-contained and recoverable on its own, and two lineages per method give two independent policies to select
  from by actual task behaviour. The c2 pair differs in its trainer seed and is the first to take the broad task mix
  (below), so the new data is exploited at once while c1 keeps continuity.
- Rounds: the DGX setting (12 rows x group 6, 12 workers, accumulate 4) to start; raise rows/workers from the measured
  actor throughput and CPU use of the first rounds (16 CPU per sandbox).

Resources per sandbox: 1 GPU (no type filter), 16 CPU, 96Gi RAM, 400Gi volume, platform maximum lifetime (720 h).
The RAM figure comes from the DGX per-lineage measurements (collection ~18 GB host, learner ~17 GB RSS plus the 15 GB
checkpoint load); it is replaced by the cgroup `memory.peak` each round logs (`round_done` events). GPU: the DGX
learner peaked at 64 GB CUDA, so 80 GB-class or larger.

## Task source is pluggable

The task source of a lineage is its config's `cases` file: any train-split natlang case JSONL with the same row
contract as today (`version`, `id`, `split: train`, `family`, `semantics` with files/root/oracle, `curriculum.rl`
level/pair groups for the curriculum). A new mix is a run-file change (`lineages.<name>.config.cases` and the file in
`extras`), shipped with the next bundle; the controller hot-reloads `cases` at a round boundary (the selection of a
bound round is sealed). Scoring is the runtime collector's: oracle kinds beyond today's (normalized match with
aliases, SQL execution, unit tests, reasoning-gym verifiers) are runtime code at the run's code commit, so they arrive
with the next bundle/runtime, not with the job. Notes for the mix builder: rows must be train split only and outside
the protection union (the learner's custody check loads `ProtectedWorlds`); the curriculum's breadth window takes
consecutive rows in file order, so interleave families in the file. `initial_train_world` binds the cases file's
sha256 and must be dropped (`null` override) or rebound for a new file.

## Commands (DGX)

    PY=/home/werg/.local/share/natlang/cw-venv/bin/python; RUN=/home/werg/data/lfm3-rl-cloud/run.json
    $PY scripts/coreweave/rl_cloud.py seed $RUN jolt-c1            # once per lineage (blob shared)
    $PY scripts/coreweave/rl_cloud.py bundle $RUN                  # after code or task-source changes
    $PY scripts/coreweave/rl_cloud.py launch $RUN jolt-c1 --dry-run
    NATLANG_CW_UPLOAD_WORKERS=4 $PY scripts/coreweave/rl_cloud.py launch $RUN jolt-c1
    python3 scripts/memory_ledger.py run --unit rl-cloud-export --budget-gb 4 --class service --priority medium \
      --wait 300 --detach -- $PY scripts/coreweave/rl_cloud.py export $RUN
    $PY scripts/coreweave/rl_cloud.py status $RUN
    $PY scripts/coreweave/rl_cloud.py stop $RUN jolt-c1

## Open items

- GPU parity checks of the fork commit `8f2fd5894` (lfm3, producer, adapter, decide, actor capture) have not run
  (the DGX GPU is in the decision eval). The smoke exercises the actor path end to end only.
- W&B blob retention: the exporter runs `rl_cloud.py prune` each pass. It keeps every lineage's newest DGX-verified
  export and anything newer (in flight), the seed blobs (fresh-init parent) and the current run inputs, and never
  deletes a blob leased through `scripts/coreweave/retention_leases.py` (namespace `.cloud/standalone-*`; a launch
  leases its parent and inputs as consumer `launch:<job>`, released when its sandbox ends or exports from a newer
  parent). Intent is durable in `.cloud/rl-blob-retirement.json` before any delete; absence is confirmed against the
  full file listing (W&B `files(names=...)` echoes absent names, and deleting an absent file is an opaque error).
- The exporter GETs whole checkpoints (15 GB, ~25 min each at the DGX downlink); with four lineages it verifies the
  newest export per lineage and skips superseded ones rather than queueing every round.

## Blobs travel as parts

Every blob (seed checkpoint, export checkpoint, archives) is stored as 1 GiB parts, each its own private run file
`blobs/<sha256>.pNNNN` with its own sha256; a failed PUT or GET repeats one part. The first single-object 9.5 GB seed
PUT over the DGX uplink (~3 MB/s) failed after a full pass and restarted from zero.

## Status 2026-10-31

- GPU smoke BLOCKED: all four new accounts refuse GPU sandboxes (`CWSANDBOX_GPU_NOT_ALLOWED`, "this organization is
  not entitled to create GPU sandboxes"; not a quota error). The owner/CoreWeave must enable GPU entitlement.
- `rl_cloud_autopilot.py` (ledger service `rl-cloud-autopilot-N`) probes one account per hour with a minimal GPU
  create (stopped at once); on the first entitled account it runs the GPU smoke (`smoke-jolt`, one round), and when
  that export is verified on DGX it seeds and launches `jolt-c1`, `sd-c1`, `jolt-c2`, `sd-c2` across the entitled
  accounts, with a coord note. A failed smoke stops it with an urgent coord alert.
- The continuation lineages `jolt-r1c`/`sd-r1c` (DGX lineages' newest trained parents, fresh controller history) are
  in the run file but launch only on the supervisor's decision (RL eval lift); their 15 GB parents upload as parts.
- CPU pipeline check (cw2/cw3 CPU sandboxes, 4 CPU/16Gi): bundle push 296 MB in ~4 min (~1.2 MB/s, 4 workers);
  HF base and inputs fetched and verified; ts-host npm ci + build 44 s (after fixing both stale package-lock.json
  files on main, which made `npm ci` refuse); SGLang fork venv from the x86_64 lock + Rust radix-tree core 237 s;
  runtime 27 s; `ProtectedWorlds.load` + learner/controller imports on NGC torch 2.14.0a0 and the fork's LFM3 imports
  pass on x86_64. Fixed on the way: bootstrap apt install of zstd hung 11 min (now bounded; Python zstandard
  fallback), the fork's editable install needs cargo (rustup minimal, `SGLANG_BUILD_RUST_EXTS=mem_cache`).
  Export round trip verified end to end: seed upload DGX->W&B 9.5 GB in 52 min (3.05 MB/s, 9 parts); sandbox GET of
  the parent from W&B 30-76 MB/s per part (209 s for parent fetch); staged export -> signed PUT -> commit -> DGX GET
  at ~12 MB/s -> hub copy `ROOT/smoke-jolt` with the checkpoint sha256 `5ff07638...` re-verified. Fixed on the way:
  exports recorded the sandbox's resolved `/work/mirror` paths (now logical paths).

## Broad task source and runtime (2026-10-31, later)

All cloud lineages use `natlang-rl-broad-external-20261031` (33,047 train cases, 23 external sources plus a curriculum
slice; families already interleaved row by row) with 16 rows and 16 workers per round, `initial_train_world` dropped,
and template runtime `runtime-9fbae2582` (broad source 7bbf8e055 + Neuralese bonus added after outcome-only centering
338c5c2ba). The cases file (293 MB) and its db/table assets (613 MB) are run `cloud_inputs`: each is
a deterministic tar.zst stored once as a content-addressed W&B blob (sha256-checked on fetch, DGX copy authoritative),
restored at the same absolute path by the sandbox before the controller starts; the bundle no longer carries them. The DGX lineages switch to the same source and runtime at their next unbound round (cutover.sh);
`lfm3-rl-broad-watch` reports per-source success of each lineage's first broad round with failure samples.
