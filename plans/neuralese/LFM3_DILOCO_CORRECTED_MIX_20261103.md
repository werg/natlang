# LFM3 corrected-mix joint SFT: one cloud learner now, DiLoCo growth next (2026-11-03)

Fallow register items [8]/[9]. Owner 2026-11-03: production LFM3 training runs mainly on cloud GPU sandboxes; the
DGX is an optional extra worker, rollout/prep machine and recovery hub.

## Step 1: one cloud learner (ready)

The corrected-mix window `ops/corrected-mix-full67-w1-1103` (prepared by `ops/prepare-corrected-mix-window-1103.py`,
never started on DGX) becomes window 0 of the cloud SFT-window lineage `corrected-mix` through
`rl_cloud` `"prepared_control": true` (e0c503361): exact control relocated, host-only controls dropped, input
closure shipped, full67 as the declared ZIP-delta chain 0c8e75f6 -> bf95 -> 71c3eb45. `sft_window.py next` from the
prior control would have rebuilt an uncorrected window, and a started DGX control is refused (one executor).
Launch: `ops/corrected-mix-cloud-launch-1103.sh` (machine path) = plan, seed (ledger), bundle, dry run, launch,
exporter. Run file `/mnt/external/natlang-lfm3-cloud-corrected-20261103/run.json` (runtime v41 archive
`ops/runtime41-corrected-mix.tar.zst`).

Fit: the corrected mix's largest graph is 335,563 KV slots (the 524,782-slot pair is gone; 130 graphs exceed 262k,
1,492 exceed 131k). The 524,782-slot graph measured 70.9 GB CUDA peak allocated (12.3 GB resident + 58.6 GB
incremental) with file-backed checkpoint-input offload (25.8 GB written), so ~50 GB is expected at 335k: it fits an
80 GB H100 or 96 GB RTX PRO 6000. The learner's capacity qualification measures the actual card before SGD; if it
refused, the option is to route the few >262k graphs to a DGX worker, not to drop them.

## Step 2: DiLoCo growth (shape below)

## Shape

- **Workers.** Worker 0 is the step-1 cloud learner; workers 1..K are further one-GPU cloud learners balanced across
  cw1..cw4, as many as the accounts grant right now; the DGX joins only when useful (its uplink and ~1/3-1/4 speed
  make it a stale contributor)
  (`SandboxResourceExhausted` is a shape problem: `cwctl.create_with_shape_backoff` lowers CPU/RAM stepwise to the
  lineage's `minimum_resources`, 4 CPU / 48Gi, always exactly one GPU, and records the granted shape).
- **Variable availability = sessions.** Shards are `data_shard(row_id, workers)`, fixed for one coordinator
  session. A session starts from the newest earned DGX window boundary with the cohort available then; when a GPU
  appears or disappears, the session ends at the next window boundary: `native_diloco export` writes one ordinary
  continuation checkpoint (coordinator globals, every learner's exposure merged onto the parent with
  `merged_exposure`, learner 0 = DGX inner optimizer/RNG), and the next session reshards for the new cohort
  (`diloco_new_outer` manifest, now any cohort size, not only 4). A slow or absent worker therefore never strands
  its shard: remaining quota is redistributed at the next session.
- **Outer optimizer.** Committed decision #8: plain averaging (outer lr 1.0, momentum 0) for rounds 0-9, then
  Nesterov momentum 0.9 / outer lr 0.7 (`--outer-schedule`), global EMA on (coordinator default 0.99).
  Merge weights are token mass x tokens/step / (1 + staleness) (`merge_pseudogradients`).
- **Sync interval.** `local_steps` 64 = `refresh_after_updates` 64: one outer round per corrected-recipe window, so
  every window boundary is a drained DiLoCo boundary. Wire BF16: 1.7B trainable params = 3.4 GB per direction per
  round. DGX uplink 2-6 MB/s puts its upload at 10-28 min against a ~4.4 h DGX round (64 updates at 14.4/h); fragments
  stream during compute. Cloud learners are 3-5x faster, so they complete ~3 rounds per DGX round: `--max-staleness
  4`, quorum = number of cloud workers present (DGX contributions merge stale, down-weighted), `--grace-seconds 600`.
- **Windows and LR.** Each worker runs the exact corrected-mix control (accumulate 16 = the scored-graph anchor,
  backbone 2e-5, rows 1e-3, 64-update linear warmup from the parent step, QAT default-adopted, EMA both). The inner
  optimizer is the single-GPU optimizer per worker; with averaging the merged update is the mean of K+1 local
  trajectories, so no LR rescaling. Warmup is per worker from the same parent step.
- **Exposure/quota custody.** Each worker's quotas are `active_exposure_quotas(..., shard=(worker, workers))`
  (stable hash, disjoint shards, appends land in a stable shard); focus promotion and packed geometry apply within the
  shard. Export sums each worker's gained visits onto the parent ledger (negative gains refuse). Diagnostic smoke
  exposure never enters production (isolated namespace below).
- **Coordinator placement.** Dedicated CPU-only sandbox (8 CPU / 48Gi, floor 2 CPU / 32Gi; never cohosted), primary in
  cw3, replica (`diloco_replica_*`) in another account than the primary. Learners reach it through its public URL with
  the share token and our bearer token, delivered as secret files (`rl_cloud.diloco_secrets`), never argv.
- **Recovery and exports.** DGX worker 0 checkpoints locally every window (its rolling slot is the primary full-state
  recovery). The coordinator root is staged by the stage-shard exporter every `NATLANG_CW_STAGE_SECONDS` (default
  30 min) to private W&B and the DGX hub; `native_diloco export-global --coordinator-url` pulls BF16 global/EMA
  model-only views to the DGX each round for dev evaluation. A killed cloud trainer restarts from its own rolling slot
  (`cloud_role learner --restarts`, the distributed rolling-resume identity); a lost cloud sandbox relaunches from its
  newest DGX-verified stage export; a lost coordinator restarts a new session from DGX worker 0's full state plus the
  last exported globals.
- **Throughput and spend.** DGX alone: ~230 graphs/h. A cloud learner (RTX PRO 6000 Blackwell or H100) is estimated at
  3-5x DGX per graph (to be replaced by the smoke's measured update time). Four cloud learners plus DGX: roughly
  2,900-4,800 graphs/h before communication (~5% at 64 local steps). Spend: K x 24 GPU-h/day (96 GPU-h/day at K=4);
  at an opaque <= $500 per account that is days, not weeks: track `cwctl.py accounts status` gpu_h per account and
  stop learners that are not merging.

## Smoke dropped (owner 2026-11-03: production is its own smoke)

The isolated smoke (`/mnt/external/natlang-diloco-smoke-20261103`) verified, before it was dropped: the full67
ZIP-delta chain reconstruction in a sandbox, code/bundle install, `native_diloco init` from full67 in a CPU sandbox
(88 fragments, current checkpoint format) and the coordinator's public share-token gateway. Its GPU learner was never
granted (RUNNER_CAPACITY_EXHAUSTED on cw3/cw4 at every shape, 15:05-15:25), so no inner step or outer sync ran.
Coordinator sandbox 589130e3 (cw3) was stopped at 15:25 with zero merges (nothing earned to export). The first
production DiLoCo rounds are the qualification: watch actual merges, exports and recovery.

## Production status

First start (7512cb2f, cw3, 16:25) restored parent and inputs but the controller failed in 3 s:
`inherited_execution_admission` verifies the selected publication named inside the source-controller admission, a
file the closure never shipped (it did not follow nested references). Fix 7422c599e: the closure ships bound
(path + sha256) JSON records reachable from the control's execution-transition/admission and task_selection_history
files (depth 4, <= 16 MiB each; source manifests not expanded): 1,401 files, 8.29 GB. The GPU was re-acquired
first as a holder (`rl_cloud launch --parent-later`: bundle installed, waits for `rl-parent`), sandbox 8120d5a5
(cw3, 1 GPU / 8 CPU / 64Gi), then re-seeded and handed the parent with `deliver-parent` (17:18). Exporter unit
`rl-cloud-export-corrected-2`.

Holder path: `launch --parent-later` + `deliver-parent` is the minimal in-place handover (acquire capacity first, ship
inputs later). General takeover of a running lower-priority sandbox (`launch --reuse`) is not built yet.

DiLoCo placement (owner 2026-11-03, AGENTS 16a944ad1): coordinator primary cohosted on a learner sandbox, replica on
a learner in another account, started only once two GPU learners train; measure its RAM/CPU beside the learner and
exercise one primary failover in the first rounds. `cloud_role` currently starts the coordinator on its own CPU
sandbox; cohosting is the change for step 2.

Starts 3 and 4 failed on further unshipped verifier inputs: the `/home/werg` symlink changed `resolve()` (fixed by
real mirror directories in bootstrap, b7ca76ea5), then the sealed-collection blocks directory (b50133206). After the
fourth failure the sandbox was released, and the closure is now qualified locally before any relaunch. The CPU
namespace replay is a docker container (bwrap is blocked by apparmor userns restrictions on DGX) that shows only the
shipped closure at its DGX paths plus the code bundle. It found the protection-root packet cases (db2a060d6), the
portable dev evidence and predecessor source shards named only inside nested manifests. The static policy cannot reach
those, so the closure now includes the observed reads of a strace'd CPU controller start of the exact window
(`sft_window closure --observed-reads`, lineage `observed_reads`, 05bfa26b6), including the content-addressed
row-index cache entries. Runtime v42 misses the index for all 51 D32 acquired-session sources (~2-2.5 h of CPU
indexing at start); the trace builds those entries on the DGX so that the GPU does not repeat the work.

The traced start then found a real runtime defect: v42 (with 4cc424565's per-payload reader cap) could not start this
control anywhere. Historical zero-SGD adoption verification rederived the quota policy with the new
`payload_reader_cap` field and failed exactly. 90571dcbb rederives pre-cap records under their own law. The fix ships
as runtime v43 (`runtime-native-corrected-mix-v43-1103`), with the control re-prepared as
`ops/corrected-mix-full67-w1-v43-1103`; it is identical to the v42 control apart from runtime bindings. Historical
checkpoints that are DGX symlinks stay symlinks (ccfe8593f). The namespace replay of the v43 closure (1,762 files,
11.4 GB) reached the learner's model load (GPU allocation) cleanly. Coverage stops at that point: reads the learner
makes after allocation are not traced. The production holder (cw1 b4caeaa0, H100, 8 CPU/64Gi) was granted at 20:31
CET and the v43b seed delivered.

Further root causes before the first update:
- A traced-read closure shipped DGX arm64 `/lib` loader files (3bf919288; seeds refuse system paths).
- Two held-packet registrations grew the protected union, which made historical qualifications fail. The digest is
  now treated as provenance (4bac2dd1e, 9f9aa5c39), shipped as runtime v44.
- The package's packed geometry carried the 1102 parser identity. It was recomputed under v44 with 0 changed slots,
  as control `corrected-mix-full67-w1-v44b-1103`.
- A stale size+mtime digest cache hit closure files restored with mtime 0 (158e076cd).
- A lineage minimum above its own request lost the bankmeta GPU (f749b0777).

Production's first update committed at 22:03Z: global step 68 on cw1 b4caeaa0 (H100, 8 CPU/64Gi). The step took 57
graphs in 262 s (about 780 graphs/h, ~3.4x DGX), with 50.6 GB peak allocated. Backbone LR is 2e-5 and rows 1e-3,
under a 1/64 warmup multiplier. Rule from this incident: prepared package artifacts get versioned successors and
are never rewritten in place.

## Open for step 2

Controller windows plus DiLoCo: each worker's window control needs `trainer_parameters.diloco` (URL from the secret
file, worker id from `NATLANG_DILOCO_WORKER_ID`) and a heterogeneous role contract (`train/diloco/role_contract.py`)
because each worker's `--out` differs; window continuation already registers drained supervised roles
(`rl/diloco_contract.register_supervised_stage`). The stage-shard cloud roles (`train/diloco/cloud_role.py`) run the
bare trainer and serve the smoke; the window integration is the remaining wiring before step 2.
