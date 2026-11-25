# Storage policy

Current operating policy, 2026-10-31. `AGENTS.md` and newer owner instructions
control. Earlier inventories and policies remain in Git history; historical path
mentions do not establish a live dependency or require permanent retention.

## Account for the whole machine

Measure allocated bytes and actual filesystem free space across project roots,
shared caches, call stores, Docker, recovery directories and training outputs.
Do not add a parent and its children as independent consumers. Distinguish
apparent size, hardlinked bytes, sparse files and allocated checkpoint reserves.
An open unlinked inode can continue consuming space until its reader finishes.

Use the configured machine paths (`common.paths`) rather than assuming that the
external volume is mounted. Check the actual destination filesystem before each
bulk job. Never write through an absent external mount into the internal root.
New bulk datasets and sealed generation outputs belong on the external volume.
Keep latency-critical active inputs and the current learner's rolling state local
when measured headroom permits. Do not lower a disk floor to force admission.

## Bound training state before spending compute

Use one stable rolling checkpoint slot per logical learner across restarts.
A continuation binds its actual latest earned parent and previous controller
parent under the controller lock; restarting must not create another full-state
output directory. Keep optimizer, RNG, source exposure and transition history.

The native writer reserves the actual future pending inode before optimizer
work, streams into its allocated blocks and publishes atomically. The old
committed slot remains intact until publication. Preserve live
`checkpoint.pt.reserved-pending`, `.reservation.json` and `.reservation.lock`.
These are active storage infrastructure, not aged temporary files.

Include all planned concurrent copies in admission: committed slot, pending
allocation, independent export, retained fallback and staging. Preserve a
completed in-memory update while temporary ENOSPC is resolved. If recovery state
must be saved during shutdown, give the process its measured write time plus
margin; a nominal 120-second grace is insufficient when writes exceed it.

Checkpoint cadence follows measured recovery risk and I/O overhead. It is not
coupled to evaluation and does not require reproducible step milestones. Save
before a necessary continuation or stop, and at the configured recovery interval.
A low loss, saved model or numerical qualification does not qualify task behavior.

## Independent recovery

A verified full-state backup includes optimizer/RNG/exposure, not just weights.
Keep exact runtime, tokenizer/template, recipe, controls, admitted sources and
protected split dependencies recoverable as well. Track the checkpoint and its
dependency closure separately until both are verified. A completed upload or
cached receipt does not replace the required independent readback.

CoreWeave snapshots supplement an export independently downloadable with zero
running sandboxes. A quota cutoff can prevent accessing snapshots. Preserve the
last verified useful recovery state until its replacement is durable. Offload
cold useful state with length/SHA verification before removing its local copy;
protect active readers and maintain logical-path custody where required.

Large files aimed at the external HDD (~60 MB/s) are written to the NVMe first
(`train/checkpoint_offload.staged_write`: GRPO FP32 states and native
checkpoints, exp0_sft checkpoints). The target path is a symlink to the NVMe
copy until a separate ledger-owned mover lands a SHA-verified copy in its
place and deletes the NVMe copy. Retention uses `checkpoint_offload.discard`.
The NVMe holds at most `NATLANG_EXTERNAL_STAGING_MAX_FILES` (2) staged files
above `NATLANG_EXTERNAL_STAGING_FLOOR_GB` (40), else the write goes direct;
`NATLANG_EXTERNAL_STAGING=0` opts out.

## Delete waste; preserve unique useful work

Delete discarded runs and obsolete infrastructure once actual live recovery
and data dependencies have replacements. Retain unique reusable source/training
data, protected evaluation worlds, needed diagnostic evidence and unintegrated
valuable changes. A failed run label alone does not prove its checkpoint is
unneeded. A historical hash mention alone does not prove it is needed.

Remove byte-identical packing intermediates and completed extracted restore-test
trees after independently validating the retained archive. Review orphaned code
trees, integrate or explicitly discard their changes, then remove the trees;
keep compact patches when review is unfinished. Do not archive entire discarded
workspaces indefinitely as a substitute for deciding what is useful.

Evict unused reproducible caches after checking current process and recipe use.
Do not treat base-model caches as earned checkpoints. Do not blanket-prune
another project's data or unique call-store evidence merely because it is large.
The owner explicitly authorized MegaToken state deletion; it completed on
2026-10-31, reclaiming about 88.5 GiB.

Before deletion refresh file identity, active FD/mapping/CWD custody and current
controls. Record exact paths, reasons and bytes; distinguish measured reclamation
from apparent size. Long scans run outside publication/retention locks.

## Operational tools and limits

- `scripts/native_storage_supervision.py` and its two-minute user timer observe
  actual native writer PIDs, rolling slots and pending allocation receipts.
  Observation is distinct from writer enforcement and full recovery proof.
- `scripts/storage_sweep.sh` and `scripts/prune_pytest_scratch.py` prune disposable
  scratch with process custody and freshness checks; never delete the shared
  pytest root or active checkpoint reservation files. The sweep also removes
  unreferenced pins and merged clean worktrees (`scripts/prune_stale_checkouts.py`,
  which walks only the root filesystem and never follows symlinks into the HDD).
  Every step is time-bounded and a timer start while a run is in flight is skipped,
  so a slow disk cannot wedge the timer; the archive and retention units carry
  start timeouts for the same reason.
- `scripts/ei_round_tier.py` (in the sweep) is the standing rule for expert-iteration drivers. Collections of
  policies the builds no longer read, and the bulk of rounds older than the previous one, move to
  `data_hdd/dgx-moved-ei` (verified, symlinked back). Regenerable exports are deleted. Protected policies keep their
  checkpoint and factors on the NVMe. The Drive tier drains `dgx-moved-ei` after half a day.
  Retention (`drop_plan`, default, `--no-drop` / `SWEEP_EI_DROP=0` opt out) deletes first what no consumer
  reads: closed collections' actor exports and native-state blocks (all but the lineage head), trained rounds'
  learner states, chains pickles and superseded exports, and off-lineage unprotected checkpoints (factors stay);
  teacher campaigns' quiet episodes are compacted. The async driver runs the same deletions at each publication and
  collection stop (`expert_iteration.py async-drive`, `--no-retention` opts out).
- `scripts/recall_to_nvme.py LINK --cold GLOB ...` is the inverse of an archive
  move: an archived directory that is a hot input again comes back onto the NVMe at
  its path (hot files copied and SHA-256 verified, cold paths left on the HDD as
  symlinks, `.keep-on-nvme` written, registry records moved to the NVMe path).
  Round builds read the teacher pools this way since 2026-11-23.
- `scripts/drive_tier.py` is the third tier: Google Drive, all under one `natlang` folder (`drive.file` sees only
  what these tools create).
  - Hourly, `natlang-drive-tier.timer` moves cold files of natlang's run archives on the HDD to `natlang/hdd`.
    Eligible: at least 64 MB, untouched for 5 days, not behind any NVMe symlink, not in a data-registry dataset, not
    open.
  - It runs at a 3 MiB/s cap, pauses while an rsync/scp to a Colab host shares the uplink, and is bounded by a
    budget and a timeout; a start is skipped while the previous pass runs.
  - An HDD copy is deleted only after Drive's md5 and size match the bytes read. It leaves `<file>.ondrive.json`, and
    `~/data/drive-tier/manifest.jsonl` (mirrored to `natlang/manifest.jsonl`) records every placement:
    `drive_tier.py restore PATH` brings a file or directory back.
  - Other projects' data (`bgkit*`, hindsight, megatoken, ComfyUI, Vibestudio) is never moved.
  - Colab transfers go through `natlang/blobs/<sha256>` (`pop_rl_actor.ship_file`; `NATLANG_COLAB_DRIVE=0` opts out),
    and VM learners put their state and factors under `natlang/colab/<run>`.
- `scripts/storage_report.py`, `scripts/storage_retention.py` and
  `training/storage-policy.json` provide older inventory/classification tools.
  Their conservative KEEP/PRUNE-ASK labels and historical cadence text are not
  newer owner policy, proof of live dependency, or authorization to delete an
  aged pending file. Review actual dependencies before applying their plans.
- `scripts/archive_to_external.py` and
  `scripts/deduplicate_closed_artifacts.py` support verified offload and immutable
  exact-copy deduplication. A relocation is not deletion of discarded work.
- Run substantial scans and preparation under `scripts/memory_ledger.py`; avoid
  starving useful training or its checkpoint write with cache or backup traffic.

Current cleanup and recovery receipts are under the supervisor's
`data_nvme/lfm3-rl-dgx-20261029/ops` directory. Recheck actual processes, objects
and files; a status document is a point-in-time observation.
