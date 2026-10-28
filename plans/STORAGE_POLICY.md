# Storage policy (DGX; Pop may opt in)

## Active scratch retention (2026-10-28)

The 30-minute `natlang-storage-sweep` timer prunes completed pytest run
directories separately. A live test previously protected the entire shared
`/tmp/pytest-of-werg` root, allowing 62 GiB of completed fixtures to accumulate.
`scripts/prune_pytest_scratch.py` now keeps the newest three runs,
`pytest-current`, live or unreadable PID locks, open/mapped files and process
working directories. Each candidate must also have no recent writes anywhere
in its tree (three hours by default); process custody is refreshed before
deletion. The shared pytest root is never deleted. Other disposable scratch
also uses the shared FD/mapping/CWD inventory. These rules preserve training
data and checkpoints outside disposable test scratch.

## Standing owner order — 2026-10-11

Discarded approaches and runs are deleted, including checkpoints, optimizer
states, exports, diagnostic outputs, frozen runtimes, obsolete recipes, registry
entries, manifests and run tracking. They are not archived or relocated as a
substitute for deletion. Historical references do not protect discarded work.
Keep current lineage dependencies and reusable training/source/evaluation data;
check live references before deleting. Deduplicate exact immutable copies, prune
superseded conversions/review packets/caches, and retain rejected outputs only
for an assigned repair, unresolved bug, or actual failure/DPO training consumer.
Clean at monitoring cycles and after large runs. Coordinate mirror cleanup with
the machine owner. See `AGENTS.md`; this order overrides older preservation and
PRUNE-ASK requirements below for explicitly discarded work.

Owner, 2026-10-09: "we will also be running out of space on the external drive if we keep on going. Let's consider
our checkpoint frequency and other space-saving measures." and "we may need to delete stuff that isn't absolutely
necessary over time."

Machine-readable form: `training/storage-policy.json`. Tools: `scripts/storage_report.py` (measure),
`scripts/storage_retention.py` (classify, delete), `scripts/archive_to_external.py` (NVMe → HDD, separate agent),
`training/neuralese/natlang_neuralese/train/checkpoint_policy.py` (how trainers write checkpoints),
`training/neuralese/natlang_neuralese/maple/ternary_pack.py` (compact ternary milestones).

## 1. Tiers

Every file ≥ 1 GiB under `runs/`, `~/data` and `/mnt/external/natlang-development-data` (archived copies are
classified as the NVMe file they replaced) gets one tier. When in doubt, PRUNE-ASK.

| Tier | What | Deleted |
|---|---|---|
| KEEP | Files pinned by a corpus or artifact manifest (`training/neuralese_corpora.json`, `training/neuralese_artifacts.json`) and their archived copies; any path named by run scripts, recipes, registries, plans, certificates, receipts or qualification files (lineage parents, qualified checkpoints, teacher outputs referenced by builders); logs, receipts, certificates, labels (Clef/human); `~/data/models`, `~/data/bird-sqlite`, `~/natlang-data-nvme`. | Never |
| ACTIVE | Open by a process; in a run a process works in; named by an active memory-ledger claim or an archiver exclude (running Mellum conversion and its teacher dumps); in a run changed within 3 days. | Never (re-evaluated next pass) |
| PRUNE-AUTO | Clearly regenerable: intermediate step checkpoints of a finished run (the latest step, `*best*`, `*final*`, `*export*` stay); checkpoints of aborted/stopped runs (`aborted`, `guard-stopped`, `floor-stopped`, `failed`) after 7 days unless a plan or script names the run directory; leftover partial writes (`*.pending`, `*.tmp`, `*.partial`) after 2 days; dangling docker images and build cache. | By `storage_retention.py --apply`, with a deletion manifest |
| PRUNE-ASK | Every other large file: latest checkpoints of finished runs no one references, optimizer state of finished runs, unregistered teacher dumps and datasets, exports. | Only when the owner lists the path in `~/.config/natlang/storage-approved.txt` |

Protected docker objects (`keep.docker` in `training/storage-policy.json`) are active infrastructure even when
exited, and are never removed by a cleanup, by hand or by script: the teacher container `natlang-qwen36-nvfp4-server`
and its image `vllm-node` (ID `666307b2…`) are started on demand by `scripts/ensure_teacher.sh`. A 2026-10-12 cleanup
took "exited five days ago, no ledger reference" for "retired" and deleted both; `storage_retention.py` now reports
protected objects that are missing, and `scripts/teacher/run-qwen36.sh` recreates the teacher from its run spec.

Deletion re-checks each file right before unlinking (size and mtime unchanged since the scan, not open) and removes
the NVMe symlink that pointed to a deleted archived copy. The manifest (`~/.local/state/natlang/storage/deletions.jsonl`)
keeps path, bytes, category, tier, reason, run and time.

## 2. Checkpoint policy (all trainers)

The largest consumer is checkpoints: the Mellum QAT conversion writes 47 GB per checkpoint (BF16 latents 23 GB +
Lion momentum 23 GB), every ~15 min at one per 100 updates.

- **Declared step points**, not a wall clock (owner 2026-10-10): evaluations and full-state writes happen only at
  step points each stage declares, so they are reproducible and comparable across runs, resumes and machines.
  `eval_every`: about 10-20 evenly spaced evaluation points per stage, sized from the measured step time, the last at
  the stage end (`steps` a multiple of it), where the stage gate runs; no added evaluations (no baseline, no
  evaluation tied to a write). The one exception is a gate that compares against the starting weights (the text
  warm-up's forgetting check, the recurrence trainer's initial report): its step-0 reference runs on a fresh lineage
  only. `checkpoint_every`: a multiple of `eval_every` sized to about every 3 h (owner: "checkpoint every few hours"),
  so every full-state write has its own point's evaluation. The full state is also written at the stage end and
  whenever the process is asked to stop (SIGTERM/SIGINT/SIGUSR1 → latents and optimizer at the next step boundary,
  without an evaluation, then exit). Every stop path gives the job time to write it (the memory ledger's per-unit stop
  grace, plans/MEMORY_ADMISSION.md), so the points only bound crash loss. Recipes from raw-recurrence-v6 on declare
  their points this way (`train/loop.py` `check_declared_points`, checked by tests/neuralese/test_recipe_inheritance.py;
  older recipes are frozen records); the loader refuses the retired `checkpoint_minutes`/`eval_minutes`. Trainers
  always evaluate the stage end (the gate), so a CLI run whose `steps` is not a multiple gets that one evaluation.
  Writes go from device tensors one storage at a time: no host copy of the model (unified memory). Measured: conversion
  v3's 46.8 GB slot took ~250 s (~190 MB/s to NVMe).
- **One rolling slot**, written atomically and durably (pending → fsync → rename → directory fsync).
  If the disk cannot hold the replacement beside the previous slot, report insufficient space and preserve the
  previous checkpoint. Failed partial writes are removed; pruning is separate from checkpoint writing.
- **Best = the best among the full-state writes**, published as `best-checkpoint.pt` by a hard link to the written
  slot (`checkpoint_policy.publish_best`): never a write or an evaluation of its own. The rolling slot's next atomic
  replacement leaves the linked inode, so a best costs one retained state on disk until a better write replaces it.
- **End of run: preserve full resumability by default.** Export final weights alongside the full optimizer,
  schedule and RNG checkpoint. Call `finalize(weights, resumable_state=latest_full_state)` to preserve the exact
  final step, or save the final full state before exporting. Explicit `drop_resumable=True` is available only for
  an owner-authorized weights-only milestone after a successful final export; it is not a space-pressure fallback.
  Pop requires optimizer continuation. DGX-specific disposal decisions must be explicit in its owned wiring.
- Implementation: `train/checkpoint_policy.py` (`CheckpointPolicy`). Wired into `maple/qat_convert.py` on branch
  `storage/qat-convert-checkpoint-policy` (d74a821e), to merge after the running conversion v2 ends. Wiring into
  `text_warmup.py` / `trajectories.py` is proposed to their owners (Pop: C1 checkpoint/export seam; architecture
  session: C2/C3), not done here.

## 2a. Checkpoint offload to the external disk; weights-only non-latest checkpoints (owner 2026-10-13)

Owner: "we should move active run checkpoints to external on an ongoing basis (async from write)", and "prune
optimizer state from every checkpoint except the latest resume checkpoint". The NVMe holds only what a resume needs:
the rolling `checkpoint.pt` (with its optimizer) plus the trainer's `.checkpoint-space.reserve`. Implementation:
`training/neuralese/natlang_neuralese/train/checkpoint_offload.py`, one hook for every trainer
(`trajectory_state.atomic_checkpoint`, `checkpoint_policy.publish_best`, `text_warmup.retain_best_checkpoint`).

- **Writes are unchanged** (NVMe, pending file, fsync, atomic rename). After a successful write the file goes to one
  background worker per process; the copy runs in a child process at nice 19 and the idle I/O class, 8 MiB chunks,
  page cache dropped every 512 MiB (unified memory). A failure is an event (stdout and
  `<out>/checkpoint-offload.jsonl`), never a training failure; the file is retried at the next write.
- **Mirror**: `<data_hdd>/natlang-active-lineage/<path relative to ~/natlang/runs>` (`repo/…`, `home/…` for other
  trees), only when that root is on a mounted filesystem other than the source's (a container without the
  `/mnt/external` mount gets no mirror). Copy to `<mirror>.offload.part`, fsync, check the opened source did not change,
  re-read and compare size and SHA-256, rename, fsync the directory, then `<mirror>.offload.json` (source identity,
  bytes, SHA-256). Atomic publication of a successor at the source pathname does
  not invalidate a byte-for-byte copy of the intact opened file. Its receipt
  names that older file identity; current-mirror checks and archive link swaps
  still require the identity of the local pathname to match.
- **Roles**: `checkpoint.pt` (latest) is mirrored byte for byte as a disaster copy and stays on the NVMe. `best-*.pt`,
  `final-*.pt`, `checkpoint-*.pt`, `step-*.pt` (archive) move to the mirror and the NVMe file is atomically replaced
  by a symlink to it (link written beside it, `os.replace`), so every consumer path keeps working. Files listed in a
  corpus manifest (`training/corpus-manifests`, verified byte for byte, symlinks refused) are never touched.
- **Weights-only non-latest checkpoints** (owner 2026-10-13): an archived full state is stored on the mirror without
  its optimizer (every top-level `*optimizer*` entry dropped; `optimizer_pruned: true` and `optimizer_pruned_keys` in
  the state and in the `.offload.json` record), re-read and compared value by value (bitwise) with the source before
  the rename; receipts beside the file that pin its SHA-256 (`best-checkpoint.json` `files`) are rebound to the
  weights-only file (marked `optimizer_pruned`) before the link swap. A Mellum direct state: 47 GB → about 23 GB.
  Pruned: bests of `CheckpointPolicy` writers (direct SFT, QAT conversion; `prune_best=True`) and, in sweeps and the
  one-shot, full states of `PRUNABLE_SCHEMAS` (`natlang.neuralese-direct/1`). Not pruned (consumer audit
  2026-10-13): the text warm-up's best (`core_text_warmup` continues from it, `recipe.py`), the causal bootstrap's
  best (stage `artifact_sha256` and foundation-certificate pins), trajectory bests, every `checkpoint.pt` (direct,
  AR-fixup and trajectory handoffs continue from the predecessor's `checkpoint.pt`; it is never pruned, and
  `PredecessorArchive` links it to the verified mirror when one exists instead of copying it again). Every
  continuation loader (`direct_sft`, `text_warmup`, `trajectories`, `causal_bootstrap` `--continue-from`) refuses a
  pruned state with "weights-only checkpoint … continue from the stage's latest checkpoint.pt". A Maple/Mellum
  direct export proof (`schema_proof.checkpoint_sha256`) is made against the file as it is at export time.
- **Resume**: when the NVMe `checkpoint.pt` is missing or truncated (cut-off zip) and the run's
  `checkpoint-policy.json` records a mirror whose record matches, `CheckpointPolicy` points the slot at the mirror (an
  atomic symlink; the next write replaces it with a regular NVMe file).
- **Crash consistency**: the NVMe path is always the old file or a link to a mirror renamed into place and made
  durable first; never a dangling link. Stale `*.offload.part` (30 min) and `*.offload-link` (1 min) leftovers are
  removed at the next sweep; another copier's `<mirror>.part` is never touched (that file is skipped).
- **Settings** (recorded under `offload` in each run's `checkpoint-policy.json`): `NATLANG_CHECKPOINT_OFFLOAD=0`
  disables, `NATLANG_CHECKPOINT_MIRROR` (root), `NATLANG_CHECKPOINT_OFFLOAD_MIN_GB` (1), `NATLANG_CHECKPOINT_OFFLOAD_LATEST=0`,
  `NATLANG_CHECKPOINT_PRUNE_OPTIMIZER=0`, `NATLANG_CHECKPOINT_OFFLOAD_HDD_FLOOR_GB` (50). A `.keep-on-nvme` marker
  keeps a directory's files local.
- **Existing runs**: `scripts/offload_checkpoints.py [RUN_DIR …]` (dry run by default; `--apply`) does the same for
  already-written checkpoints; defaults: the Mellum direct lineage and Pop's mirrored LFM v8 runs.
- **Deleting a run** deletes its mirror under `natlang-active-lineage/` too (standing order above).

## 3. Compact milestones for ternary models

A qualified Maple/Mellum milestone that QAT will not resume from keeps its deployed form only: `qat_export`
(ternary values in the HF layout) packed by `maple/ternary_pack.py` into 2-bit codes + BF16 row scales, ~8x smaller
than BF16 for the converted tensors (Mellum: ~23 GB of latents → ~3 GB; embedding/head/router/norms stay raw).
Unpacking is verified bit-exact per tensor and loads with `load_maple` like the export. Latents + optimizer are kept
only for the latest resumable slot of a run that may continue.

## 4. Other measures

- NVMe holds working data; finished large files move to the HDD hourly (archiver, verified copy + symlink).
- Regenerable caches (pip, uv, triton, torch inductor, …) are cleared by the archiver; HF snapshots are archived.
- Teacher-logit dumps whose generating command is recorded are regenerable; they are PRUNE-ASK until a builder
  records the command in a receipt (then they can move to PRUNE-AUTO).
- Corpora: only registered snapshots are training inputs; unregistered large `.jsonl` under finished runs are
  PRUNE-ASK.

## 5. Schedule

- Weekly (Sun 04:00) user timer `natlang-storage-retention.timer` writes the dry-run report
  (`~/.local/state/natlang/storage/report-latest.md`). `--apply` stays manual until the owner enables it on the
  timer after reviewing the first pass.
- Owner approval of PRUNE-ASK items: add paths (one per line) to `~/.config/natlang/storage-approved.txt`; the next
  `--apply` deletes them if they are still PRUNE-ASK.

## 6. First measurement and pass (2026-10-10 00:30)

Disks: NVMe `/` 916 GB (90% used, archiver running); `/mnt/external` 3.6 TB (83%, 644 GB free).

Scanned roots `runs/` (181 GB real files; 685 entries are symlinks into the HDD from the 10-02 relocation),
`~/data` (187 GB) and the archive (54 GB so far). By category:

| Root | Largest categories (GB) |
|---|---|
| `runs/` | registered corpora 120.3, checkpoints 26.7, teacher dumps 18.2, weights snapshots 11.1, exports 3.7 |
| `~/data` | checkpoints 90.9 (Mellum conversion slot 43.6), registered corpora 38.2, data-other 31.6 (BIRD SQLite), exports 10.9, unregistered corpora 10.1 |
| `archive/` | teacher dumps 22.8, checkpoints 13.6, registered corpora 7.5, weights snapshots 7.1 |

Tiers (files ≥ 1 GiB): KEEP 91 files / 208 GB, ACTIVE 20 / 54 GB, PRUNE-ASK 16 / 43 GB, PRUNE-AUTO 0. Runs keep one
rolling `checkpoint.pt` (+ `best-checkpoint.pt`), not step series, so there are no intermediate checkpoints to
prune; the first `--apply` deleted nothing (docker: image prune 0 B, build cache 67 B).

Awaiting owner approval (PRUNE-ASK, top items):

| GB | Path |
|---|---|
| 7.6 ×3 | `archive/data/neuralese-converted/v13-2026100{4,5,5}-v{5,6,7}/teacher.neuralese.jsonl` (superseded by registered v8) |
| 2.8, 2.4, 1.6, 1.6, 1.3 | `~/data/maple-slices/*.gguf` (slice/export checks) |
| 1.6 | `runs/maple-nested-20261005/exec-n2b/24x64-q.gguf` |
| 1.3 | `runs/maple-foundation-20261005/c12-v1-stopped/embedding_distillation/checkpoint.pt` |
| 25.4 | docker image `80394cb1e815` (pre-py-spy natlang-neuralese base) still referenced by exited experiment containers (maple recurrence, probes); removing those containers frees it. Never the teacher container. |
| 9.0 | unused docker volumes (content not inspected) |

Biggest structural consumers are KEEP by rule: registered corpora (~166 GB across roots) and the running Mellum
conversion. The levers are therefore (1) checkpoint cadence and end-of-run handling (§2), (2) compact ternary
milestones (§3), (3) owner approval of superseded corpus versions and exports, and (4) the full HDD report
(`~/.local/state/natlang/storage/report-latest.md`, weekly) for the rest of `/mnt/external`.

## Exact duplicates in closed runs

For reviewed byte-identical historical payloads on the same filesystem, use
`scripts/deduplicate_closed_artifacts.py PLAN --sha256 HASH --receipt RECEIPT --execute`.
The plan lists each canonical keeper, duplicate path, byte length and SHA-256.
The implementation verifies both files, rejects live references and preserves
both paths using an atomic hardlink replacement. Shared payloads become read-only;
future revisions must use new files rather than modifying a shared inode.
Run-specific logs, manifests and receipts remain separate. Execution journals
record physical bytes reclaimed. This changes storage, not dataset membership,
source grouping, splits or training admission. Unique old artifacts still require
manifest-driven verified offload or an explicit reproducibility review.

On Pop, run the storage-report/retention tools with `python3.11`; their shared
configuration reader uses Python's `tomllib`. Override scan roots for the local
machine rather than using the DGX external-drive roots.

## 7. Checkpoint cleanup (2026-10-10 ~12:00)

Owner: "let's make sure to delete all the excessive checkpoints we have accumulated" (standing deletion authorization).
Inventory of every weight file over 300 MB on NVMe and the external drive (189 files, 334 GB); every file named in a
corpus manifest or the artifact registry (123 files, 161.5 GB) was kept. Deleted, with the exact path list in the DGX
deletion log (`owner-approved-deletions.sh` §8): 74.2 GB on NVMe, 65.5 GB on the external drive, plus 20 NVMe
symlinks into the archive.

- Mellum conversion v3 (stopped by decision): the 47 GB rolling slot and the original `best-weights.pt` (the registered
  copy `mellum21-convert-v3-latents-20261010` is what the QAT foundation reads).
- Failed v2's heads (`runs/mellum-qualify-20261010/heads/heads.pt`).
- Retired Maple line: member-warmup smoke, native text warm-up v10, QAT latent smoke (incl. a partial `.pending`),
  raw-recurrence v1–v4 and ablation m0, `maple-foundation-20261005/heads-c23.pt`, the 8x16 member GGUFs. Kept:
  `maple-nested-20261005/n2b-v1/nested-state.pt` (named by recipe latent-sketch-consumer-maple-v2).
- S3 pilot lineage: 20261003, 20261003-v2, resume-d-v1, 20261005-v3, the AdamW/Muon optimizer comparison and three
  stream-Muon smokes. Kept: resume-ef-v1 (S3_FULL_RUN_HANDOFF) and 20261005-v4 (S3_FULL_RUN_PLAN parent).
- Unregistered local-recurrence experiments: fit-v5…v9, resume-proof-v1, wide-chain-v1…v3, and the external
  `inputs/port-checkpoint.pt` (byte-identical to the registered NVMe copy).

Kept on purpose: base models, teacher outputs (teacher-v7*, teacher-v3 on NVMe as the fast copy of the registered
artifact), `models/decision/clef-flash` (Clef scripts), the Phase F export (HANDOVER), the browser-CI LFM GGUFs,
the fork follow-ups' GGUFs, `ling-lora` (unreferenced, pre-Neuralese; owner call), and Pop-owned `runs/luna-*`.

## Deleted runs (2026-10-11)

Deleted on DGX under the standing owner order on discarded work (AGENTS.md); exact paths in `runs/DELETED-DISCARDED-20261011.txt` on DGX. References above are historical.

- `maple-foundation-20261005`: deleted 2026-10-11.
- `maple-nested-20261005`: deleted 2026-10-11.

## DGX cleanup, round 3 (2026-10-12 ~15:45)

NVMe fell to ~100 GB free (the disk-guard pause line). Deleted under the standing order, exact paths and reasons in
`/home/werg/.claude/jobs/9c849f00/tmp/storage-20261012T1545.sh` on DGX; df went from 108 GB to 116 GB free (+18 GB, concurrent jobs also write):
the five `run-v4/runtime-superseded-*` copies, `neuralese-direct-mellum-20261011/{run-v1,run-v2,smoke-1..3}`,
41 old code snapshots (kept only `natlang-5d212530`, used by live v4; nothing else references a snapshot),
`~/data/render-smoke`, orphaned `~/data/tmp-port-calls.*` partials, `~/data/{tmp-ckbench,v17-audit}`, the two
retired-Maple stream render caches `~/data/maple-sft/{,v13-r2/}maple-v13.sft.jsonl.cache` (registered
`maple-sft-v13-r2` lists only the jsonl, manifest and rejected file), and ten exited diag/test/probe containers.
No registry entries were removed. The paused round-2 verified offload of Pop's registered inactive model
candidates (`.local/retained-model-candidates-20261010-v1`, ~27 GB left) was re-queued now that v4 has loaded.
It completed at 16:07 (unit `storage-cold-model-offload-resume-20261012`, all 45 rows, SHA-256-verified moves to
`/mnt/external/natlang-development-data/archive/.local/` with symlinks at the logical paths; receipt
`~/.local/state/natlang/storage/cleanup-20261012-round2/cold-model-offload-receipt.jsonl`): ~31 GB more off NVMe.
At 16:08 df showed 194 GB free, but ~49 GB of that is the v4 trainer's `.checkpoint-space.reserve`, which was consumed by
the 15:45 checkpoint write and not yet re-reserved. Steady state is about 145 GB free.

Left on purpose: the live v4 `direct_sft/` (134 GB: `checkpoint.pt` and `best-checkpoint.pt` of 47 GB each plus the
49 GB `.checkpoint-space.reserve` for the next write), unions r2/r3, registered corpora (nearly every `runs/` dir is
registered), the call store (51 GB, self-bounded at 50 GB), `~/data/harness-bench` (SWE twins still rendering),
the pending `natlang-native-turns-view-ask-v4-render-20261011-v1` (in `pending-renders.json`), `bird-sqlite`
(protected source path in storage-policy.json), staged `vendor/datasets` (live symlink target), and the teacher
container and images. The rest of `runs/` is registered data; a further big gain needs a decision on superseded
registered render/proposal versions (e.g. the `neuralese-reviewed-restart-cohort-20261007-v*` series, ~15 GB).
