# Storage policy (DGX; Pop may opt in)

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
