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

- **Cadence by wall clock**, not steps: one rolling resumable slot every 45 min (configurable) and whenever the
  process is asked to stop (SIGTERM/SIGINT/SIGUSR1 → checkpoint at the next step boundary, then exit). A run loses
  at most 45 min of work; the disk sees ~3x fewer 47 GB writes.
- **One rolling slot**, written atomically and durably (pending → fsync → rename → directory fsync).
  If the disk cannot hold the replacement beside the previous slot, report insufficient space and preserve the
  previous checkpoint. Failed partial writes are removed; pruning is separate from checkpoint writing.
- **Best = weights only** (no optimizer state), on eval improvement.
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
