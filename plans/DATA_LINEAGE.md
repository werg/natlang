# Training data inventory and carry-forward — 2026-09-30

## Default behavior

Both builders now discover completed native teacher jobs across `runs/` and
`data/teacher/`. `ts-host/scripts/snapshot-generated-training.mjs` creates a
content-addressed immutable result snapshot, a per-file reasons/replacement ledger,
and a manifest with model/family counts. It requires current IR, explicit teacher
role, train split and current admission. Failed, obsolete, held, evaluation and
unsupported rows remain on disk and in the ledger; their omission is not deletion.

The two older static reference exports and all four published static source bundles
remain defaults. Historical code inventories/captures continue through the staged
recipe's existing acquisition, assembly and replay lanes. Old model-rendered SFT is
inventoried for recovery/rerendering, not blindly mixed across chat templates.
Explicit supplied exports take precedence over automatic snapshot overlaps;
admission records `duplicate_input_trajectory_id_not_written` for duplicate IDs.
Within automatic snapshots, the latest eligible artifact per trajectory ID wins;
older versions/copies are retained and linked to the selected file and hash.

## Persistent view

Policy: `training/data_sources.json`. Catalog: `data/teacher/data-inventory/current.json`.
Each scan writes an immutable report as well as refreshing current.json. The catalog
covers data JSON/JSONL (including compressed files), CSV, Arrow, Parquet, source archives, published static manifests,
and saved run IR/result exports. Individual completed native jobs are covered by
snapshot ledgers. Frozen runtime copies are not dataset sources. Unknown artifacts
are a visible review backlog; the catalog does not claim these are distinct datasets
or usable samples. Missing artifacts remain listed. First-record format observations
are discovery hints, not full validity checks; selected snapshots have full hashes.

`data/teacher/data-inventory/overview.md` presents counts, source work and blockers.
Entries record presence, size/version, recipe membership, whether they have ever
been recipe inputs, status and next action. Policy records source dispositions and
replacement lineage. Reports include each recipe transformation's inputs, outputs
and command; source row provenance and split groups remain intact.

A default recipe fails if a required input is missing, or an earlier recipe input is
omitted without a recorded policy decision or replacement in the new inputs.
Explicit input overrides are recorded with an omission list. New canonically
versioned bundles must record old-to-new paths in policy `replacements`.
Do not delete old input records to make a build pass.

Before joint training, `audit-data-inventory` requires no unresolved inclusion
blockers. Current blocker: WorkflowEvals primitive scripted references need complete
visible source-state proof. Creating/reviewing a recipe remains possible; this guard
prevents using the affected bundle for joint training until repaired or explicitly
held. Final native materialization, split isolation, rendered-pair deduplication,
student-template/token checks and the25% reducer-share gate still apply.

## Current audit

Reviewed planning recipe: `runs/data-lineage-20260930/recipe.json` (not executed;
no training started). At22:33 UTC the automatic scan selected2,745 trajectories
across2,414 unique programs from5,490 completed files;684 eligible repeated IDs
were omitted. Inputs include the2,192 older static references and four newer bundles.
These are overlapping candidate layers, not a summed final ready corpus.

The first catalog contained3,836 artifact paths; adding raw JSON and source archives
expanded it to6,170 paths, including archival/derived duplicates.
Its backlog includes630 transformation-pending and125 rerender-review artifacts;
these numbers count files, not examples or distinct source datasets. Priorities:
Workflow primitive visibility; FinQA numeric evidence; expansion of state/dialogue/
scene adapters; reviewed leaf-bank joins; legacy teacher/IR conversion; old SFT
rerendering. Quality holds include unreviewed paper answer equivalence, tree
transition contracts, synthetic labels and obsolete runtime interactions.

## Commands

- Inventory without collecting or training: `python3 scripts/inventory_training_data.py`
- Reconcile a recipe: `python3 scripts/inventory_training_data.py --recipe PATH`
- Refresh completed-job snapshot: `node ts-host/scripts/snapshot-generated-training.mjs --repo /home/werg/natlang`
- Create a recipe: `python3 scripts/create_training_pipeline.py --output PATH`

`--teacher-results` replaces automatic result selection and is an explicit override.
The shell builder offers `NATLANG_GENERATED_RESULTS=off`; its inventory records that
opt-out. Prefer defaults for the complete corpus. Updating adapters does not release
raw data automatically: replay, provenance, source/group and quality checks are still
required. A build snapshot intentionally excludes jobs finishing later; the next
build discovers those completions automatically.
