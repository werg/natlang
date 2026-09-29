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
blockers. WorkflowEvals visibility blocker is now repaired by the versioned v2
bundle and independently audited full-input proofs. Old bundle paths are retained
with explicit replacements. Final native materialization, split isolation,
rendered-pair deduplication, student-template/token checks and the25% reducer-share
gate still apply. Inventory readiness does not mean student training readiness.

## Admission and preference disposition

Native admission checks source conversion/provenance/release scope, source quality
and quarantine, current runtime outcomes, final oracle agreement, observed decisive
evidence, answer visibility and task-specific file/edit conditions. Materialization
checks linked executed actions and clean outcomes; failed, repeated, unexecuted and
proposal decisions are excluded. Synthetic gold/reasoning exceptions are narrowly
scoped; workflow references now require complete final-context source-input proofs.
Delegation versus direct work alone is not a rejection criterion.

Per-file snapshot ledgers preserve raw reasons plus categorized next actions from
`admission-dispositions.mjs`: migration/replay, source/oracle review, candidate
failure, evaluation/unsupported, duplicate/superseded or unclassified review. These
categories can overlap. Migratable artifacts remain discoverable; held-out scope
must never be migrated into train. Rejection does not establish a DPO negative.

Correction and preference builders preserve input hashes, runtime identity, parent
IDs, source groups, per-candidate causal checks and skip reasons in audit manifests.
The recipe now includes these offline stages. Current audited output has43 native
/2 pairs from116 selected candidates (35 failed actions,8 wrong results), not282
pairs:282 handoff result rows are inputs. Five legacy /1 pairs need migration.
Current native pairs remain pending student rendering/token/split/dedup review;
byte-identical outputs are counted once in the inventory. No DPO training started.


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
FinQA numeric evidence; expansion of state/dialogue/
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

## Visible-input publication

Workflow v2:4,805 cases,28,975 native decisions, complete-input proof recomputed from
actual tool/page outputs retained in the final answer context. Source records and
oracles are unchanged; old unversioned files remain archived and fail closed under
current source admission. Audit/publication lineage:
`runs/workflowevals-visible-20260930/publication.json` and `publication-audit.json`.
32k model-free replay context prevented evidence loss seen in37 cases at16k.
This does not waive the eventual student's context limits. Four newer static
bundles now total32,993 decisions/7,910 reducers (~23.97%); the older63.43% figure
is obsolete, and final unique admitted reducer mix still needs measured expansion.
Reviewed recipe-v2 includes published v2; it has not been executed as training.

Latest recipe-v3 planning snapshot:2,758 selected native trajectories/2,427 programs
from5,520 completed files. Inventory reports no missing carry-forward inputs or
included quality blockers. Held-file disposition counts overlap:1,038 migration/
replay,663 source/oracle review,1,214 failure candidates,213 evaluation/unsupported,
684 duplicate/superseded. These are artifact counts, not independent failures or
DPO labels. Offline history-migration pilot is explicitly held by admission via
history_migration_review_pending; same final answers do not establish equivalent
observations or provenance. Initial two same-answer candidates changed observations,
so strict pilot rejects them; eight others had ambiguous invocation ownership.
