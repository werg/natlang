# Training data discoverability and organization: findings (2026-11-20)

Written while building the training-data catalog of the ops page (`scripts/ops_dashboard/data_catalog.py`). The catalog had to find
datasets by parsing `grpo.py`, globbing run directories and keeping a hand layer (`scripts/ops_dashboard/catalog_sources.json`),
because no single registry knows what we train on. Each item below is a concrete gap with paths and evidence. They are input
for a follow-up project: a centralized registry that producers write automatically (datasets, tasks, runs consuming them). The
catalog is built as providers (`code_constants`, `grpo_runs`, `teacher_campaigns`, `registry`, hand layer) so it can switch
to that registry and drop the hand layer.

## Datasets no registry knows about (live training inputs)

1. **The two largest GRPO task pools are unregistered.** `/mnt/external/natlang-rl-composed-20261103/memetic-program-source-20261105/cases.jsonl`
   (28,733 rows, 341 MB) and `/mnt/external/natlang-rl-bgkit-kr-20261103/natlang-rl-bgkit-kr-20261103/cases.jsonl` (4,734 rows) are
   in `grpo.py:108-110` `POOLS` but have no entry in `training/neuralese_corpora.json` and no corpus manifest (only the bgkit
   `-dev`/`-confirmation` held entries are registered).
2. **memetic-program-source-20261105 has no provenance at all**: no receipt, build.json, sha256 or producer commit. The producer is
   probably `scripts/rl_tasks/assemble_selection_pool.py` with `natlang-rl-composed-20261103/pool-spec.json`; this is inference.
3. **Teacher-gen campaigns are unregistered**: `ops/teacher-gen-20261115` (47 GB: pool-v1, pool-v2-rich, pool-progtune-decisions,
   smokes) and `ops/teacher-gen-20261119/gap-v1` (26 GB). Their license and provenance exist only as prose in the sft-mix entry's
   `license_provenance`.
4. **SFT work directories are unregistered**: `/home/werg/data/sft-sources-20261117/-20261118/-20261120` (~17 GB) and
   `/home/werg/data/sft-teacher-demos-20261118/-20261119` (~9 GB). They are defined only by `grpo.py:153 SD_TEACHER_SOURCES` and
   `ops/sft-v5/build.sh`.
5. **Memetic tasks, hindsight index, operator rows and operator chains are unregistered**: `/home/werg/data/memetic-tasks-2026111{7,8}`
   (`grpo.py:107 MEMETIC_TASKS`), `/home/werg/data/hindsight-index-20261117` (producer visible only in its `index.log`),
   `/home/werg/data/natlang-operator-rows-20261117-*` (manifest records `runtime_code.dirty: true`), `ops/operators-v1/merged`
   (an sft-mix aux input, `ops/sft-v5/build.sh` line 31).
6. **Two built SFT sets are missing from the registry**: `runs/sft-verified-20261117-v4` and `runs/sft-supplement-20261117-v1`
   (supplement's `merge.json` consumes v4).
7. **`/home/werg/data/vibestudio-live-corpus/seed-v2` is unregistered** although it is the sft-mix aux "live" input
   (`ops/sft-v5/build.sh` line 32); it appears only in `plans/neuralese/SFT_WARMUP_MIX_20261118.md`.

## Sets found only by reading code

8. **The task pool set is hand-maintained Python**: `POOLS` is a 16-element list with comments (`grpo.py:108-142`), and each task
   supply round is a code commit (61308efdf, 934dfabc9, 33b5fa03c). The only other record of a run's actual pools is its
   `WORK/pool.json` (and each teacher-gen `pool.json`).
9. **Every `ops/...` path means `/home/werg/data/lfm3-rl-dgx-20261029/ops`**, while the repo has its own `ops/` with only
   `decisions.log` and storage (`sft_sources.py:45 OPS`). A reader cannot resolve a plan's `ops/` path without knowing this.
10. **The supply builder and batch patch scripts are not in git**: `/mnt/external/natlang-rl-authoring/round10/build_supply.py`,
    `batches/w01/b0001/patch_*.py`. The registry note is the only pointer.
11. **A data directory doubles as a code dependency**: `scripts/rl_tasks/qualify_authored.py:37` imports `union_groups_fast.py` from
    `/home/werg/data/broad-rl-tasks-20261031`.

## Missing manifests, provenance or licenses

12. memetic-tasks rows carry no `license` key; `/mnt/external/external-rl-sources/_raw/*` has no `PROVENANCE.json`.
13. Rollout archives (`/mnt/external/natlang-grpo-v*/run*/rollouts`) have no manifest of their own; the run's `runs.jsonl` and the
    archive directory naming (`natlang-<run group>/<run>/rollouts`) are the only link back to the run.
14. Held-out material is spread over `ops/held-v3` and `ops/held-v2` manifests, `training/protected-worlds.json`,
    `training/memetic-held-programs.json`, registry admission strings and per-pool `held/` directories; no single list says what
    is protected.

## Inputs that change in place, and what reached the learner

15. **A training input grows while runs read it**: `/home/werg/data/sft-teacher-demos-20261119` and
    `ops/teacher-gen-20261119/gap-v1/manifest.jsonl` were still being written on 2026-11-20 (15,034 manifest lines vs 14,656
    extracted). `build.sh` uses `--allow-stale-teacher`, and grpo reads the sources at start or resume, so a run's input depends
    on when it started and no snapshot sha is recorded for it.
16. **What reached the learner is mostly not recorded.** SFT: nothing records which chains were in step N (re-derive with
    `exp0_sft.step_groups` and the seed); load-time masks (empty think, after `<|im_end|>`, second `</think>`) are not stored.
    GRPO: per-step learner batches (`step-chains/`) are kept for the last 4 steps only; whether a group came from the pass band or
    exploration is decided in memory and never written; rho and final per-token weights are not stored.

## Duplicates, orphans and supersession

17. Copies and symlink chains across locations: per-file `chains.pkl` symlinks in `runs/sft-merged-20261117-v3` and
    `runs/sft-verified-20261117-v4` into `/mnt/external/natlang-development-data/archive/runs`; `/home/werg/data/bgkit-export`
    resolves through two links to `nvme-cold-20261022`; `dgx-ops-archive-20261118` holds copies of sft-v3/v4 and grpo-v4; `runs/`
    holds 1,531 symlinks.
18. Supersession is not recorded for: internal-admission-readers v1-v5, `ops/operators-v1/merged-v1-stale`, memetic-tasks-20261117
    vs -20261118, 14 contractnli-v10-reference-readout versions, broad-rl-tasks `assembled-*` directories, and
    `natlang-rl-authoring/rotation.jsonl.pre-reconcile-*`.
19. Probable orphans (no consumer found in code or run records): `/mnt/external/natlang-full67-onpolicy-rollouts-20261103` (17 GB),
    `/mnt/external/natlang-rl-bgkit-kr-20261104` (pool-spec uses it, POOLS still names -20261103), `/home/werg/data/memetic-{20261012,
    20261013,actor,e2e-v3}`, `vibestudio-teacher-pilot-20261117`, luna/contractnli sets. Not verified: consumers may exist outside git.
20. Non-data directories sit beside training data in `/home/werg/data`: `natlang-internal-admission-readers-*` (code restoration
    snapshots), logs, scripts and a 20.7 GB `tmp-hz2-state-34.safetensors`.

## Tracked by hand

21. Excluding a task needs a hand-written JSONL under `ops/pool-errata/`; excluding an SFT row, teacher demonstration or episode
    has **no mechanism at all**: `sft_sources build` reads neither pool errata nor any row exclusion list (only `--exclude-run`).
    The ops page now records such exclusions in `ops/data-edits/exclusions.jsonl` and RL task exclusions in
    `ops/pool-errata/ops-dashboard.jsonl`; the SFT builder still has to read the former.
22. Corrections of a task (oracle, inputs, program file) have no path back into the pool: they need re-qualification
    (`qualify_authored.py` verify/smoke) and a new pool version. The page records them in `ops/data-edits/corrections.jsonl`.
23. Which runs consumed a dataset, and how much, has to be reconstructed from scattered records (`WORK/pool.json`, teacher-gen
    `pool.json`, sft-mix `build.json` inputs, SFT `launch.json`), each with its own schema.

## What a centralized registry needs (proposal for the follow-up)

- Producers append a record when they finish: `{id, kind, path, files[{path, bytes, sha256, lines}], producer{script, commit,
  dirty, argv}, inputs[{id, sha256}], license, protected, supersedes}` (the corpus-snapshot manifest generalized, written by
  `teacher_gen.py`, `sft_sources.py`, `qualify_authored.py`/supply builders, `memetic_tasks`, `operator_rows.py`, `grpo.py`
  archives).
- Consumers append a consumption record at start/resume: `{run, dataset id, sha256, how, amount}`; grpo `POOLS` becomes a
  registry query (`kind=rl-tasks, admitted=true`) instead of a code list.
- Exclusions, errata, flags and corrections are one registry layer keyed by `(dataset id, item id)`, read by every builder.
- The held/protected set is one registry attribute that every builder and the ops page read.
