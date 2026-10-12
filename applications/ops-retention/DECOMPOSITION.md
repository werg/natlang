# Ops retention: decomposition, part by part

Storage retention decides, for every large file under the run and data roots, whether it is kept, in use, deleted
automatically or listed for the owner (AGENTS.md "Storage hygiene and discarded work", plans/STORAGE_POLICY.md). It
is the first of our own operations policies ported to natlang (plans/REVIEW_20261012.md, recommendation 7). Until
now it lived only in crisp Python: `classify()` in `scripts/storage_retention.py`.

Decisions per part:

- **fn**: its own natural-language function with a typed contract.
- **inline**: one instruction inside its caller.
- **implicit**: left to the model (rare).
- **crisp**: a helper, only for exact facts, durability and plumbing.
- **pluggable**: one interface, a crisp and a natural-language implementation, selected by a setting
  (`ts-host/src/runtime/pluggable.ts`). Crisp stays the default.
- **service**: the outside world (file system, processes, the memory ledger, docker).

## Policy and mechanism

- **Natural language: the decision.** Given the facts of one file, which tier it falls in and why
  (`classifyRetention.nl`).
- **Crisp: the facts, which have one right answer.**
  - the walk of the roots, sizes, inodes, the archive's logical paths;
  - which files a corpus or artifact manifest pins;
  - whether a run script, recipe, registry, plan or certificate names a path (substring search over those texts);
  - open files, process working directories, ledger claims, archiver excludes;
  - step numbers parsed from checkpoint names, and the latest step per directory.
- **Crisp: the exact bound and the effect.** A natural-language verdict never deletes a file the facts mark as
  registered, named, a record, under a keep path, in use or in a recently changed run: such a verdict is refused and
  the crisp verdict serves instead (`refusal` in index.ts). The deletion itself (re-stat, unlink, deletion manifest)
  stays in `storage_retention.py`; it acts only on PRUNE-AUTO and owner-approved PRUNE-ASK, whichever implementation
  classified.
- **Teach in errors, not prompts.** The instructions state the wanted decision for each case. A refused verdict is
  recorded with the rule it broke; that record is the feedback for the next revision of the instructions.

## Data

All in `types.ts`; the Python side builds the same field names in `retention_facts()`.

- `RetentionFacts`: one file's `category`, `age_days`, `run_age_days`, `registered`, `named_by_reference`,
  `directory_named_by_reference`, `record_file`, `policy_keep_path`, `in_use` (why, or null), `partial_write`,
  `run_marked_stopped`, `step`, `latest_step_in_directory`, `step_files_in_directory`, `best_or_final`.
- `RetentionThresholds`: `finished_after_days`, `aborted_after_days`, `leftover_after_days` (from
  `training/storage-policy.json`).
- `RetentionVerdict`: `tier` (`KEEP | ACTIVE | PRUNE-AUTO | PRUNE-ASK`) and `reason`.
- `RetentionItem`: `id`, `path`, `logical`, `bytes`, `facts`, and `crisp` (the crisp verdict).
- Shadow record (JSONL, one per disagreement or refusal): `time`, `mode`, `id`, `path`, `logical`, `bytes`,
  `crisp_tier`, `crisp_reason`, `nl_tier`, `nl_reason`, `refused`, `error`, `served`.

## Parts

| Part | Decision | Unit | Why |
|---|---|---|---|
| Walk the roots, sizes, inodes, logical paths | crisp | `scan()` (Python) | outside world; exact |
| Category of a file (checkpoint, export, cache, ...) | crisp | `category()` (Python) | a pure function of the name and suffix, used for totals of every file, millions of them; a fact the policy reads |
| Registered files | crisp | `registered_files()` | exact set membership over manifests |
| Named by a script, recipe, plan or certificate | crisp | `referenced()`, `dir_referenced()` | substring search over about 4 MB of text; exact |
| In use (open, cwd, claim, exclude) | crisp | `active_state()`, `active_reason()` | processes and the ledger: outside world |
| Steps per directory, latest step | crisp | `retention_facts()` | regex and max over the directory's entries; exact arithmetic |
| Facts of one file | crisp | `retention_facts()` | gathers the above into `RetentionFacts`, the single input both implementations read |
| **Decide the tier of one file** | **pluggable** | crisp `crisp_verdict()` (Python); nl `classifyRetention.nl` | the policy. One file per call keeps each call small for small executors |
| Keep rules: registered, named, record file, keep path | inline | steps 1-4 of `classifyRetention.nl` | each is one comparison of a fact |
| In use, or run changed recently | inline | steps 5-6 | one comparison each |
| Leftover partial writes | inline | step 7 | one comparison |
| Checkpoints of stopped runs | inline | step 8 | two-way choice on one fact |
| Intermediate checkpoints | inline | step 9 | four comparisons over facts already computed |
| Everything else is listed for the owner | inline | step 10 | the default |
| The reason text | implicit | inside each step | free wording; shadow compares tiers only |
| Bound on an nl verdict | crisp | `refusal()` in index.ts | an exact verifier of the stage's output: deletion must never follow a refused verdict |
| Mode setting, shadow comparison, trace event | crisp (existing) | `pluggable()` in ts-host | one implementation per concept; not reinvented |
| Disagreement log | crisp | `classifyItems()` in index.ts, written by `ts-host/scripts/retention-classify.mjs` | durability: append-only JSONL |
| Python to natlang bridge | crisp | `nl_classify()` in storage_retention.py runs `node ts-host/scripts/retention-classify.mjs` | the same subprocess bridge as `explain_gate.py` to `explain-advisory.mjs` |
| Deleting, deletion manifest, docker prune | crisp | `delete()`, `docker_prune()` (Python) | outside world, exact durability |

Why one function and not one per rule: each rule is a single comparison over a fact that crisp code has already
computed, so a separate function per rule would add a call and a contract for one comparison. Folding them into
numbered steps of one function keeps the order explicit (first rule that applies wins), which is the policy's core.

## The algorithm (as `classifyRetention.nl` spells it out)

Go through the steps in order and stop at the first that applies.

1. `registered` → KEEP.
2. `named_by_reference` → KEEP.
3. `record_file` → KEEP.
4. `policy_keep_path` → KEEP.
5. `in_use` not null → ACTIVE, reason `in_use`.
6. `run_age_days < finished_after_days` → ACTIVE.
7. `partial_write` and `age_days >= leftover_after_days` → PRUNE-AUTO.
8. category checkpoint, optimizer-state or weights-snapshot, `run_marked_stopped`, `age_days >= aborted_after_days`
   → PRUNE-ASK when `directory_named_by_reference`, else PRUNE-AUTO.
9. category checkpoint or optimizer-state, `step` not null, `step_files_in_directory > 1`,
   `step < latest_step_in_directory`, not `best_or_final` → PRUNE-AUTO.
10. Otherwise → PRUNE-ASK.

## Modes and how to run them

The setting is `classifier.mode` in `training/storage-policy.json`, overridden by the environment variable
`NATLANG_RETENTION_CLASSIFIER`, overridden by `--classifier`. Values: `crisp` (default), `nl`, `shadow`.

- `crisp`: today's behaviour; no model is called.
- `shadow`: both implementations classify every reported file. The crisp verdict serves (and alone decides what
  `--apply` deletes). Every disagreement in tier, every failed nl call and every refused nl verdict is appended to
  `~/.local/state/natlang/storage/retention-classifier-shadow.jsonl`; a summary (items, agree, disagree, refused,
  errors) goes into the report under `classifier`. A failure of the bridge as a whole is recorded and the crisp
  result stands.
- `nl`: the natural-language verdict serves, after the exact bound; a refused verdict falls back to crisp. A failure
  of the bridge stops the run before anything is deleted.

The model is an OpenAI-compatible endpoint: `classifier.server` and `classifier.model` in the policy, or
`--nl-server` / `--nl-model`. The one-week shadow run (2026-10-12 onward) is a dry run on the DGX teacher, inside its
teacher window, through the ledger:

```
scripts/memory_ledger.py run --unit ops-retention-shadow --budget-gb 4 -- \
  python3 scripts/storage_retention.py --classifier shadow \
    --nl-server http://127.0.0.1:8082 --nl-model <served teacher id> \
    --from-report ~/.local/state/natlang/storage/retention-latest.json \
    --out ~/.local/state/natlang/storage/retention-shadow-latest.json
```

`--from-report` reuses the last report's walk (hours on the HDD); classification is redone. Without `--apply`
nothing is deleted; with `--apply` in shadow mode only the crisp verdicts delete. After the week, read the JSONL:
each disagreement is either a bug in the facts, a wording problem in the instructions (fix it, measure it live as
plans/MODEL_FACING_CHANGES.md asks) or a crisp rule that does not say what AGENTS.md says (fix the crisp side).

## Not yet ported

- Judging a run as a discarded approach from its name and the current recipes (AGENTS.md: delete discarded runs).
  Neither implementation sees the path today; the crisp rules approximate it with run markers. A next unit would
  give the function the run's name and the list of active lineages, with the path as untrusted data.
- Sync-revision directories, docker pruning and approvals stay crisp mechanism.
