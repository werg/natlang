# Focused SQL and graph reasoning successors

The 2026-10-31 source qualification sample supports these narrow actions:

- BIRD count/filter and simple Spider rows reached 6/6 or 5/6 success in same-policy rollouts. Route only those exact task groups to the existing successful-RFT/retirement path; do not claim that all SQL is mastered.
- Keep multi-join, nested-query, aggregate-with-HAVING, and set-operation SQL. Build the `spider-hard` and `bird-hard` successors with executable queries checked against the bound read-only SQLite database. Candidate rows are round-robined across normalized operation signatures within each database. The successor profile records query operations and requires ordered-result comparison whenever SQL specifies `ORDER BY`; a reversed row list is not an accepted alias.
- Keep bAbI's full history and CLUTRR's original story. The focused `babi-temporal3` helper receives all numbered facts and the query, preserving event IDs and before/after semantics. `clutrr-guided` extracts one direct relation per sentence with an awaited helper call, then composes the target-to-anchor path. Neither successor truncates input. They retain upstream story IDs as source groups, so related rows remain on one split.

These changes define new source names and do not rewrite source files, rollout identities, or already collected policy data. Their current evidence is limited to loader/reference checks and the round-8 per-case rollout outcomes; no success claim is made for the new successors until the exact verifier and then a same-policy behavior sample complete.

## Build and qualify a candidate successor

Use a new build directory and preserve the existing broad source build as the earlier `--build` input. Run each step under the memory ledger with a medium 2 GiB budget. For hard SQL, 12 Spider and 24 BIRD rows per database are upper bounds; the builder may produce fewer after exact SQLite execution and compositional filtering.

```sh
BUILD=/home/werg/data/broad-rl-tasks-20261031/build-focused-v1
python3 scripts/memory_ledger.py run --unit broadrl-focused-spider-hard --budget-gb 2 --priority medium --wait 600 -- \
  .venv-neuralese/bin/python -I scripts/rl_tasks/build.py spider-hard "$BUILD" per_db=12
python3 scripts/memory_ledger.py run --unit broadrl-focused-bird-hard --budget-gb 2 --priority medium --wait 600 -- \
  .venv-neuralese/bin/python -I scripts/rl_tasks/build.py bird-hard "$BUILD" per_db=24
python3 scripts/memory_ledger.py run --unit broadrl-focused-babi-temporal3 --budget-gb 2 --priority medium --wait 600 -- \
  .venv-neuralese/bin/python -I scripts/rl_tasks/build.py babi-temporal3 "$BUILD"
python3 scripts/memory_ledger.py run --unit broadrl-focused-clutrr-guided --budget-gb 2 --priority medium --wait 600 -- \
  .venv-neuralese/bin/python -I scripts/rl_tasks/build.py clutrr-guided "$BUILD" min_hops=4
```

Then qualify each source and inspect `verified/verification.json`, the task-family/query-profile histogram, source-group split collisions, exact SQLite execution receipts, rejected negative controls, and package/source hashes. Verify `babi-temporal3` preserves complete input histories and `clutrr-guided` preserves original relation stories. Only then assemble to a fresh successor source, passing the prior broad build first and the focused build last. Preserve every held packet and require a new plan/prompt/source identity for collection. Do not append these cases to a live or in-flight collection.

```sh
for S in spider-hard bird-hard babi-temporal3 clutrr-guided; do
  python3 scripts/memory_ledger.py run --unit "broadrl-focused-verify-$S" --budget-gb 2 --priority medium --wait 600 -- \
    sh scripts/rl_tasks/verify_source.sh "$BUILD" "$S" 2
done
```

The exact assembly and policy-sampling command must be written into the next reviewed source transition after the `ProtectedWorlds` admission receipt and dataset bindings are inspected. This document is a prepared procedure, not approval to mutate a live recipe or collect trajectories.
