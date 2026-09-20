# Experiment laboratory

The finite laboratory has two natlang entry points. `design.nl` chooses cases from a
supplied, bounded menu. The application host validates that plan, freezes candidate and
case identities, then runs isolated trials outside a crisp eval call. `report.nl` asks
natlang to interpret exact counts, repeatability and outstanding semantic review.

The host records one trial for every planned candidate, case and attempt, including
exceptions and missing results. Trial seeds are derived from logical case/attempt IDs;
model and world seed namespaces remain separate. The first backend compares semantic
merge programmes. Long model calls are application jobs, so the QuickJS eval timeout
does not silently turn them into uncertain nested effects.

Run a document strategy pilot against an explicitly identified model:

```bash
.venv/bin/python scripts/run_experiment_lab.py runs/document-experiment.json \
  --model-id MODEL-AND-CHECKPOINT --model-seed 43 --world-seed 71 --analyst-seed 29 \
  --budget 2 --repeats 2
```

The host accepts a domain-neutral trial backend. Included backends evaluate semantic
merging, type inference, dependency-plan testing and SQLite migration. A trial records
an application-specific `label`, exact status, provenance, review quality and trace
identity. A merge trial's `quality` remains pending until separate semantic review;
equal digests measure only observed repeatability, not correctness or convergence.

For a live application evaluation, run for example:

```bash
.venv/bin/python scripts/run_application_evals.py types runs/type-eval-1 \
  --model-id MODEL-AND-CHECKPOINT --model-seed 31 --world-seed 41 --analyst-seed 51
.venv/bin/python scripts/inspect_long_run.py \
  --journal runs/type-eval-1/journal.jsonl --trace-dir runs/type-eval-1/traces
```

There is no default agentic time, token, turn, tool-call, action, episode or nesting
budget. The case budget selects how many frozen scenarios to run; it does not bound
how long the model may reason. Traces append timestamps, model-request start/end
events and actions while the run is active. The inspector reports requests in flight
and repeated actions as observations, not an automatic stuck verdict. Raw teacher
turns are captured after each completed episode for later sample review.

The commands write an append-only journal beside the report. `inspect_journal`
identifies started and unobserved slots after an interruption. It never assumes
that an interrupted external operation had no effect and never silently retries it.
