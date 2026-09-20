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

The generic host accepts a trial backend. The included merge backend binds one programme
per experiment, so its candidate source revision is unambiguous. It supports both
strategies for documents and whole-history trials for the other typed shapes. A trial's
`quality` remains pending until separate semantic review; equal digests measure only
observed repeatability, not correctness or convergence.

The command writes an append-only journal beside the report. `inspect_journal` identifies
unobserved planned slots after an interruption. It never assumes that an interrupted
external operation had no effect and never silently retries it.
