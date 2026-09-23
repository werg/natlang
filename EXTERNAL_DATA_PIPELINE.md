# External decision data pipeline

> **Status (2026-09-23):** historical. The decision-data adapters and materializer described here ran on the Python runtime and were removed with it. Bringing these sources back means writing `natlang.program/2` adapters; see [PROGRAM_IR_PIPELINE.md](PROGRAM_IR_PIPELINE.md#retired-sources).

This is the executable companion to [DATASET_TRAJECTORIES_PLAN.md](DATASET_TRAJECTORIES_PLAN.md).
It creates typed natlang interpreter trajectories from direct input/output
pairs and from prompts labeled by classifier.dev's free Jev endpoint. Every
retained trajectory is run through the natlang harness and checked against its
expected value and turn grammar. Use Python 3.12 through `.venv/bin/python`.

## Task format

Adapters produce JSONL with `id`, `source`, `group_id`, `split`, `state`,
`instruction`, `kind`, `labels`, optional `criteria`, source revision and
license. Direct pairs also have `gold` and `gold_source`; prompts awaiting
Jev have neither. `kind` is `choice`, `boolean`, or `score`; boolean labels
are `false` and `true`, and score labels are stringified discrete levels.
The Jev labeler retains the input record and appends the request, response,
per-item model and scores. It fills absent `gold` only for a valid response
attributed to Jev. Source gold is never replaced.

## Direct pairs

Download the source files separately, then normalize them:

```bash
.venv/bin/python scripts/prepare_direct_pairs.py nanojev \
  PATH/TO/stage1/train.jsonl --source-revision DATASET_SHA \
  --nanojev-families smart_home_v2 catalog_lookup_v2 support_decisions_v1 \
  -o data/external/nanojev-tasks.jsonl

.venv/bin/python scripts/prepare_direct_pairs.py typed-decisions \
  PATH/TO/customer_service/train.parquet --source-revision DATASET_SHA \
  -o data/external/typed-tasks.jsonl

.venv/bin/python scripts/prepare_direct_pairs.py jeff \
  PATH/TO/jeff/bench/data -o data/external/jeff-test-tasks.jsonl
```

Sources: [NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data),
[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
and [jeff benchmarks](https://github.com/logan-markewich/jeff/tree/main/bench/data).
The Typed Decisions Parquet path needs `pyarrow`; a JSONL export works without
it. jeff's committed benchmark records are marked `test` and upstream reuse
terms must be checked before any training use. NanoJev rows with several
optimal actions are skipped instead of assigning an arbitrary hard target.
The recommended family filter keeps support, catalog and smart-home
decisions. Grid navigation and tic-tac-toe require exact search; use them to
exercise crisp helpers and program control flow in a separate experiment.

## New Jev labels from sales conversations

The [sales dataset](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations)
is a 7.17 GB CSV with 3,072 embedding columns. Its adapter streams the CSV,
keeps speaker roles, and creates objection and intent questions on conversation
prefixes. Future turns, scenario metadata, outcome and automatic scores are
excluded from the classification input.

```bash
.venv/bin/python scripts/prepare_recent_tasks.py sales \
  PATH/TO/cleaned_custom_dataset.csv --out data/external/sales-tasks.jsonl \
  --max-conversations 1000 --max-prefixes 3

.venv/bin/python scripts/label_classifier.py \
  data/external/sales-tasks.jsonl data/external/sales-labeled.jsonl \
  --dry-run --max-items 20 --max-requests 2

.venv/bin/python scripts/label_classifier.py \
  data/external/sales-tasks.jsonl data/external/sales-labeled.jsonl \
  --max-items 100 --max-requests 10 --batch-size 20
```

The default endpoint is the versioned `POST /v1/classify`, with `tier: fast`.
Items with identical labels and instructions are batched. Both caps bound a
run; `--max-requests` includes retries. Results are appended and a rerun skips
completed IDs after checking input hashes. The audit field stores the actual
model for each item. Inspect Jev/source agreement or audit a sample before
scaling; Jev's score alone does not establish correctness in a new domain.
Current free limits are documented at [classifier.dev](https://classifier.dev/docs).

### Bulk Jev generation

The small commands above are contract checks. For sustained generation, use
the disk-backed deterministic shuffle and daily runner. The runner labels up
to 18,000 tasks per invocation by default, leaving space below the current
20,000/day free fast-tier limit for other calls. It writes full requests and
responses once per batch in `LABELS.jsonl.batches.jsonl`; each item carries a
`batch_id`, score map, model, status and final gold label. Re-running with the
same input skips completed IDs and retries HTTP error rows.

```bash
# Option A: use the entire 7.17 GB source CSV for exhaustive task generation.
.venv/bin/python scripts/prepare_recent_tasks.py sales \
  PATH/TO/cleaned_custom_dataset.csv --out data/external/sales-all-tasks.jsonl \
  --max-prefixes 3

# Option B: spread a smaller sample over the whole pinned source file.
.venv/bin/python scripts/sample_sales_ranges.py \
  --out data/external/sales-spread.csv --chunks 80 --chunk-bytes 4194304
.venv/bin/python scripts/prepare_recent_tasks.py sales \
  data/external/sales-spread.csv --out data/external/sales-spread-tasks.jsonl \
  --max-prefixes 3

.venv/bin/python scripts/shuffle_tasks.py \
  data/external/sales-all-tasks.jsonl data/external/sales-shuffled.jsonl

# Run once daily, or use --continuous for unattended successive quota windows.
.venv/bin/python scripts/run_bulk_classifier.py \
  data/external/sales-shuffled.jsonl data/external/sales-bulk-labeled.jsonl

.venv/bin/python scripts/audit_classifier_output.py \
  data/external/sales-bulk-labeled.jsonl --out data/external/sales-bulk-audit.json
```

Use the task path from option B in the shuffle command when range sampling.
The range sampler saves exact source byte spans and validates complete CSV
records; it does not claim full-dataset coverage. The full adapter reads the
CSV as a stream, ignoring embedding columns in the emitted tasks. All tasks
are ordered by a seeded hash before labeling so early daily tranches cover
many conversations instead of one source company. The `--continuous` runner
waits between windows and obeys the API's `Retry-After` on a 429. Run it under
a process supervisor for a multi-day job. A one-day invocation is safe to
rerun from a scheduler.
The audit checks each item against its batch request and response, then reports
accepted counts, per-question label distributions, models, splits and source
groups. Review its distributions and spot-check examples before training.

## SQL injection pairs and label audit

The [Kaggle dataset](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset)
contains `Modified_SQL_Dataset.csv`. Its adapter removes empty queries,
duplicates and exact conflicting labels. Capped pilots draw deterministically
from both classes. A normalized payload hash is used only for split grouping;
the model sees the original query.

```bash
.venv/bin/python scripts/prepare_recent_tasks.py sql \
  PATH/TO/Modified_SQL_Dataset.csv --out data/external/sql-tasks.jsonl \
  --max-tasks 100

.venv/bin/python scripts/label_classifier.py \
  data/external/sql-tasks.jsonl data/external/sql-jev-audit.jsonl \
  --max-items 20 --max-requests 1 --batch-size 20
```

Kaggle reports the license as **Unknown**. The trace generator excludes those
rows by default. `--allow-unknown-license` permits a local research run and
retains that license in every trace; settle reuse terms before distributing
or training on these records.

## Verified traces and SFT export

Pass any number of normalized direct or Jev-labeled JSONL files to the shared
generator. It defaults to the `train` split and emits three program families:

- `external_leaf`: one typed decision and its checked return;
- `external_map_report`: classify a list, then count labels with exact code;
- `external_case_report`: call separate typed decision functions over one
  source state and fill a structured report.

```bash
.venv/bin/python scripts/generate_external.py \
  data/external/nanojev-tasks.jsonl data/external/typed-tasks.jsonl \
  data/external/sales-labeled.jsonl \
  --out data/external/train-traces.jsonl

.venv/bin/python scripts/export_sft.py \
  data/external/train-traces.jsonl data/external/train-sft.jsonl \
  --workers 8 --resume
```

`export_sft.py` uses the running local model server's `/apply-template`, as
the rest of natlang training does. Parallel template requests remain bounded
and preserve source order. With `--resume`, the exporter checks the last
existing ID against the source before appending. To generate a held-out file, pass
`--splits test --out ...`; never mix its records into training. A JSON manifest
beside each trace file records task file hashes, filters and turn counts.
Direct adapter output also has a manifest with source file hashes; the sales
adapter records its pinned source revision and input size, while SQL records
its CSV hash.

## Verified small pilot (2026-09-19)

The local pilot in ignored `data/external_pilot/` used 20 NanoJev states
(60 direct questions), three Typed Decisions cases (15 direct questions),
and three sales conversations (12 Jev-labeled prefix questions). The live free
API returned `jev-1.13.0` for all 12 sales items. The combined run generated
**135 programs and 763 verified turns**, then exported **763 SFT pairs**.
The program mix exercised `read`, `call`,
`run_code`, `write`, and completion. A separate balanced 20-query SQL audit
returned Jev answers for every item and agreed with 19 source labels; one
disagreement remains an audit case. This pilot checks the pipeline and its
format, not the statistical quality of a large training corpus.
The jeff adapter also imported all **1,600 committed benchmark rows** as
`test`; 20 local research trajectories were verified from that pool. The
source records are kept outside the training mix.

The full NanoJev stage 1 train file produced **3,182 labeled tasks**, **5,032
verified programs** and **25,166 SFT pairs**. It includes algorithmic game
families, so it is an exploratory corpus. Filtering to the three recommended
families yields **1,272 direct tasks**. With the 15 Typed Decisions and 12
Jev-labeled sales tasks, the curated mix contains **1,299 tasks** and **1,961
verified programs** (9,925 turns). The source files, task JSONL, trace JSONL
and SFT JSONL are in ignored `data/external_pilot/`.

## Bulk sales run (2026-09-19)

The pinned, distributed 80-range sample covered **4,601 conversations from
20 companies**, producing **27,606 prompts**. A seeded shuffle made a
16,000-item free-tier tranche span all 20 companies. The API returned **16,000
accepted scored `jev-1.13.0` results** in 161 requests. The compact batch
audit matched every item to its saved request and response. The tranche has
**12,779 train** and **3,221 dev** tasks. The train tasks generated **19,632
verified programs**, **122,373 turns**, and **122,373 ordered SFT pairs**.
The files are `sales-spread-shuffled.jsonl`, `sales-spread-labeled.jsonl`,
`sales-spread-audit.json`, `sales-bulk-train-traces.jsonl`, and
`sales-bulk-train-sft.jsonl` in ignored `data/external_pilot/`.

An additional run consumed the remaining free allowance: **1,900 more Jev
labels** were accepted, and the next 100-item request returned HTTP 429 with
`rate_limit_day` and zero daily classifications remaining. The full labeled
file now has **17,900 accepted items** plus 100 HTTP-error placeholders. The
1,900 accepted additions generated **1,884 verified programs** and **10,550
turns/SFT pairs** from 1,507 train tasks. Those additions are in
`sales-spread-topup-labeled.jsonl`, `sales-topup-train-traces.jsonl`, and
`sales-topup-train-sft.jsonl`.

The remaining **9,706 prompts** are queued in the shuffled input, including
the 100 rejected at the daily limit. Re-run `run_bulk_classifier.py` with the
same input and output paths in the next free-tier window; it skips accepted
IDs and retries HTTP error rows. This corpus is a large pipeline and format
validation, not a claim that all Jev labels are correct. Review label
distributions and examples before treating it as final training gold.
