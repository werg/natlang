# Directory-reducer teacher and judge runbook

Run from `ts-host/`. Dataset acquisition, shard generation, and verification use CPU/network only. The teacher and
judge calls start only at the `teacher/cli.js` step. Use separate train and test seeds; dataset-backed cases also use
disjoint source-record hash pools. Raw datasets and generated rows live outside git.

The acquisition and build commands below have already been run for this checkout. The ready shards are
`data/teacher/directory/train.ir.jsonl` (48 verified cases) and `test.ir.jsonl` (24 verified cases), with reports beside
them. Their source groups have zero train/test overlap. When the GPU is free, start at the collector commands below;
rerun the CPU commands only to refresh a shard after changing its generator.

```bash
npm run build:node
node scripts/inline-curriculum/acquire.mjs \
  --source sms_spam,sst2,ag_news,emotion,banking77,clinc_oos,coedit,hotpotqa,cuad \
  --cache ../vendor/datasets --manifest ../vendor/datasets/sources.manifest.json
node scripts/inline-curriculum/build.mjs --seed 1101 --shapes 4 \
  --families folder_triage,folder_index,folder_edit,folder_find,folder_extract,folder_mixed \
  --split train --out ../data/teacher/directory/train.ir.jsonl
node scripts/inline-curriculum/build.mjs --seed 9101 --shapes 2 \
  --families folder_triage,folder_index,folder_edit,folder_find,folder_extract,folder_mixed \
  --split test --out ../data/teacher/directory/test.ir.jsonl
```

Each build verifies scripted references through the runtime and writes an adjacent `.report.json`. A failed case
stops the build without writing an IR shard. The source manifest records exact revisions, checksums, and licenses.
Never feed the test IR to the SFT exporter or teacher training.

When model inference is available, point the teacher and judge at their own OpenAI-compatible endpoints. Pi providers
can be used instead with `--provider` and `--judge-provider`. The collector journals each model turn, resumes matching
jobs, and records both model identities in provenance. Give the teacher and judge distinct output directories for each
shard and configuration. The judge is used only for `judged` oracles; exact, normalized, and span checks are local.

```bash
node dist/teacher/cli.js ../data/teacher/directory/train.ir.jsonl \
  ../data/teacher/directory/train-jobs ../data/teacher/directory/train.results.jsonl \
  --model-id TEACHER_ID --server http://127.0.0.1:8081 --root-seed 1101 \
  --judge-model-id JUDGE_ID --judge-server http://127.0.0.1:8082 \
  --workers 2 --context-tokens 16384 --all
node scripts/inline-curriculum/admit.mjs ../data/teacher/directory/train.results.jsonl \
  --ledger ../data/teacher/directory/train.admission.jsonl \
  --admitted ../data/teacher/directory/train.admitted.jsonl --require-technique
node dist/teacher/cli.js ../data/teacher/directory/test.ir.jsonl \
  ../data/teacher/directory/test-jobs ../data/teacher/directory/test.results.jsonl \
  --model-id STUDENT_OR_TEACHER_ID --server http://127.0.0.1:8081 --root-seed 9101 \
  --judge-model-id JUDGE_ID --judge-server http://127.0.0.1:8082 \
  --workers 2 --context-tokens 16384 --all
node scripts/inline-curriculum/score.mjs \
  --ir ../data/teacher/directory/test.ir.jsonl \
  --results ../data/teacher/directory/test.results.jsonl \
  --train ../data/teacher/directory/train.ir.jsonl \
  --out ../data/teacher/directory/test.score.json
```

The score file includes missing case IDs, completion/acceptance counts by family and oracle level, and overlapping
source groups. Any train/test source overlap makes scoring exit with code 2. For an honest student evaluation, collect
the held-out shard with the student model after training; a teacher run on test measures teacher coverage only.
