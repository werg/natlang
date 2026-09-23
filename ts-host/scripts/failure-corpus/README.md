# Semantic failure-repair corpus

`cases.mjs` contains hand-authored, versioned tasks with an intentionally wrong first `eval`, a reference repair, and an exact output oracle. The production pipeline generates its frozen training input under the run directory from the frozen runtime. The ten cases cover foreign-key joins, identity normalization, revision cutoffs, locale-aware money, sparse arrays, overlapping-rule precedence, configuration inheritance, settled-ledger signs, resettable event streams, and percent-encoded path segments.

Each failed first eval is executed by the normal native compiler and runtime. The teacher collector then receives the actual failed tool result, dynamic repair prompt, and immutable `debug` binding. Successful continuations are admitted only if the final value matches the oracle. The seeded bad action appears in the trajectory for context but is not an approved training target. The materializer emits the subsequent repair decisions with their exact, snapshotted request contexts.

Regenerate the frozen input after changing a case:

```sh
cd ts-host
npm run build
node scripts/failure-corpus/freeze.mjs ../data/teacher/failure-repair-cases.jsonl
node --test test/failure-corpus.test.mjs
```

The freeze command refuses to overwrite changed artifacts. Remove or archive the old corpus files deliberately before regenerating. The production training recipe prepends these records to its teacher-seed file. To collect them independently against a running Bonsai server:

```sh
node scripts/teacher-collector.mjs ../data/teacher/failure-repair-cases.jsonl \
  ../data/teacher/failure-repair-jobs ../data/teacher/failure-repair-trajectories.jsonl \
  --model-id /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --root-seed 17 --limit 10 --workers 1
node scripts/materialize-native-teacher.mjs ../data/teacher/failure-repair-trajectories.jsonl \
  ../data/teacher/failure-repair-turns.jsonl
```

Collector jobs are individually durable and resumable. A case with an unaccepted teacher continuation is not gold data; keep its seed record for a later collection, but do not manually mark it accepted. The corpus is intentionally a starting set of distinct failure mechanisms rather than many near-duplicate parameter variations.
