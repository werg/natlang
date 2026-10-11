# Native E2E input inventory — 2026-10-11

Read-only inventory for planning a shared consumer end-to-end (E2E) adapter. This records what the local recipes and registered artifacts contain; it does not grant new data admission or model qualification.

## Active direct-stage text input

The current `luna-neuralese-direct-lfm-base-20261011-v8` recipe is [pinned here](../../training/neuralese/recipes/luna-neuralese-direct-lfm-base-20261011-v8.json), SHA-256 `6cede93b8d68dc3aa6717b38fb6f598f90c6d0d3618b8e9f4be86bf480728abe`. Its `text_data` input is `/srv/storage/natlang/prepared/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl`, SHA-256 `30ef44f4670d47a065ff8ba96aff3e46650526b07236c03e6a9d8429480c8576`.

The corresponding composition review, [root-input-composition-review-v1.json](../../runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/root-input-composition-review-v1.json), SHA-256 `f86c58c6b79e23155d7ab8a28616f5c6a1afb21e144b48fd5862058b223b2666`, verifies 4,962 exact ordered rows: 3,258 train and 1,704 test. It reports 82 added train rows over the earlier 4,880-row prefix and no change to held rows; it explicitly says there is no new example admission or runtime/alignment qualification.

`natlang_neuralese.train.text_warmup.load_text_rows` takes its `text_data` branch whenever that argument is set. Thus the v8 direct-stage loader reads these 4,962 text rows as its row source; the recipe's separately pinned `records` and `pieces` are not substituted for text rows in that branch. Recipe versions v1–v8 pin the same three paths and hashes. The recipe also states that this stage is native-text SFT only: no recurrence, no harness benchmark, and no new action admission.

## Native103 action rows

The recipe separately pins `runs/luna-authored-root-remaining-four-multilambda-20261009-v1/evidence/compact-62-luna-1-step-native-text-proposal-v1/compact-112-expanded-v1/root-independent-targetbound-replay-v1/repair-native103.jsonl`, SHA-256 `a261046dacb5e53ddda168acfd4a13ab2ea59a52312e933aa6e901149c1889e6`, and cumulative `native-pieces.jsonl`, SHA-256 `b3a30bae2767cbeb7fa5dd4ebd7b8bd062f5eb82449c48d13b9de6a6965592f0`.

The native103 file has 103 `natlang.teacher_training_turn.native/1` action-level rows: 87 train and 16 test, covering 21 teacher trajectories, 85 distinct invocation IDs, and 42 source groups. Rows include `source_ref` (trajectory and raw source-row hash, invocation ID, and sometimes inline-instruction site/capture metadata), `task`, `messages`, `target`, and trace/admission fields. This is not the `natlang.replay-record/1` whole-program schema.

There are 88 `provider-expanded-read-context` references in 75 rows: 75 `same-run-producer` refs (73 train, 2 test) and 13 `configured-function-definition` refs (train). All 75 same-run refs say `writer_target_selected: false`. In 42 of those refs, the referenced producer invocation also has a row in native103, but it is still not a selected writer target. These references attest context/read provenance; they do not establish 75 supervised writer-to-consumer recurrence pairs. Do not derive writer targets from typed message parts or missing rows.

Descriptive message-part counts across native103: 67 rows contain a Neuralese part, 15 a soft part, and 4 a read part; 15 rows contain `neuralese_conversion` metadata, and 12 have `teacher_reasoning_trained`. These counts do not establish training-admitted recurrence edges.

## Cumulative recurrence assembly

The reviewed cumulative v3 recurrence input is `runs/v18-v19-native-soft-edge-review-20261009-v1/cumulative-full-assembly-v3/recurrence-records.jsonl`, SHA-256 `06a7de7eea9e21667464b21f88561a92a34156307fb89e7e6fdb4dd029456be3`. It has 4,961 rows: 3,072 train and 1,889 test. The assembly's [root integration review v2](../../runs/v18-v19-native-soft-edge-review-20261009-v1/cumulative-full-assembly-v3/root-integration-review-v2.json) describes recurrence as exactly unchanged from the previously admitted cohort (zero added recurrence rows); the assembly itself grants no new recurrence admission or qualification.

The existing [recurrence structural audit](../../runs/v18-v19-native-soft-edge-review-20261009-v1/cumulative-full-assembly-v1/recurrence-structural-audit-v1.json), SHA-256 `36c1c4e23a0c64d2351f595bb90358c7081b579936c8d6ca54b6a7b2388b9d26`, reports 908 writer records, 898 reader records, and 1,020 structurally closed producer-to-consumer edges. It reports no missing or ambiguous producers, cross-split edges, cycles, or invalid selected-return bindings; maximum dependency depth is 6, maximum producers per reader is 3, and maximum consumers per writer is 7. An independent stream count by split found:

| Split | Rows | Writer rows | Reader rows | Producer-to-consumer edges |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,072 | 509 | 494 | 554 |
| Test | 1,889 | 399 | 404 | 466 |

Depth histograms, counting distinct records: train `{0:2586, 1:265, 2:115, 3:53, 4:34, 5:18, 6:1}`; test `{0:1488, 1:202, 2:97, 3:51, 4:32, 5:18, 6:1}`. The audit explicitly reports `selected_direct_typed_return_writers: 0` and labels its result structural-only, still requiring source/quality/causal-use checks. A closed structural graph is not by itself proof of useful consumer behavior or a new admission.

## ProgramReplay readiness and current limits

`training/neuralese/natlang_neuralese/train/graph_replay.py` defines the `ProgramReplay` substrate. `load_records` accepts only `schema: "natlang.replay-record/1"`; each record is a whole multi-call program with `arguments`, weighted `terms`, `producers`, and `blocks_dir`. `load_blocks` loads one-block sidecars and validates each block ID against its decoded content. `ProgramReplay._rewritten` reconstructs downstream values through producer calls when propagation is enabled, allowing downstream losses to reach upstream writers.

The cumulative recurrence rows are native teacher-turn rows, not `natlang.replay-record/1` rows with ProgramReplay's producer terms and sidecar block files. The native103 rows are also teacher-turn action examples, not replay records. No local materialized `natlang.replay-record/1` corpus or corresponding block sidecars were found in the bounded local run/tree search. The registry does list `neuralese-s5-operator-replay-records-20261009-v1` under `runs/neuralese-s5-operator-smoke-20261009-v1`, with 216 records (144 train from 24 cases; 72 held from 12 cases), but that registered path is absent locally. Its registry entry says it is smoke data with backbone-specific blocks for older LFM2.5-350M pilot-v4 heads, `training_admission: false`, and no task admission.

For a consumer E2E stage, use existing ProgramReplay-format records and exact compatible block sidecars if they are made available and separately reviewed, or build a shared explicit adapter from the existing trajectory evidence. Do not silently convert `PortRecord`, infer replay producers from text/typed-part presence, or treat the 1,020 structural edges as causal-use or training admission.
