# Private teacher append preparation

`prepare_private_teacher_append.py` prepares source, CE, capacity, recipe and
future-config bindings for an already admitted private teacher export. It never
deploys the next config or changes a live operation.

Run it with the model line's project venv and frozen runtime import root:

```sh
NATLANG_ROW_INDEX_CACHE=/actual/learner/cache/row-index \
NATLANG_HASH_CACHE=/actual/learner/cache/hashes.jsonl \
python scripts/neuralese_data/prepare_private_teacher_append.py \
  --descriptor /path/to/private-teacher-append-descriptor.json
```

Use the selected learner's actual production cache paths. The frozen readers
still authenticate source, tokenizer, role and parser-policy identities before
reusing an index. Omitting these environment variables can reparse the entire
existing production population unnecessarily. Keep disposable test caches
separate from these production caches.

The descriptor has this shape:

```json
{
  "schema": "natlang.private-teacher-append-descriptor/1",
  "runtime_root": "/frozen/runtime/repo",
  "artifact_repo": {
    "root": "/artifact/repo",
    "registry_path": "/artifact/repo/training/neuralese_corpora.json",
    "manifests_root": "/artifact/repo/training/corpus-manifests"
  },
  "supervision": {
    "root": "/supervision",
    "config_path": "/supervision/config.json",
    "plan_path": "/supervision/plan.json"
  },
  "ready_export_root": "/exports/ready",
  "model_role": "lfm25",
  "output_root": "/staging/append-v14",
  "stage_id": "native_full_sft",
  "append_key": "private_teacher_append_20261026",
  "private_receipt_custody": {
    "source_map": {"path": "/transport/.natlang-source-map.json", "sha256": "<64 hex>"},
    "transfer_manifest": {"path": "/transport/transfer.json", "sha256": "<64 hex>"}
  }
}
```

For transported exports whose `ready.json` still names a receipt on another
machine, bind the exact staged `.natlang-source-map.json` and `transfer.json` in
`private_receipt_custody`. The source map must use
`natlang.staged-source-map/1`; each original `private-receipt.json` path must map
to a staged file with a matching transfer-manifest `original`, `path`, SHA256 and
byte count. The transfer manifest's model role must match the descriptor. The
preparer copies both custody files into the output, checks staged receipt bytes
against the transfer binding, validates the full receipt against `ready.identity`,
and binds declared and actual paths plus both custody manifests in the census
and selection proofs. It does not rewrite ready records or change a global path
alias.

The tool snapshots all `ready.json` files present at start; later arrivals remain
eligible for a later append. It validates registry admission, complete manifests,
role receipts, tokenizer/history settings and private export receipts. LFM
requires explicit `keep` history behavior. Mellum inherits the selected stage or
model default and does not receive an implicit LFM override.

Existing source bytes win exact source SHA/size matches. Within other sources,
rows are deduplicated only by exact graph ID and binding; conflicts fail. Partial
duplicates are filtered with the shared JSONL reader/writer and reloaded through
`NativeGraphSource`. No graph is truncated. Any graph outside the existing
capacity envelope fails with its identity and measured geometry so the operator
can preserve it and obtain a real larger-case qualification.

The output contains `prepared.json`, the exact source census and selection
proofs, updated native source tree, CE policy, capacity requests, recipe,
`next-config.json`, and a byte-identical copy of the active plan. The CE policy
retains its old source coefficients, held contract, predecessor CE binding and
inventory. The recipe retains the predecessor recipe and full append record in
append history. A reviewer must check the output before the separately supervised
future-only config/plan CAS adoption. Preparation does not establish learner
exposure, GPU qualification or task lift.
