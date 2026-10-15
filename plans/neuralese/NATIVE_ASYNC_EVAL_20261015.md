# Native LFM historical task evaluation on DGX

The existing periodic private exporter continues independently on CoreWeave.
The existing task dispatcher can fetch its newest independently verified LFM
checkpoint and run the protected held packet on the DGX GPU. Both cloud training
processes remain active. This path intentionally does not fetch Mellum's 47GB
checkpoints or modify the exporter's cursor/service.

After source review, launch from the canonical checkout through the ledger:

```sh
python3 scripts/memory_ledger.py run --unit native-lfm-async-dispatch --budget-gb 4 --priority low -- \
  .venv-neuralese/bin/python scripts/task_eval_dispatch.py --every 60 \
  --native-export-state /home/werg/data/dgxops/coreweave-native-lfm-periodic-export-state-20261015.json \
  --native-eval-spec training/neuralese/eval-configs/native-lfm-dgx-periodic-20261015.json \
  --native-eval-out /home/werg/data/dgxops/native-lfm-async-eval \
  --native-base /home/werg/data/dgxops/qad-drift-probe/base
```

The dispatcher admits each evaluation through its existing separate memory
ledger unit. Owner-host W&B authentication remains in the existing SDK Python
(`--native-fetch-python`); no account credential goes to a cloud sandbox.
Requests use the existing task-eval queue. At most one native snapshot is
admitted at a time: a newer export never cancels an active historical eval.
After its result and snapshot drain, the newest export is selected; intermediate
exports coalesce. Restart preserves the active request and exact file identity.

Reports are in `/home/werg/data/dgxops/native-lfm-async-eval/latest-result.json`
and `checkpoints/<original-checkpoint-sha>/task-eval.jsonl`; detailed existing
collector outputs remain in that checkpoint's `task-eval/step-*` directory.
Inspect dispatcher logs for ingest/ferry errors and the existing per-request
`dispatch.log` for execution/retries. Infra/unresolved results have null task
success/rate and preserve observed partial successes separately. Pending work
has no outcome. Failed exhausted dispatches are unscored, never model failures.

Each request/result binds original full checkpoint SHA/bytes, saved model and
precision configuration, actual tokenizer/base/native embedding identity,
compiled runtime/npm/protection closure, held packet/case indices, prompt mode
and sampling settings. Restore uses the shared native checkpoint loader without
legacy heads or implicit precision upgrades. Runtime/protection changes during
collection invalidate that historical result. Each new request binds the then
current runtime; comparisons must account for different runtime identities.

All reports explicitly say `historical=true` and
`qualifies_current_weights=false`. They do not satisfy final exact-weight
qualification, measure newer weights, or switch off existing cloud task checks.
The provided protected packet spec evaluates ordinary task capability with the
native backbone. It does not claim cross-model semantic-channel quality.
`neuralese_server=true` optionally routes the existing native harness transport;
no new model-facing prompt/error wording is introduced. Shared/participant
checkpoint exports are not accepted by this initial full-LFM export path until
the exporter supplies that format and its exact dependency closure.

Validation: focused tests exercise real tiny native checkpoint restore, exact
logits, existing TaskEval result/drain, sampler/runtime tamper rejection, private
download restart safety, infrastructure null results, and no cancellation when
newer exports arrive. Fixture scoring is not production task success. Actual
full-size restoration evidence is recorded separately under
`/home/werg/data/dgxops/native-async-eval-validation-20261015/`.

Actual full-size check, 2026-10-15: the existing private ferry downloaded and
verified LF2750 (2,053,480,587 bytes, SHA
`c2396948f084a7ca28f6fdbdad3c0ba1c7600cf5a986a9c55c8572ff9b3d7e66`)
in 349.8 seconds. The new sealed-request native branch then restored it on the
DGX CUDA device and produced finite `[1,12,65536]` logits in 34 seconds, with
3.8 GiB process memory peak. Its saved FP32/IEEE arithmetic and legacy sparse-row
configuration remained unchanged; this historical pre-repair checkpoint was not
upgraded. The exact proof is `actual-restore-proof.json` in the validation root.
No held task collection or model-lift claim follows from that forward check.

The first attempt exposed an unrelated shared inventory startup cost: cached
hash lookups scanned a 33MB append-only global cache for every small runtime
file. The shared runtime inventory now streams those small files directly using
the existing SHA helper, retaining exactly the same inventory keys and digests.
Large model/checkpoint hashing still uses the existing cache.
