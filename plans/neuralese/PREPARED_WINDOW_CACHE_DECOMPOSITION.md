# Prepared text window cache

## Goal

Reuse exact tokenized direct-SFT windows across safe resumes and recipe
handoffs. The cache removes repeated tokenizer work and window construction;
it does not change the training objective, sampling schedule, held split, or
model state.

## Natlang-native decomposition

1. **Name the prepared corpus.** `prepared_windows.cache_spec` receives the
   source text file, tokenizer, rendering controls, masks, recovery selection,
   edge spans, and window settings. It returns one semantic identity for those
   exact preparation inputs. This is a crisp verifier because identity and
   content hashes must be exact; it does not inspect or select model behavior.
2. **Reuse a completed preparation.** `prepared_windows.load` checks the
   identity, manifest, shard hashes and descriptor bounds, then returns the
   same train/test windows over read-only mapped token and role bytes. This is
   exact cache plumbing; any mismatch returns a miss.
3. **Prepare when needed.** `direct_sft.stage_data` continues to call the
   existing `stage_rows`, `prepare_text_windows` and held-window selector on a
   miss. `prepared_windows.save` records that exact result in bounded shards
   and publishes the manifest atomically. Preparation remains the source of
   truth.
4. **Use the result.** `stage_data` derives target-bearing and held windows
   through the existing helpers on both hit and miss. Training schedule and
   continuation logic stay in the existing shared trainer.

The cache key uses explicit preparation policy versions, not a package-wide
code digest. Bump the relevant policy version when tokenization, role parsing,
masking, window geometry, row filtering or merge semantics change. Changes to
optimizer, kernels, evaluators, logging or unrelated code do not invalidate
prepared data. No model-facing text changes.

## Exact data retained

Each descriptor points into one content-keyed token shard and optional role
shard. It retains all window fields, including supervision spans, prefix,
offset, document identity, cohort, source groups, sampling group and merged
row identities. The cache also retains row metadata needed to recover declared
context-only spans, plus mask receipts. The held-window selection is rebuilt
from the restored test windows with the normal selector.

## Failure and recovery

A missing entry, unsupported tokenizer fingerprint, digest mismatch, corrupt
metadata, wrong schema, invalid offset, or cache I/O error causes the existing
preparation path to run. A write is built in a private directory; the manifest
is written last and the complete directory is renamed into place. A concurrent
winner is kept. Cache failure cannot fail training.
