# Whole-graph failure repair DPO

## Natlang-native policy

For each original, unhinted training task and exact world/oracle/source binding, compare the complete failed execution with a verified corrected execution. Admit the pair only when both complete native execution graphs are captured and custody is verified. Keep every producer, consumer, import, occurrence binding, forced input, and sampled action in each graph. Score all declared action tokens in one shared graph forward so producer actions receive gradients through downstream state use. Repair hints are evidence for the repair process only and never enter the student prompt. Exclude protected evaluation tasks and mismatched worlds or sources.

## Exact mechanisms

- Add a distinct genuine reference-policy DPO mode; preserve the old reference-free pairwise objective and its history unchanged.
- Bind one fixed reference checkpoint, tokenizer, runtime, graph/action digest, content receipt, and per-alternative log probabilities with explicit token-mean normalization.
- Admit only receipts binding the original unhinted prompt, same task/world/oracle/source groups, actual failed and repaired actions, source protection, and full native replay graphs.
- Keep graph producers and imports connected in a disjoint-union graph; do not flatten messages or detach producer state.
- Emit graph JSONL, objective JSON, and a content-addressed native source manifest through a runnable CLI. The result enters the existing `native_offline_preference` continuation using `--objective-continuation`, preserving weights, optimizer, RNG, and per-row exposure. New pair/objective rows append without rewriting old row bindings.

## Verification

Focused tests cover producer gradient dependence, reference score custody/math, hint leakage, held split and oracle/world mismatch, and dynamic append/restart binding. Generation, data publication, sandbox execution, and deployment remain with the supervising owner.
