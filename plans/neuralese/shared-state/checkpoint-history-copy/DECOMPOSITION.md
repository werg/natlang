# Copy checkpoint history without repeated Python dispatch over atomic row maps

The owner requires recoverable optimizer/RNG/exposure and prompt return to useful
training. A live Mellum stack during a prolonged checkpoint pause reached Python
`deepcopy` from `collect_sft_progress`. This is a measured implementation cost,
not a reason to discard training history or weaken continuation requirements.

1. Preserve the existing continuation policy and its exact source obligations.
2. Copy metadata into independent mutable containers, preserving values, types,
   iteration order, aliases, cycles and custom-copy behavior. Bulk-copy exact
   builtin dictionaries/lists with immutable builtin children; delegate other
   types to the existing Python deepcopy protocol.
3. Keep optimizer/RNG/FP8 exclusions, pending-source merging and all exposure
   counters unchanged. Use one shared helper for both model lines.
4. Compare against deepcopy on realistic row/history shapes and actual checkpoint
   continuation tests. Record elapsed time and memory; do not infer an observed
   learner speedup from a CPU microbenchmark.
5. Before production adoption, validate exact restored full state and real
   checkpoint/continuation behavior. Declare the changed package honestly and
   preserve earned progress. A local helper is not a deployed improvement.

Steps2–5 are crisp copying, identity and durability mechanisms. The natural-
language policy remains comprehensive adaptive training with accurate history.
No model prompt, loss, graph dependency, numerical kernel or optimizer routing
changes are intended.
