# Complete derivative inventory

The qualification instruction is: compare every declared production trainable
and the input embeddings against the cached reference, preserving model state.

1. Use the shared production inventory to prove that optimizer names cover
   exactly all trainable model parameters, with unique names and identities.
2. Compare those parameters and one separately named input-embedding derivative.
3. Report the parameter count and comparison count separately. Counts describe
   the exact inventory; a historical scalar count never selects the cohort.
4. Validate any reported counts against the actual inventory and comparisons.
   Older proofs without the additive count fields retain their existing checks.

These are crisp identity checks and report plumbing. The existing complete
gradient comparison, four-role exercise checks, tolerances and production
training policy remain the single implementations of their concepts. Mellum's
28-layer production topology has 308 backbone parameters and six learned
interface parameters: 314 parameters and 315 derivative comparisons.
