# Luna prepared-campaign fairness (2026-10-15)

Seed 52073 had zero finished jobs while seed 52074 held four of six active
worker slots. The replenisher had prepared 52073 under memory pressure, then
created 52074 once slots briefly looked free. Its per-campaign completion pass
attempted 52073 first but deferred those workers for RAM; new-seed preparation
then filled the slots with 52074. This can repeat across seeds and obscure
older ungenerated work.

The replenisher now checks for unfinished, inactive workers in prepared
campaigns before creating another seed. It prefers those workers when capacity
returns. A worker with four no-progress relaunches is bypassed so that an
investigation does not halt unrelated generation. The global six-worker cap and
memory-ledger admission remain unchanged. This is queue plumbing, not a change
to task selection, model-visible instructions, or training admission.
