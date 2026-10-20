# Exact capacity-index validation reuse

The training supervisor's natural-language stage prepares each independently
useful capacity case and keeps actual numerical/gradient/capacity gates. Its
metadata policy is to reuse a completed exact index validation while the whole
witness and underlying file remain unchanged, rather than reread millions of
row entries for every case.

1. The existing request resolver authenticates every logical request/alias.
2. The existing check_index_witness_union gathers complete witnesses, retaining
   every conflict; the normal _check_indexes verifier validates each distinct
   witness. This remains the only content/key/entry-digest verifier.
3. A call-local crisp receipt remembers only the complete witness digest and
   actual resolved file identity (device, inode, size, mtime, ctime), plus the
   expected active row-index cache path. File identity must be stable across
   the strict check. Every later use re-stats before trusting that receipt.
4. Any new/conflicting witness, changed file or cache-root change takes the
   normal strict verifier again. No proof, source parser, numeric execution,
   model state, optimizer/RNG/exposure or capacity-certificate law changes.
5. The receipt lasts only for this owned run_prepared call; it is neither a
   durable training certificate nor permission to skip logical request checks.

This is exact verification plumbing, not a task selection or resource policy.
No model-facing strings change. All intrinsic seven parser/classification files
remain unchanged. The same mechanism serves both model ports.
