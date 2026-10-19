# Bounded source receipts

Policy: preserve every admitted source identity and exact whole-source group
totals, without repeating those totals at every historical manifest ancestor.
This changes diagnostic metadata only; source hashes, row bindings, exposure,
optimizer state, RNG and numerical training remain unchanged.

1. Traverse the complete source identity tree, retaining each path, SHA, row
   count and communication count. Sum manifest communication counts from its
   children rather than scanning every ancestor's accumulated entries.
2. Count source groups once over the selected root's exact entries. Record the
   `root-only/1` scope explicitly. Nested identities omit redundant group maps.
3. Use these bounded receipts in shared SFT checkpoints/reports and foundation
   evidence. Existing report compaction still backs the root group detail by
   the actual checkpoint. Original sources remain available for subtree detail.
4. Give the existing classification verifier its exact entries and complete
   source identity tree through an explicit read-only consumer view. Its
   producer module and parser-policy identities remain unchanged, so this
   metadata optimization does not invalidate authenticated execution caches.

All four stages are crisp inventory/durability plumbing. They make no source
selection, admission, completion or quality decision. Deep-manifest tests compare
the original identities and root counts, bound row traversal to twice the selected
rows, and exercise the actual classification verifier through both interfaces.
