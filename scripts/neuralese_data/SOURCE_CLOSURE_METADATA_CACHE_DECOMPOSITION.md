# Authenticated source-closure metadata reuse

This is an exact cache for the existing source-closure verifier, not an admission
policy. Existing natural-language source, quality and held decisions stay with
that verifier and its unchanged protected-source authority.

1. The existing verifier produces the full referenced-artifact inventory once.
   This remains the original implementation and executes every original check.
2. A crisp helper retains an isolated copy within that Python process. Its key
   binds the actual manifest file, its supplied parsed value and cache mode.
3. Before reuse, authenticate every referenced artifact against the original
   SHA-256 and size using the existing strict five-tuple hash cache. Check the
   resolved path and pre/post device, inode, size, mtime and ctime as well.
   Retain each original reference-to-target edge, including pre-existing aliases,
   derived source/complete paths, and declared receipt references; reject reuse
   if any edge retargets while its old target remains intact. The exact bytes
   of complete.json used to capture those edges must match its final producer
   binding; a first-call metadata rewrite cannot leave uncaptured aliases.
4. Return a copy only while all identities still match. Otherwise call the
   original producer again; the normal verifier decides whether its resulting
   inventory remains acceptable. Explicit full-validation calls never reuse.
5. Restore the original function on context exit. This opt-in external helper
   does not edit the verifier, the seven parser-policy files, protected-source
   receipts, active trainers, or their frozen runtime packages.

The mechanism is crisp because byte identity, function lifetime and cache
invalidation must be exact. Sequential admissions can use one context to avoid
reopening thousands of immutable export JSON files. A new process still performs
its first normal validation; this is not a persistent certificate or scan bypass.
