# Reuse independently verified private checkpoint exports

The owner wants useful training promptly and minimum unnecessary data movement.
Manual handoffs and periodic durability share that policy. The existing exporter
remains responsible for source hashing, scoped private PUT and independent full
GET; no new upload implementation or scheduling policy is introduced.

1. An exporter identifies the exact source bytes and provider/owner scope.
2. Serialize exporters of those bytes using a process lock. Distinct artifact
   paths may name identical content, so do not key the lock on its source path.
3. Reuse only a complete receipt whose SHA/size and independent full GET agree,
   and whose private object still exists according to the authenticated provider
   metadata check. A network error is not evidence of absence.
4. Otherwise run the existing exporter and publish its complete verified receipt
   atomically. Exceptions release the lock; unfinished uploads never earn reuse.
5. Preserve the original receipt and upload source. A caller using another local
   path records an explicit equivalent-source binding rather than rewriting the
   upload history. Actor qualification still checks its own required name and
   checkpoint provenance.

This is crisp external-service/durability plumbing. Test simultaneous independent
processes (one actual export), provider deletion, incomplete verification and
source-identity isolation. Adoption in the existing owner-host exporter changes
future calls; already running uploads complete under their existing ownership.
