# Keep immutable source meaning across identical file replacement

Policy: continue useful training when an operational copy/touch changes only file
metadata. A different source still requires explicit admission/continuation.

1. Retain the validated source SHA, row bindings, split, weights and index key.
2. On the existing metadata-guard mismatch, hash the complete current file with
   the uncached SHA helper. Require stable device/inode/size/mtime/ctime across
   that hash and equality with the original source SHA. Otherwise reject.
3. Update only the in-memory file metadata binding, then retain all existing
   parsed-row checks. Check manifests after child streaming as well as before.
4. Reuse old indexes only through a separately verified AST-limited runtime
   extension. All parser/schema/compiler dependencies and index metadata remain
   exact. No source-policy whitelist or corpus-wide rescan is substituted.

The guard and continuation certificate are crisp exact byte verifiers; the
supervisor's decision to admit changed data remains its existing declared policy.

Initial uncached parsing and manifest assembly retain strict five-field identity
stability: A-to-B-to-A replacement during construction cannot authenticate newly
parsed B rows under A’s original SHA. Relaxed metadata revalidation applies only
to previously authenticated cached entries, retained preparation and iteration.

Validation: `native-source-metadata-race-final2-20261016` passed 89 focused
source, resident, preparation and exact-continuation tests (7.66 s, 620 MiB).
This includes genuine initial A→B→A rejection, stable identical replacement,
changed bytes with restored mtime, and replacement racing fresh hashing.
