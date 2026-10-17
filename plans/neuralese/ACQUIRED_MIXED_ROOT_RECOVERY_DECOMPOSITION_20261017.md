# Acquired source recovery across verified roots

## Parts

| Part | Form | Reason |
| --- | --- | --- |
| Choose which immutable location supplies one inventory path | Explicit `materialized_files` receipt entry | This records where exact bytes were recovered; the loader must not infer or search for alternatives. |
| Resolve the selected path and verify its containment, size, and SHA-256 | Crisp shared acquired-source verifier | Filesystem paths and bytes are exact external facts. |
| Read and normalize the acquired row | Existing shared acquired-source adapter | The recovery layout does not change source rows, parser semantics, split, or trajectory identity. |
| Decide source quality, protection, split, and admission | Existing source review and admission pipeline | Materialization is custody only and grants no training admission. |

## Ordered recovery behavior

1. Keep the embedded acquisition inventory and all original relative paths,
   sizes, hashes, and order unchanged.
2. Resolve a row from the primary source root unless the recovery receipt names
   an exact alternate-root label for that relative path.
3. Resolve the named root once, require it to be an existing directory, and
   reject any resolved file path that escapes that root. Never search another
   directory or accept an undeclared symlink target.
4. Verify each selected file against the original inventory before yielding
   rows. Read the same bytes through the existing acquired-source adapter.
5. Preserve the private-archive, embedded-inventory, held/split, quality, and
   admission checks. A mixed-root receipt proves recoverable custody only; it
   does not change which rows are usable or claim that a model trained on them.

This is storage-layout plumbing for sources that were already recovered in
separate verified operations. It adds no model-facing names or behavior.
