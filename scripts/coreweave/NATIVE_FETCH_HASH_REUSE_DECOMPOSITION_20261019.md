# Reuse verified immutable payload hashes

The supervisor selects which privately verified checkpoint to evaluate. This
change is only an exact file-verification mechanism, with no policy or
model-facing change.

1. Keep streaming and interrupted-download hashing unchanged: resuming a prefix
   needs the incremental SHA state.
2. For complete files, use the existing shared content-verified SHA cache,
   keyed by device, inode, size, mtime and ctime, with archive receipts disabled.
3. Reuse that same verifier when exposing a complete payload through a logical
   symlink. An unchanged payload need not be read twice in one fetch.
4. Reject a content mismatch and invalidate cache reuse after replacement or
   even a same-size rewrite with restored mtime. Keep independent private
   receipt verification and the transfer/storage floors unchanged.

No trained model, source data, private object or in-flight transfer is changed.
