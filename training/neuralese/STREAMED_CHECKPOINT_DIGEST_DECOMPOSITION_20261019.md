# Avoid rereading each newly written full checkpoint

The supervisor keeps full, recoverable saves while reducing idle GPU time.
Checkpoint integrity and atomic durability are crisp mechanisms shared by all
trainers; no model-facing instruction or training policy changes.

1. Hash exactly the bytes successfully written by the existing atomic writer.
   Reject partial or nonsequential writes rather than publishing a digest.
2. Keep the existing file fsync, atomic replacement and directory fsync. Bind
   the resulting file to its actual device, inode, size, mtime and ctime.
3. Publish that writer-computed SHA through the existing hash-cache protocol,
   for both strict content identity and ordinary callers. Never infer a digest
   from an archive receipt or from tensor values.
4. Verify both cache lookups against an independent byte hash and reload actual
   model/optimizer/RNG data; exercise overwrite, write failure and mutation.

Frozen cloud runtimes require an explicit reviewed deployment; landing this
shared mechanism alone does not claim it has accelerated an active learner.
