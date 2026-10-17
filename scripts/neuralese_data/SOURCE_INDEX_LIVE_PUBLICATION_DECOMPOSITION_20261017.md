# Preserve source indexes produced by a live trainer

The supervising policy is: reuse a previously authenticated index only under an
existing exact source-validation equivalence proof. Keep every destination that
already exists, including one published while continuation is running. Accept
it only when its complete decoded value equals the qualified continuation.
Conflicting values stop provisioning; they never authorize replacing training
state or restarting the trainer.

1. The existing proof functions decide exact old/new policy equivalence. Their
   pinned files, parser checks and source admission remain unchanged.
2. The existing continuation verifier checks the old key, complete entries and
   metadata digest, changing only the policy key.
3. A crisp filesystem helper serializes a complete index into a private file,
   flushes it, then publishes it with an atomic no-clobber link. It removes the
   temporary name immediately; no mutable shared alias remains. A racing winner
   is compared in full and retained. Directory fsync makes publication durable.
4. Existing destination values are checked without serialization or replacement.
   Every output still passes the existing complete-value check and SHA receipt.

This is standalone cache provisioning, not a model-facing change. It does not
modify the row-index writer used by training or any of the seven source-policy
files. A trainer may independently publish its own index; continuation never
replaces that file. Tests inject equal and conflicting winners precisely at the
publication syscall, checking retained inode/bytes and temporary cleanup.
