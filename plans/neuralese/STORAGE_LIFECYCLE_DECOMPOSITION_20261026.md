# Continuous checkpoint storage lifecycle

## Natlang-native policy

Keep useful training moving while maintaining enough space for the largest
atomic full-state write and concurrently pending checkpoint transfers. Keep at
least 80 GiB free on the external volume. Use the existing ancestry-aware LFM
retention verifier for superseded operation states; move retained closed files
through the existing copy, verify, and logical-link archive path. Preserve the
current full local parent, latest private export, optimizer/RNG/exposure state,
source and held data, protected evaluations, and every live reader. A missing
or changed mount identity, stale owner, failed copy, or failed read-back blocks
that action and records a repairable reason.

## Functions and exact mechanism

1. `continuous_storage_manager.py` is a five-minute systemd timer under the
   memory ledger. It reads the reviewable policy, current owner/config/cursor
   and recovery records, mounted-volume identity and disk availability, then
   emits an atomic status and fsynced journal event. When headroom is low, it
   requests a root-scoped dry-run from the existing archiver and references
   the exact LFM ancestry-retention plan. The archive apply path is separately
   policy-gated (disabled in the checked-in live policy); when enabled after
   review, it rechecks owner, config, selected cursor/checkpoint, mount UUID and
   free-space floor before invoking the existing bounded verified-copy tool.
   The tool independently rechecks open/mapped readers and source stability.
   External Mellum pins are inventoried and
   current, previous and pending full states are protected; older pins remain
   ineligible until their local bytes are independently reverified against
   their private receipts.
2. `checkpoint_safety.py` supplies the shared checkpoint admission primitive.
   Each writer computes a tensor-storage upper bound from the state it is about
   to save, acquires a per-filesystem reservation lease, checks the bound plus
   configured reserve before the first or later atomic write, and releases the
   lease only after the new checkpoint is durable. Existing target size alone
   cannot size a first save. One lease per filesystem serializes checkpoint
   writes, reserves concurrent-transfer blocks on that filesystem, and
   rechecks space after acquisition; model updates continue outside the lease.
3. The manager uses `local-checkpoint-retention.py --plan` for LFM. Only an
   exact, currently reviewed plan may reach its apply operation. It never
   implements a second checkpoint ancestry or ownership algorithm.
4. Closed retained LFM artifacts are selected by the existing
   `archive_to_external.py` eligibility checks and moved with its existing
   bounded, fsynced, read-back-verified copy and atomic symlink swap. The
   manager caps each archive batch to current external free space minus the
   80 GiB floor and a copy safety margin. Source paths remain resolvable; no
   hardlinks are created.
5. Old Mellum external full-state mirrors are eligible only when exact local
   full checkpoint and private export identities are currently verified,
   current and previous-local recovery parents are protected, same-model
   ancestry is proven, and no config, receipt, process, fd, or mapping names
   the candidate. Each deletion plan is durable and explicit; the manager does
   not remove pins from the CoreWeave sandbox.

## Tests and operation

CPU tests cover sizing for first writes and concurrent transfers, one-writer
lease contention, live readers and named dependencies, copy interruption,
source mutation, failed read-back, mount absence/identity change, external
floor enforcement, failed LFM retention, Mellum recovery/private receipt
currency, and idempotent status recovery. A passing test or prepared plan is
not live protection. The timer is installed and run under the memory ledger; its
durable status shows the current owner and mount identity, internal headroom,
recovery parents, or an explicit blocker. The archive plan is root-scoped and
dry-run only. Root reviews any retention plan whose apply removes a checkpoint
or moves a retained artifact.
