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
   the exact LFM ancestry-retention plan. Archive apply is enabled in the
   reviewed policy under the standing owner authorization and root review. It
   rechecks owner, config, selected cursor/checkpoint, mount UUID and free-space
   floor before invoking the existing bounded verified-copy tool. Mellum cache
   pruning remains disabled until its first concrete eligible plan is
   independently reviewed.
   The tool independently rechecks open/mapped readers and source stability.
   External Mellum pins are inventoried and
   current, previous and pending full states are protected; older pins remain
   ineligible until their local bytes are independently reverified against
   their private receipts. Verification is per-file resumable: each completed
   8 MiB streaming SHA-256 is durably cached against exact inode, device, size,
   mtime, ctime, link count, allocation and receipt bytes. The first plan is
   limited to one verified eligible pin so review and cleanup can begin before
   scanning the remaining cache. The latest private object is checked through
   the authenticated SDK without downloading it.
2. `checkpoint_safety.py` supplies the shared checkpoint admission primitive.
   Each writer computes a tensor-storage upper bound from the state it is about
   to save, acquires a per-filesystem reservation lease, checks the bound plus
   configured reserve before the first or later atomic write, and releases the
   lease only after the new checkpoint is durable. Existing target size alone
   cannot size a first save. One lease per filesystem serializes checkpoint
   writes, reserves concurrent-transfer blocks on that filesystem, and
   rechecks space after acquisition; model updates continue outside the lease.
   The currently running frozen B1 learner has not been rebound to this guard.
   Its manager status therefore reports an advisory estimate and explicitly
   records `live_writer_guard_installed: false`; no current runtime qualification
   is claimed for first-write admission.
3. The manager uses `local-checkpoint-retention.py --plan` for LFM. Only an
   exact, currently reviewed plan may reach its apply operation. It never
   implements a second checkpoint ancestry or ownership algorithm.
4. Closed retained LFM artifacts are selected by the existing
   `archive_to_external.py` eligibility checks and moved with its existing
   bounded, fsynced, read-back-verified copy and atomic symlink swap. The
   manager caps each archive batch at 16 GiB and to current external free space
   minus the 80 GiB floor, a 1 GiB copy margin and any exact pending Mellum
   local-copy size. Source paths remain resolvable; no hardlinks are created.
5. Old Mellum external full-state mirrors are eligible only when exact local
   full checkpoint and private export identities are currently verified,
   current and previous-local recovery parents are protected, and the candidate
   digest appears in the authenticated current checkpoint's
   `native_snapshot_ancestors` for that exact Mellum model identity and run.
   The current checkpoint metadata is read without materializing tensor bodies
   through a version- and source-hash-bound metadata reader. Its exact ancestry
   is recorded in the plan. No configured protected path or live fd/mapping may
   name a candidate. A one-candidate deletion plan is durable before unlink;
   the manager does not remove pins from the CoreWeave sandbox. Pruning remains
   disabled in the checked-in live policy until independent review of the
   actual plan and lineage binding.

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
