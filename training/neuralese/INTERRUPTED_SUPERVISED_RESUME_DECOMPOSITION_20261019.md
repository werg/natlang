# Continue earned work after an intentional supervised interruption

The supervisor chooses a useful correction and stops the old controller after
its worker has saved. A failed operation remains failed. Crisp mechanisms:

1. Authenticate its saved-update ACK, complete parent/recipe/model/process
   ancestry, and retained failed provider result. An incomplete final operation
   may be inspected solely to resume it; it never qualifies a handoff.
2. Hold both existing controller leases. Preserve cycle, stage position, earned
   previous stages, optimizer/RNG payload and all recipe debt. Publish the exact
   independently pinned checkpoint and clear obsolete checkpoint readiness.
3. Advance only the retry namespace, so the next same-stage operation has a new
   request with the earned parent. Preserve the old operation's immutable files.
4. Project only its already verified adopted source recipe through the existing
   resident mechanism. Operational tuning follows the normal explicit boundary.
5. Durably record exact before/candidate files. Recover a partial commit by CAS;
   a historical completed retry must never rewind subsequently earned progress.

This shares the actual scheduler, source projection and ancestry verifier for
both models. No stage, task, runtime or whole-recipe qualification is granted.
