# Continue useful teacher work as the catalog changes

Policy: preserve the currently qualified generation queue. Previously issued raw
tasks and explicitly quarantined derivatives must not be rediscovered as new
tasks. A pending packet must not prevent independently qualified later packets
from supplying the teacher. Qualification, held-data protection and trajectory
admission remain separate requirements.

1. Read the current ready prefix, original raw dedup inventory and explicit
   quarantined packet inventory. Authenticate every packet and its actual cases.
2. Preserve the ready prefix in its exact order, including independently authored
   or derived tasks absent from the raw catalog. Record historical variants that
   are absent from the current catalog instead of requiring a clean restart.
3. Exclude the union of those variants from new catalog work. Select every
   remaining eligible task through the existing shared eligibility query and
   retain all current provenance/protection requirements.
4. Qualify new packets through the existing exact reference verifier. With
   independent packet scheduling explicitly enabled, append each qualified packet
   in source order; keep zero-qualified packets as pending debt. Publish a bound
   candidate inventory whose prefix exactly matches the ready entries. Retain
   original candidate ordinals and the complete original inventory for accounting.
   Bind current protection, decision holds and reference runtime through an
   explicit qualification-authority artifact. Historical prefix runtimes remain
   independently bound; the first historical packet does not choose the runtime
   for newly executed references. Pending and unprocessed packet counts remain
   part of the complete candidate inventory.
5. Adopt only the completed, verified successor through the existing source
   sequence mechanism. No teacher engine change or training-admission credit is
   implied by catalog preparation or reference qualification.

Steps 1–4 extend `teacher_reference_append`, the existing mechanism. Crisp code
implements exact byte identities, set membership, SQLite eligibility, durable
publication and verifier invocation. These are deterministic custody operations;
the supervisor decides which explicit inventories to retain or quarantine.
Default strict legacy preparation and qualification remain supported. Production
uses original169 raw packets for dedup, the current137 ready packets as retained
prefix, and the38 blocked grounded packets as explicit exclusions. This excludes
all2393 problematic originals through the original raw inventory and all2369
blocked derivatives through their packet proofs, while preserving the useful
120 authored and24 diagnostic pilot tasks. Newly eligible catalog tasks require
the same reference qualification before generation and verified trajectories
before training admission.
