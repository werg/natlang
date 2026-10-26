# Facility complaint triage design

The dispatcher keeps informal location aliases attached to the official asset location rather than replacing it. It links complaints by location and mechanism evidence, schedules observations during the conditions when the issue occurs, and binds each visit to current access consent. A failed-access visit is recorded and renegotiated; it is never treated as entry. Recurring faults retain their former work-order history and escalate to mechanism investigation. Updates to occupants are generated separately for each recipient from consented fields, keeping reporter identity out of shared case notes.

Visit completion flows back through an append-only receipt ledger. It preserves no-access, cancellation, partial work, and unknown outcomes separately, checks that the receipt names the planned official location and issue set, and leaves recurring complaints open until technical and occupant or objective verification support follow-up. A replayed receipt cannot close work twice.


## Nested evidence interpretation (2026-10-26)

Complaint linking compares each report against the original observation and known repair mechanisms, preserving uncertain links and privacy partitions. Work-order formation receives the actual symptom and observation windows that its tasks must address, and dispatch remains blocked when those prerequisites cannot be specified.
