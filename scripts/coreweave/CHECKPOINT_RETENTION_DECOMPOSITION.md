# Keep only useful private checkpoint exports

Owner policy (2026-10-16): delete everything except what is necessary to carry
on. A successful periodic export replaces obsolete recovery snapshots; it does
not create a permanent archive. Explicit live diagnostic and handoff consumers
may retain additional named snapshots. Useful training/source and held data stay.

`chooseCheckpointExportsToDiscard(verifiedLatest, liveConsumers, remoteFiles)`:
1. Require the latest full checkpoint's private upload and independent complete
   readback to agree on bytes and SHA-256.
2. Require that exact latest file to exist in the declared run.
3. Keep the latest file and each explicitly named live consumer's file.
4. Keep all source bundles, inference exports and unrecognized names.
5. Keep other files at the same step and steps newer than the verified latest;
   they may be in-flight uploads. Only an explicit discard decision can remove
   a different-hash snapshot at the same step.
6. Discard only older, recognized full checkpoint exports in this same run.

This policy is supplied through the supervisor's explicit policy file. Crisp
code authenticates the receipts, implements the exact name/set comparison,
records each deletion intent before mutation, calls the external storage API,
and verifies metadata absence. It does not decide whether a diagnostic is useful
or whether a source dataset should be discarded. Those decisions remain with
the supervisor and are recorded as explicit consumer names.

One mechanism serves both students. The existing export process commits its
verified state; the independent retention process reads it without interrupting
uploads or model training. Failed or incomplete exports cannot replace recovery
authority. Polling and the local ownership lock are operational plumbing.
Each storage run is bound to one canonical policy path in the host's shared
CoreWeave state directory, with a per-run lock independent of journal location.
A conflicting policy is rejected. Cross-machine ownership continues to use the
project's coordination protocol; this host-local lock is not a distributed lock.
Dry runs and invalid recovery receipts do not acquire durable policy ownership.
The deletion intent preserves the exact policy digest, protected names and full
verified-export receipt, and each success record is fsynced.
