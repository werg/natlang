# Reuse exact index metadata certificates

Continuation policy remains the existing recipe and objective verifier. It still
checks every source, row identity, topology class, held split and withdrawal.
This change is a crisp authentication optimization, with no training policy.

1. Resolve the declared index and record device, inode, size, mtime and ctime.
2. Authenticate its declared content SHA and optional byte count normally.
3. Load its pickle afresh. Keep no loaded row objects between calls.
4. Compute its canonical entries digest unless the exact resolved path, content
   SHA and five stat fields already have a successful digest certificate.
5. Require the loaded metadata digest to equal that certificate. Reject changes
   of the declared path target or stat fields during checking.
6. Keep at most 1,024 compact certificates; eviction only repeats work. Continue
   all caller-specific key, source, row-count and topology checks on every call.

The cache is process-local and stores only identities and digests. Intrinsic
classification and row-index code are unchanged, so source admission policy and
the seven-file index policy do not change. Running numerical packages are not
patched in place. Verify reuse, changed declared authority, same-mtime mutation,
inode replacement, metadata corruption and changes during loading.
