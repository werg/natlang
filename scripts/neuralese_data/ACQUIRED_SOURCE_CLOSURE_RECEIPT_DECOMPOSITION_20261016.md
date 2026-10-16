# Reusable acquired-source closure receipts

The D24 preference builder currently reparses every normalized full-session
source for every model/view invocation just to re-prove the same transitive
held-group closure. Policy stays in the existing protection projection and
source-review files; this change adds one exact, durable proof artifact for
that shared closure.

1. A full closure pass still streams every normalized source, validates each
   export receipt and complete inventory, computes connected groups, and
   compares the result with held, quality, and pending protection.
2. At that boundary, `write_acquired_source_closure_receipt.py` writes a
   durable receipt binding the exact
   projection manifest, inherited component/projection files, all normalized
   export receipts and source bytes, each `complete.json` inventory, and the
   resulting accounting/component digest.
3. A later model/view process may reuse the receipt only when it revalidates
   the current protection sets and hashes every bound input. It checks the
   exact input set, source/export relationships, closure accounting, and
   stored component projection. Any changed/missing/new input rejects reuse.
4. Preference conversion and admission continue their per-view source,
   decision-hold, row, and target checks. The receipt replaces only repeated
   parsing for global connected-source closure; it does not waive those checks.

The receipt is explicit on the CLI so separate processes can share it. The
default path remains the existing full verification behavior. Exact hashing
is the freshness boundary; no path-only or process-local cache is accepted.
