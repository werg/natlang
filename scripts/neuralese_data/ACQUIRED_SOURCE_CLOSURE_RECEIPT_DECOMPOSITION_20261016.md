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
   resulting accounting/component digest. It also records exact code-file
   bindings for the closure verifier, session parser, connected-group closure,
   current-protection/source-review policy, JSONL I/O, and hashing.
3. The first builder given that receipt path acquires the existing output
   lease, performs the full pass, and atomically publishes the receipt before
   rendering. Later model/view processes reuse it only when they revalidate
   the current protection sets and hashes every bound input. It checks the
   exact input set, source/export relationships, closure accounting, and
   stored component projection. Any changed/missing/new input rejects reuse.
4. Preference conversion and admission continue their per-view source,
   decision-hold, row, and target checks. The receipt replaces only repeated
   parsing for global connected-source closure; it does not waive those checks.

The receipt is explicit on the CLI so separate processes can share it. The
default path remains the existing full verification behavior. Exact hashing
is the freshness boundary; no path-only or process-local cache is accepted.

The current-protection component now records the exact scoped AST law of
`current_protection` and `file_rows`, including recursively referenced globals
and import aliases. The shared identity helper remains a physical code-file
dependency. Every other closure implementation and every protected/source
artifact remains bound as before. An unrelated converter entry point, cache or
renderer edit therefore does not change this protection law; a protection
function, referenced constant/import or external policy change does. A legacy
whole-file receipt requires an explicit independently reviewed transition before
reuse under the scoped representation. Physical converter provenance remains
available in the admission/conversion receipts; the semantic digest is named
`policy_sha256` and is never presented as the SHA of a physical file.
