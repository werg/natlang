---
args: {"register": "WaiverRegister", "revocation": "WaiverRevocation"}
returns: "RevocationReview"
---
Verify event identity, exact waiver key, expected register revision and authority for the stated effective revocation. Missing or stale bindings leave the register unchanged. For a supported event append the revocation and mark future dependent reliance from its actual effective boundary; preserve prior valid actions and original grant history. Identify dependent owners from recorded reliance and prepare scoped notices without leaking private rationale. Record a notice delivered only on actual matching notification evidence; unknown delivery remains pending readback. Repeating the same event is idempotent; conflicting event reuse requires reconciliation.
