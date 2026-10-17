# Drained recipe decode reuse

The existing source-host handoff decides whether an earned resident owner can enter the shared recipe scheduler. That policy stays in the shared lineage verifier. The crisp change only passes the same already-verified immutable parent and loaded state through its nested exact checks.

1. With no supplied decode, run the original full handoff verifier once.
2. With an invocation-local supplied decode, reuse the existing exact receipt/parent-binding verifier; reject a different record or parent.
3. Preserve resident retirement, complete adoption history, external launch/package, recipe and runtime checks.
4. Pass the authenticated pair to final and retry verification, avoiding another optimizer deserialization. No persistent tensor cache, fabricated terminal or readiness waiver is introduced.

Tests use a real drained resident owner, both fresh and interrupted plan publication, default and reused decode paths. They count actual verifier entries and reject a mismatched supplied record.
