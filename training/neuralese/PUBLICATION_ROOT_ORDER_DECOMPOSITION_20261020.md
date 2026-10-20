# Exact publication roots at the native trainer boundary

The existing controller chooses the publication roots. This change only verifies
and transports that declaration; it adds no scheduling or admission policy.

1. Parse the existing forwarded roots list and validate the sealed producer selection.
2. Preserve the declared list ordering in the manifest, matching the unchanged trainer argv.
3. Use the existing `emit_current` and `read_owner` durability and identity checks.

The implementation is crisp because parsing, exact argv identity and immutable
manifest publication require mechanical equality. Trainer objectives and numerical
execution remain unchanged.
