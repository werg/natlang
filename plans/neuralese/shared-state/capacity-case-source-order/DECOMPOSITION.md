# Capacity cases use the actual source order

The typed metadata operation resolves declared cases against the same warmed
source tree used by the trainer. Classification-proof part order is provenance,
not a promise about manifest traversal order.

1. Load or retain each exact source once through the existing cache-only loader.
2. Resolve each declared case and reference by its unchanged ID and row binding.
3. Reject missing or ambiguous identities and changed declared geometry.
4. In strict mode require the declared index to select that identity. The explicit
   producer mode may change only index fields and returns every before/after check.
5. Preserve original requests and fresh numerical qualification requirements.

These are crisp identity verifiers and immutable metadata plumbing, not a new
qualification policy. No graph payload parsing, model, optimizer or exposure work
belongs to this operation.
