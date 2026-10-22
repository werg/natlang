# Owned observer failure diagnostics decomposition

The owned publication observer follows the declared current learner, verifies its exact checkpoint publication, and exports an independent private pin. Those policy and verification paths are unchanged. In particular, this change does not alter hashing, cache trust, source-stat checks, or immutable-pin validation: the observed R26 runtime has `sha256_file_hex_cached(path, *, trust_archive_receipt=True)`, and the checkpoint verifier already requests `False`.

1. The operator may select one cache file directly under the chosen production root's `cache/` directory. The CLI validates the exact normalized absolute path and passes it into both fixed-publication and current-owner observer programs before they import the frozen runtime or hashing module. Omitting it preserves existing behavior.
2. The selected cache remains only a cache: the qualified runtime's strong digest mode is used with `trust_archive_receipt=False`, so cache reuse is keyed by exact file identity and misses still hash bytes under before/after stat checks. No checkpoint-selected executable helper or weakened source/pin verification is introduced.
3. Remote observer and exporter failures retain their concrete exception type, a bounded message, and a bounded traceback through the worker event and retry journal. Exact plumbing redacts common credential-shaped values and never serializes the process environment. Retry and publication decisions remain unchanged.
4. Tests cover cache-path scope, generated-code import ordering, exact-stat cache reuse, redaction, traceback bounds, worker propagation, and persisted retry evidence.

The implementation is shared by LFM and Mellum. It changes diagnostics only; it does not claim to resolve the observed long first-time full-checkpoint verification. That path still needs measured call/cache evidence and the actual exception traceback before attributing the delay or class-only failure to hashing.
