# Apply declared metadata environment and expose measured RPC phases

The configured sandbox resource owns its paths, cache roots and executable PATH.
The provider is crisp external transport and exact verifier plumbing.

1. Prefix the resource's explicit string environment before any imports in
   dependency_code, including its standalone CPU-loader use. Apply the same
   prefix in remote for hash checks, mirror inventories and other metadata.
   Default empty environment preserves existing generated code bytes.
2. Avoid repeating the same prefix when dependency code already contains it.
   Do not alter model-worker RUNNER, resource selection, source hashes,
   checkpoint loading/trust, runtime inventories or ownership semantics.
3. Record host-observed sandbox binding, large script upload, exec RPC and
   cleanup durations for existing operations. These include SDK overhead; they
   do not claim child CPU/unpickling or GPU time. No extra probe is launched.
4. Keep timing data to hashes, sandbox IDs, operation names, durations and
   return/error classes. Never log source code, environment values or output.
   Telemetry failure warns without replacing the original operation outcome.
5. Test early import/loader environment, actual short/large remote subprocesses,
   unchanged empty-environment output, failure preservation and timing fields.
   Independent review precedes any controlled provider adoption.

Ledger provider-metadata-environment-20261017 passes 44 focused provider-worker
and dependency tests in 13.20 seconds (790.8 MiB peak), including real short and
large remote subprocesses and first-import/CPU-loader environment checks.
Quick checks pass 22 ratchets plus TypeScript check. These are metadata/control
checks only; no model probe or active provider mutation occurs. Host RPC timings
are observational and cannot be attributed directly to restricted unpickling,
source enumeration, or child CPU time without further measured phase evidence.
