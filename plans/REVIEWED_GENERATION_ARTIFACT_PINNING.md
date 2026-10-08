# Reviewed generation artifact pins

Reviewed generation plans should separate **execution inputs** from **source
history**.

Execution pins identify the exact files that the running campaign consumes:
the materialized source cases, source proof and review receipt, frozen runtime
receipt, queue or dispatcher, provider configuration and readiness evidence,
and any launch helper code. A launcher must verify these hashes before it
starts a provider request.

Source history records how an immutable source artifact was authored. Builder
and data-module hashes belong in a separate provenance section or source
manifest. They help reproduce and audit the source, but are not execution
inputs when the campaign consumes only the already-materialized source file.
Changing an unused builder must not invalidate a plan that pins an unchanged
source file. If a new source is generated, it receives a new source hash and
review; an earlier source artifact and its plan remain unchanged.

These pins describe reproducibility and launch scope. They do not grant
semantic quality, training admission, trace admission, or DPO admission.
