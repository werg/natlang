# Zero-update supervised retry decomposition

## Policy

The supervisor may retry a mandatory supervised stage when its exact sealed
operation failed before producing any optimizer update or checkpoint, the
original controller and worker owners have exited, and the saved full state is
still the exact state/cursor parent. The retry keeps the same stage and all
earned exposure, uses a fresh unused operation namespace, and receives no
training, completion, or readiness credit. A corrected qualification request
must be versioned and bound to the restored model, runtime inventory, packing,
and an actual capacity measurement. Failure evidence, not a fabricated drain
or successful operation, authorizes this recovery.

## Mechanism

1. A crisp verifier checks the exact failed bridge request/result, its artifact
   inventory, zero update/checkpoint evidence, exact current stage and parent,
   and absence of every captured owner/descendant. The verifier also confirms
   the corrected runtime/request authority supplied by the supervisor.
2. A durable transaction snapshots config, state, and cursor; derives candidate
   config/state/cursor; reserves the next unused `raw_retry_generation`; and
   commits by compare-and-swap under the canonical and production leases.
   Retries are idempotent only for the same request and durable intent.
3. The candidate config is prepared through the existing completed-supervised
   future-config and lineage machinery, so runtime/provider/publication policy
   stays shared and source lineage remains exact. It does not create a completed
   supervised receipt or alter checkpoint/exposure state.
4. The normal runtime handoff/configuration boundary verifies and installs the
   retry receipt once, then the ordinary `student_cycle` runs the unchanged
   stage from the unchanged checkpoint in the fresh namespace.

## Crisp boundaries

Filesystem identity, hashes, owner census, capacity result bindings, locking,
snapshots, atomic writes, and compare-and-swap are exact external-world facts
and remain crisp. Which retry is useful and which versioned request/runtime
controls should be attempted is supervisor policy; it is supplied as an
explicit reviewed future invocation rather than inferred by the verifier.
