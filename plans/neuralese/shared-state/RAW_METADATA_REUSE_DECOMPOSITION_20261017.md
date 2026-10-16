# Reuse exact verified metadata within raw continuation

The training coordinator keeps useful work moving. At each boundary it validates
ownership and current policy before reusing a previously verified installed
configuration. It does not repeatedly deserialize an unchanged original parent.

1. Separate exact live-owner/immutable-handoff verification from checkpoint-state
   validation. Recheck process ownership each time. Initial state validation and
   every changed proof/config/helper/source/runtime still use the full verifier.
2. Persist a host-generated verification receipt bound to the exact configuration,
   plan, handoff/proof files, current source inputs, runtime and verifier code.
   Reuse only an unchanged receipt; do not store a model state or persistent
   tensor cache. Preserve current-policy and lineage checks separately.
3. Share one already-verified handoff state within initial configuration.
4. During a provider dependency invocation, register every logical consumed path
   before deduplicating the exact path/SHA/participant/semantic/relocation context.
   Use one explicit local loader for shared-manifest classification and subsequent
   metadata inspection. Release mapped states when that traversal finishes.

These are exact verifier/durability/external-service mechanisms, not a change to
scheduling policy, model arithmetic, objectives, source admission or gradients.
