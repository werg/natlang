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

The seed operation inherits the complete learner/backend settings, then retains
explicit seed-objective overrides. Its command expands those seed settings,
including the existing refresh boundary and resident source authority. Existing
fully revalidated configurations receive the same correction before reuse;
policy updates retain their own arguments. This closes argument plumbing, without
changing the declared finite-boundary scheduling policy.

Optional per-model `rows_per_round` and `supervised_quantum_updates` are positive
integers; their absence preserves existing defaults. The latter uses the shared
checkpoint-every/refresh boundary for seed and mandatory supervised operations
only. If the previous held cadence is longer, evaluate once at the quantum
boundary (250→16); retain a shorter cadence when it divides the quantum, and
reject incompatible shorter cadences. Held token/graph coverage is unchanged. It changes no epochs, objective, geometry, optimizer history or earned
credits. Existing checkpoint alignment means the actual report delta remains
authoritative, rather than promising exactly N updates. A pending old packet
retains its cases, selection, source and bootstrap binding; only supervised stop
cadence and seed-argument repair may apply, with the exact declaring plan bound
separately. New packet size/task source adopt after that operation completes.
There is no completed-group prefix finalizer in the current collector: stopping
resumes the same sealed packet; it does not make a partial packet complete.
