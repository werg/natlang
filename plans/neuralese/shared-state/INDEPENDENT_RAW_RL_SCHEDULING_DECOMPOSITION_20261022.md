# Independent RAW RL scheduling decomposition (2026-10-22)

## Natlang-native policy

The policy is a shared RAW scheduler declaration: when a model owns its optimizer and a RAW RL bridge operation has completed, continue its own mandatory supervised recipe obligations and schedule another RAW RL round at the next local cycle boundary even if another participant has not completed a communication foundation. The policy never grants foundation qualification, VAE readiness, task success, or training credit. The two-participant foundation request remains the sole rendezvous path for cross-model semantic work.

## Mechanism decomposition

1. **Recipe/scheduler intent — declarative policy.** Add an explicit `rl_scheduling_policy` to the sealed RAW declaration. The shared coordinator validates the supported value and records the exact declaring plan binding in its effective configuration. Omission keeps historical `foundation_qualified` scheduling behavior.
2. **Cycle transition — crisp cursor plumbing.** `student_cycle` already executes the actual RAW bridge operation and the full pending supervised stage inventory. At a fully completed local cycle boundary, choose the next cursor phase from the recorded policy. `independent-local-cycles-v1` selects `rl` without changing `foundation_qualified`; default policy retains the existing foundation-gated transition.
3. **Training/eligibility — existing shared mechanisms.** Reuse `round_job`, its bridge request/result validation, whole-episode graph capture, numerical checks, held/source checks, and exact full-state publication. Do not add a second RL eligibility shortcut or require held-task wins for learning. Failures remain failures and no scheduler policy converts them to rewards or readiness.
4. **Supervised debt and cross-model readiness — existing exact verifiers.** Preserve every mandatory SFT/DPO stage and communication exposure obligation. Keep the >=2 exact foundation-candidate gate, role evidence, and VAE readiness unchanged. A local RL cycle is not a completed foundation and cannot populate its readiness receipt.
5. **Boundary rebinding — source-bound controls.** When a verified source/resident boundary reconfigures the RAW coordinator, reapply the policy only from the boundary's current sealed declaration and bind it to that declaration; never infer the setting from a stale cursor or peer state.
6. **Verification — crisp tests plus declared semantics.** Exercise the real scheduler's cursor transition with a missing peer candidate and actual bridge failure propagation; assert supervised pending obligations remain and no cross-model readiness is created. Keep exact whole-graph/native update tests in the existing shared trainer suite.
