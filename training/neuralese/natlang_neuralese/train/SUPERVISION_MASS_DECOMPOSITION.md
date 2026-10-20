# Native supervision mass and complete-trajectory intent

The policy is a typed natural-language decision: given the admitted trajectory,
its authenticated assistant targets, lower-weight context/feedback, and declared
supervision policy, decide whether its loss teaches successful execution without
allowing a long prompt to drown out a short action. Preserve every usable source,
source split, completed row credit, optimizer moment, and RNG state. The whole
actual communication graph remains one differentiable example.

1. `inspectSupervision(examples, policy) -> MassEvidence` measures actual native
   graph CE masses, distinguishes raw per-position weighting from normalized
   context/target allocation, and compares the known shared text objective.
2. `chooseSupervision(evidence, intent) -> DeclaredContinuation` is policy, not an
   automatic numeric threshold. A change needs exact source/target provenance
   and an explicit continuation; historical rows and weights remain authentic.
3. `applySupervision(example, declaration) -> WeightedExample` is crisp only for
   the exact selected weight algebra. Reuse `balanced_position_weights` and
   `segmented_position_weights`; do not create a second loss or trainer. Never
   infer assistant targets from approximate weight magnitudes on an arbitrary
   native graph. Retain node boundaries, imports, forced zero weights, failed
   actions, and all producer/consumer derivatives.
4. `verifySupervision(reference, actual) -> Evidence` is an exact verifier:
   compare effective normalized loss and every gradient to the shared objective,
   including a short target after a long context, full trajectories, and native
   communication. Per-graph normalization must not accidentally change the
   relative credit or multiplicity of distinct examples.

The audit's declared implementation is `native_supervision.apply_policy`, called
by the existing source stream after raw row-binding verification and by the
existing semantic objective before producer/consumer CE. The declaration is
bound to source SHA plus exact producer encoding. `build_native_ce_supervision_policy.py`
is only immutable inventory/CLI plumbing; it does not decide admission or
invent assistant intent. New eligible undeclared sources fail explicitly, while
already segmented whole trajectories retain their exact objective. The existing
recipe input role and provider dependency closure transport this declaration.

No live recipe, immutable corpus, checkpoint optimizer,
exposure counter, tokenizer identity, or frozen runtime changes implicitly. Text
and graph-policy RL have separate objectives; a CE weighting repair must not
silently rescale policy likelihood/reward/advantage terms. No model-facing text
change is proposed by loss-weight algebra alone.
