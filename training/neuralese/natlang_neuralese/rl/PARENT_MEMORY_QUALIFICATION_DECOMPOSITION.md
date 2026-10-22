# Parent capacity metadata at the policy-training boundary

The production policy is unchanged: a sampled whole-graph policy operation must
check the capacity of its actual admitted objective before updating the model.
A supervised parent's capacity receipt is provenance for that parent, rather
than evidence about the new sampled policy objective.

1. The existing natural-language production recipe selects the policy operation.
2. The crisp CLI boundary consumes inherited supervised-only flags, including
   `--memory-qualification`, alongside foundation and resident-source metadata.
   The original controller configuration and parent retain their provenance.
3. The existing graph admission mechanism binds actual sampled graphs and their
   policy objective, constructs their policy-capacity request, and passes it to
   the actual trainer. Its normal runtime, numerical and capacity checks remain
   required. No supervised request is reinterpreted as an RL qualification.
4. The actual trainer continues exact optimizer/RNG and carried supervised
   exposure. Completing a policy operation does not complete the mandatory SFT
   recipe, and retrying a completed operation must not update twice.

This is argument plumbing, not a new model-facing instruction or scheduling
policy. Both models share this boundary. The real-trainer continuation test
passes an inherited supervised request that cannot qualify the RL objective,
then proves a real update, carried SFT exposure, a bound policy-capacity request,
and an exact no-op on the second invocation.
