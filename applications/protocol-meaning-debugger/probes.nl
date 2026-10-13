---
description: Construct an admissible experiment agenda with materialized predictions.
args:
  context: InvestigationContext
  state: InvestigationState
returns: ProbeCandidate[]
---
1. Select unused probe seeds from context.seeds whose IDs remain pending. Reuse a cached candidate
   from state.candidates when its immutable seed and definition bindings match this context; status
   updates do not change the source mechanism or trial. Materialize each missing seed once. Interpret its
   changedPremiseIds and visible authorization against the target's probe-scope clauses.
2. A seed is admissible only if it changes permitted raw trial facts and preserves source definitions
   and identity semantics. A seed label or allowed:true flag still requires a coherent changed trial.
3. For each admissible seed call predict(context, seed, state.hypotheses). Consume its typed plans,
   exact receipts and mechanism-specific predicted changes.
4. Determine targetedHypothesisIds from actual changed premises and predicted membership or
   numerator consequences. A numerical difference caused by two roles is not a clean single-role test.
5. Keep inadmissible seeds as ProbeCandidates with reasons and witnesses, without fabricating exact
   execution receipts. They are consumed once as rejected agenda items, not regenerated indefinitely.
6. Prefer meaningful intermediate distinctions: changed selectedReadingIds can reveal an anchor
   divergence even when complementary numerators leave the final percentages equal.
7. Return the pending cached and newly materialized candidates with seedId, cost, prediction, targets,
   admissibility and reasons. Candidate choice
   belongs to choose; do not choose the first file lexically as an implicit investigation policy.
8. The agenda is the finite supplied seed universe. No network experiment or external field-device
   operation occurs; all exact evaluations run on the immutable fictional trial snapshots.
