---
description: Settle one connected repair component and persist its decision on a private folder.
kind: directory-reducer
args:
  state: RepairState
  analysis: Analysis
  impact: ImpactResult
uses: [proposals]
returns: RepairState
---
1. If remainingComponentIds is empty, return state. Otherwise find the first component in impact.
   Keep the complete protected-rule and obligation conjunction for this connected component.
2. Call proposals(analysis, impact, component, state.selectedEdits). Its reviews already include
   counterfactual reconstruction and independent preservation audits; do not substitute raw recipes.
3. Call compareCandidates(component, reviews, analysis.intake.meaning.rules). Consume the returned
   choice, including rejected reasons and genuinely equivalent alternatives.
4. Select edits only from reviewed candidates named in choice.selectedCandidateIds. Keep all their
   audit findings, consumedLocalDecisions and choice.localDecisions IDs in a RepairDecision for this component.
5. If no candidate survives, append an explicit unresolved component reason. Settle that component
   anyway; uncertainty is a result, not a trigger to regenerate the same candidate frontier forever.
6. Before adding selected edits, run exact compatibility checks against already selected edits and
   original allowed paths. An overlap defect settles unresolved rather than silently replacing prior edits.
7. Write a decision record under decisions/<componentId>.json in this private folder, containing
   the typed choice and audits. Never overwrite original story evidence or oracle files.
8. Return the new RepairState with this ID removed from remainingComponentIds, appended once to
   settledComponentIds, its decision appended, compatible edits accumulated and unresolved preserved.
9. The component count decreases in every continuing step. The root's folder.iterateOn owns the
   transaction history; this reducer has no external publication or mutation of the original snapshot.
10. Components sharing necessary premises must already be joined by impact. If replay exposes a
    missing cross-component dependency, retain that defect explicitly for final coverage review.
