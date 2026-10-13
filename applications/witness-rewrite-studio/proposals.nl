---
description: Generate coherent repair alternatives and replay each against all obligations.
args:
  analysis: Analysis
  impact: ImpactResult
  component: AffectedComponent
  selectedEdits: ProposedEdit[]
returns: CandidateReview[]
---
1. Read component.obligationIds against impact.obligations and the protected rules. Build the
   required coverage conjunction before selecting a repair text; one fixed line may need several edits.
2. Read inventory.proposals as recipes, not facts. Extract each alternative's visible proposal ID,
   intended changes, old source quotations, replacement text and claimed preserved meanings.
3. Expand recipes into Candidates with precise ProposedEdits and exact source path ownership.
   Associate every edit with removed fact IDs and added meanings so replay can inspect its claims.
4. Reject a recipe from generation only when its required input cannot be located; retain that
   unresolved reason. Semantically tempting but invalid candidates belong in audit, not silent omission.
5. If source instructions authorize authored alternatives, formulate additional bounded candidates
   over the allowed paths. Their edits must address the same obligation conjunction without adding
   unsupported actors, objects or resolved motives. Label generated IDs distinctly from visible IDs.
6. Keep alternatives distinct when their information route, physical transfer or protected ambiguity
   differs. Cosmetic paraphrases of one repair are not separate semantic candidates.
7. For each complete candidate call simulate(analysis, impact, component, candidate, selectedEdits).
   Consume the returned CandidateReview, including the actual replay and audit findings.
8. Return all reviews. An empty list means this component lacks an executable proposal and must
   settle unresolved; it does not authorize repeating generation indefinitely.
9. Do not hardcode the expected proposal ID or use a numerical score as story validity. Child audits
   determine constraint satisfaction; the tradeoff specialist chooses only among justified survivors.
