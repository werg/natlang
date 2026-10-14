---
description: Reconstruct one declared answer path from original evidence and actual historical deltas.
args:
  answer: QueryAnswer
  source: DialogueSource
  scoped: ScopedConversation
  episodePlan: EpisodePlan
  evolution: PerspectiveEvolution
returns: AnswerAudit
---
1. Start with the answer's actual focus and asOf moment. Independently follow its declared proof roots
   through actual generated endpoints to original clauses, acts, receipts, conditions and revision IDs.
2. Re-evaluate whether those original premises license the observable in that requested dimension.
   Compare original statement time to due-time anchor; compare actor belief to explicit adoption, not truth.
3. For knowledge require actual participant availability and endorsement of verified evidence. For
   absence answers require a closed log or source-explicit non-adoption, rather than an omitted array row.
4. For ambiguous referents/illocution check no eligible alternative disappeared and no resolved actor
   was invented. For report-only quotation inspect the outer actor separately from the original quoted issuer.
5. Compare with actual history at or before asOf. A later correction may explain another answer but
   cannot validate this one retrospectively. Return a failed audit for that temporal perspective mismatch.
6. Return AnswerAudit with checkedProofIds, actual nullable pass and original supporting witnesses.
   Free reason wording is judged semantically; this stage never edits the answer or a past view in place.
