---
description: Judge one original candidate-domain clause and finish its actual one-item result.
args:
  state: CandidateDomainFrontier
returns: CandidateDomainFrontier
---
1. If pendingClauseIds is empty, return state. Otherwise select its first original ID and exact SourceClause. Keep earlier entries, the remaining IDs, actual inventory and governing rules unchanged.
2. Directly await one scoped nl<DomainClauseReading> judgment over this clause, its actual inventory role and relevant original governing rules. Decide whether it defines a candidate slot with finite values, supplies context for those values, or leaves a genuine missing premise. Return clauseId, kind, slots, reason, witnesses and unresolved details.
3. For each source-declared slot, retain exactly the values and meaning attributed to this primary clause. A governing rule may explain interpretation, but it cannot donate a different clause's slot to this unit. An explanatory candidate-use clause can return kind context with no slot.
4. Bind actualReading to the awaited result. Check its clauseId equals the selected original ID; bind witness sourceId/path from actual original clauses and require each quoted substring to occur in its cited original text. Keep the actual semantic fields rather than rebuilding a slot table in JavaScript.
5. Record a read entry for a grounded domain or context; retain kind unresolved and its missing premise as an unresolved entry. A tool/type failure creates a failed entry with actualReading null and its failure, while preserving earlier completed entries.
6. Bind next:CandidateDomainFrontier with exactly this pending ID removed, one appended DomainClauseEntry and all prior entries unchanged. Confirm the pending measure decreased by one without changing original clause text or source identity.
7. Finish this invocation immediately with eval({code:"return next;",finish:true}). The parent alone selects the following original clause and projects the completed actual slot readings.
