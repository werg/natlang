---
description: Derive activation and timely promised-consequent fulfillment through actual typed child judgments.
args:
  act: ActReading
  episode: Episode
  view: ParticipantView
  views: ParticipantView[]
  source: DialogueSource
  scoped: ScopedConversation
returns: CommitmentDelta
---
1. Select a source-established promise meaning performed true and owned by its actual attributed issuer. A quoted original
   can identify that issuer's original promise, while a report-only outer act creates no new promise.
2. Join actual issuer, beneficiary, consequent proposition and due-time fields from the act reading.
   Preserve a day-only due hour null and each independent prerequisite predicate with its exact scope.
3. For every actual predicate call judgeCondition(predicate, act, episode, views, source). Consume its
   actual truth, issuer knowledge and beneficiary belief separately. Neither belief nor assertion fulfills
   a condition whose source meaning concerns a real collection event or personal versioned acknowledgment.
4. With actual conditions all true the commitment can activate according to its source rule; with any
   false it remains conditional in this speech model. A source-resolved predicate with an open future
   window stays pending/conditional when actual is null; a genuinely unresolved interpretation yields
   unresolved. Keep that distinction grounded in the child reason and actual source cutoff.
5. For each actual consequence predicate call judgeCondition with role fulfillment and the original
   source due scope. Consume its actual result in Commitment.fulfillment; do not reuse activation truth
   as evidence that the promised consequent happened. An intention is not an actual key release.
6. Classify timely fulfillment from those actual returned consequence readings. A known completed
   consequent fulfills only in its source due scope. After the due day, complete absence can establish
   failed timely fulfillment of an active unconditional promise; unknown meaning/scope stays unresolved.
   A known future consequent not yet performed leaves the active promise active until its due scope closes.
7. A false activation predicate keeps this model's commitment conditional, even if its due horizon
   passed. This differs from an activated/unconditional promise whose timely consequent demonstrably failed.
8. Preserve the established commitment when its issuer privately wants to cancel it. Only the actual
   communicated/adopted retraction stage may replace its status as retracted for the relevant audience.
9. Locate any existing commitment by its actual original sourceActId before recomputation. If it is
   retracted, preserve that terminal status: later facts and due boundaries cannot reactivate it. Recomputed
   condition/fulfillment evidence may be retained without changing that status. A source-established new
   performed promise has its own original act lineage and does not overwrite the old retracted promise.
   Recompute an existing commitment's condition readings using newly available actual event evidence,
   retaining its original source act ID and establishedAt. Replacement changes condition/status fields,
   not its original promise date, issuer, version predicate or audience.
10. Return additions/replacements for this participant's actual ledger with original exact support. The
   caller applies these typed effects; it does not substitute its own successful activation interpretation.
