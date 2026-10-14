---
description: Derive actual conditional commitments and independently judged activation.
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
   false it remains conditional in this speech model. Any unresolved required condition yields unresolved.
5. Test fulfillment only against an actual source-established consequent event through the current
   episode horizon. An intention, promise or reported desired release is not an actual key release.
6. Preserve the established commitment when its issuer privately wants to cancel it. Only the actual
   communicated/adopted retraction stage may replace its status as retracted for the relevant audience.
7. Recompute an existing commitment's condition readings using newly available actual event evidence,
   retaining its original source act ID and establishedAt. Replacement changes condition/status fields,
   not its original promise date, issuer, version predicate or audience.
8. Return additions/replacements for this participant's actual ledger with original exact support. The
   caller applies these typed effects; it does not substitute its own successful activation interpretation.
