---
description: Separate utterance anchors from reporting and receipt dates.
args:
  clauses: SourceClause[]
  turns: TurnRecord[]
returns: TemporalRegistry
---
1. For each actual turn collect clauses naming its creation, archival attribution, forwarding or hearing.
   Use original identity joins, retaining conflicting readings rather than choosing by file order.
2. Form a scoped NL literal returning TemporalEntry for this turn. Capture the selected original clauses,
   actual turn and the source clock rule. Directly call it to judge originalTime and reportingTime separately.
3. Its numbered instructions first identify the event described, then bind the speaker's utterance frame,
   then identify any outer report frame. A later authenticated copy does not reset the original anchor.
4. A calendar day with no hour is sufficient for a day-based deictic phrase, but not an invented precise
   occurrence time. Keep an unresolved entry when an actual episode cannot be ordered from the source.
5. Bind its nonempty exact witnesses to original clauses. Validate all nonnull source times against the
   quoted metadata. Arithmetic compares established day/hour pairs; it does not select their meaning.
6. Return actual entries and unresolved reasons, including conflicting original/report dates. A receipt
   timestamp is handled by readReceipts and never silently overwrites a turn's original utterance time.
