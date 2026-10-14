---
description: Read delivery and attachment provenance while keeping adoption separate.
args:
  clauses: SourceClause[]
  turns: TurnRecord[]
  physicalFacts: PhysicalFact[]
returns: ReceiptRegistry
---
1. Gather original hearing, transport, receipt and attachment clauses. Keep dispatch and arrival events
   distinct, including an explicitly delayed delivery or a receipt ID whose spelling resembles another day.
2. For each original receipt unit directly await a scoped NL ReceiptReading literal capturing its clause,
   actual turn inventory and physical fact IDs. Ask which original content reached which participant when.
3. The literal's numbered instructions distinguish sent, received and read times; identify the exact
   attached content; then determine whether a verified evidence attachment actually accompanied it. Return its actual verifiedFactIds,
   not just a boolean claim of verification; the parent consumes those exact source-log identities.
4. Consume actual turnIds, recipientId and times. Bind original quote/source identity from the selected
   receipt clause; reject absent IDs, empty support or a fabricated quote with that receipt's identity.
5. An authenticated quote packet provides access to the earlier content, but not access to every diary
   or verified log in the application's folder. Retain only the actual named carried/attached content. Check returned verifiedFactIds against actual physical facts.
6. A missing read time is not an invented reading event. Receipt may establish availability according
   to the actual speech model without proving endorsement. Adoption and doubt remain later speech acts.
7. Return receipts and unresolved transport/content joins. Two receipts about the same content remain
   different original evidence events if recipient or clock differs; exact duplicates alone may merge.
