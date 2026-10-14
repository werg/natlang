---
description: Resolve a question without switching from requested perspective to global facts.
args:
  query: DialogueQuery
  source: DialogueSource
  scoped: ScopedConversation
  evolution: PerspectiveEvolution
returns: QueryFocus
---
1. Read the actual question's declared dimension, target, participant and asOf time together. Interpret
   whether it asks about original statement, believed proposition, verified event, act or commitment.
2. Validate source-visible target IDs. Resolve original target utterance to actual attributed act IDs;
   resolve a proposition key to its declared meaning. Do not use a generated proof ID as a source target.
3. For a participant query choose the latest actual history snapshot at or before asOf, then its named
   ParticipantView. Before the first episode use the genuine empty view, not the final actor ledger.
4. For a global verified-fact question retain the actual completeness horizon and occurrence semantics.
   It may inspect global truth while still being barred from participant belief/knowledge evidence.
5. For nested quotation targets retain the relevant actual QuoteNode and original turn attribution.
   An ambiguous pronoun cannot be collapsed merely because the question names another convenient actor.
6. Return QueryFocus with actual IDs/view, interpreted dimension, source witnesses and unresolved scope.
   The query wording is not evidence that its presupposition is true; it only identifies what must be tested.
