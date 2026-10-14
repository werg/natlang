---
description: Reduce original quote spans with a decreasing character-mass frontier.
args:
  turn: TurnRecord
  rules: SourceClause[]
  temporal: TemporalRegistry
returns: QuoteTree
---
1. Initialize QuoteSpanFrontier with the original turn object, actual rule/temporal contributions and
   one pending span covering its content. Give that root no parent and the actual turn speaker/time.
2. Call advanceQuoteSpan.iterateOn(initialFrontier, finished). The natural-language finished predicate
   is pending spans empty. Every step consumes one actual pending span and only activates proper nested
   disjoint quote interiors; their combined character length must be smaller than the consumed span.
3. Consume the returned frontier's nodes. Validate that each span refers to this original turn, lies
   inside its parent's source range and is represented once. Empty spans cannot enter the frontier.
4. Pending character mass is the sum of pending end-start lengths. Verify strict decrease at each
   step using exact arithmetic. This is structural termination, not a model call budget or depth cap.
5. An unresolved attribution is still a completed node with typed uncertainty. It does not trigger
   repeated re-judgment of the same span or force selection of a convenient speaker.
6. Return the actual QuoteTree. Preserve parent/child order by original source offsets, retaining both
   the outer reporter and each nested quoted original as separate meaning scopes.
