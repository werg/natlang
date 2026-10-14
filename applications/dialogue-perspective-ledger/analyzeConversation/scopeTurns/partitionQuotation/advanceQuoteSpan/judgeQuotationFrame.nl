---
description: Judge one quotation frame from its actual source substring and attribution.
args:
  span: QuoteSpan
  turn: TurnRecord
  rules: SourceClause[]
  temporal: TemporalRegistry
returns: QuoteFrameDecision
---
1. Capture the exact original substring named by span, original turn metadata, inherited speaker/time,
   visible source quote rules and actual temporal entries. Do not pass an expected speaker or due day.
2. Form a scoped natural-language literal returning QuoteFrameDecision. Directly call it on that
   substring; its policy is to identify this layer and its immediate quoted interiors independently.
3. Within the literal first determine whether this layer is the speaker's own act, a reported layer,
   an attributed quotation or genuinely unresolved. Use explicit attribution rather than string proximity.
4. Identify the original speaker/time only from actual attribution and authenticated original references.
   A reporter may quote another reporter who quotes an original; each next interior gets its own frame.
5. Return immediate child interiors with absolute source offsets, proposed original turn IDs and their
   attributed frames. Nested grandchildren belong to later frontier steps, not a flattened assertion list.
6. Retain unresolved attribution and competing original-time interpretations. A privately mistaken
   temporal belief is content in the diary speaker's own frame, not a correction to an original quote.
7. Require nonempty supporting exact quotations for decided fields. Return that literal's actual
   decision; the parent validates spans and binds original identity before downstream reference resolution.
