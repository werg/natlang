---
description: Consume one span and bind a judged frame to strictly smaller child spans.
args:
  frontier: QuoteSpanFrontier
returns: QuoteSpanFrontier
---
1. Select and remove one actual pending QuoteSpan. Read only that exact substring from frontier.turn;
   keep the original turn, span bounds and parent frame as provenance throughout this step.
2. Call judgeQuotationFrame(span, frontier.turn, frontier.rules, frontier.temporal). Consume its
   actual attributed speaker, original time, original target turn, mode and proposed immediate children.
3. Require nonempty exact quote support for each decided frame. Bind witness source/path/quote from
   the original turn witnesses; an absent quote produces a source-span repair error before activation.
4. Validate child start/end coordinates are absolute offsets within this span. Children are proper
   nonempty subranges, pairwise disjoint and in source order; their combined length is strictly smaller.
5. For each valid immediate child create one generated span ID with this parent ID, original turn ID,
   returned attributed speaker and returned originalTime. Copy these actual child decisions; do not
   replace them with the reporter's inherited attribution when the judgment resolved an original speaker.
6. Append a QuoteNode for the consumed span and activate only those validated children. Record local
   unresolved judgments on that node; a null speaker does not invent an actor or omit the quoted passage.
7. Return the updated typed frontier with the remaining spans and actual nodes. No call from this
   step goes to itself, partitionQuotation or another ancestor in the active NL call chain.
