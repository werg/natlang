---
description: Resolve expressions using actual quote frames while preserving ambiguous pronouns.
args:
  turn: TurnRecord
  tree: QuoteTree
  source: DialogueSource
returns: ReferenceBindings
---
1. Traverse actual QuoteTree nodes in source order. Read each node's owned substring excluding immediate
   child interiors; an expression is judged in exactly its owning frame rather than every enclosing report.
2. Construct and invoke a local NL expression-inventory literal over that substring. It identifies
   person/time/place expressions that affect the source question, with exact source quotes and node ID.
3. For each returned expression call judgeDeixis(expression, node, turn, source). Consume its actual
   referent set, original anchor, offset and resolvedAt rather than inferring a date from document location.
4. If the returned meaning supplies an established anchor and day offset, apply exact day arithmetic
   and compare with resolvedAt. A missing original hour remains null in a day-based result, not midnight.
5. Bind exact source quotes to the original turn's clause. Require at least one quote for any resolved
   or ambiguous contribution. An empty quote array cannot certify a reference even if IDs are well typed.
6. Keep all original eligible referents when meaning is ambiguous. Never remove one because a later
   verified fact makes another interpretation more convenient; that would silently rewrite the speaker.
7. Return actual ReferenceBindings with node IDs and witnesses. Unknown referents/time remain explicit
   and block only decisions that depend on them, not unrelated resolved parts of the conversation.
