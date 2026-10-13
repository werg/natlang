---
description: Separate submitted intent structure from rule examples and future triggered consequences.
args:
  clauses: SourceClause[]
  request: TurnRequest
  initialBoard: BoardSnapshot
returns: IntentIndex
---
1. Find the request's announced intent and submitted response IDs in source declarations. Check
   that each is described as an actual submitted action; a rule saying what would happen supplies
   a possible consequence, not a new player submission.
2. For the announcement, preserve printed part IDs and order, actor, target instance identities,
   effect descriptions and proposed quantities. Capture of controllers is a later owned transformation;
   this stage retains the source's action structure without applying its effects.
3. For each submitted response, interpret its actor, explicit response order and state change. Keep
   a control transfer distinct from leaving/re-entering a zone, and an attachment move distinct from
   spending the attachment's replacement token. Record costs already paid as source context.
4. Bind target references to original current instances where the intent requires them. A rule's
   permitted future instance is not silently substituted for the submitted target. Missing identity or
   response ordering remains unresolved rather than being inferred from file names.
5. Record context clauses that resemble actions but are not submitted: hypothetical ability
   consequences, illustrative alternatives and result instructions. State the source role excluding each.
6. Return the actual intent readings, excludedSourceClaims and unresolved, with exact witnesses.
   Later capture and window stages consume these fields; the intake stage does not execute responses.
