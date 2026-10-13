---
description: Derive the requested fact delta and hard preservation rules.
args:
  request: RewriteRequest
  instructions: SourceRecord[]
  canon: SourceRecord[]
  selectedScenes: SourceRecord[]
returns: RequestMeaning
---
1. Read request.requestedChange together with the director instructions. Resolve references to
   the target event using its explicit scene, actor, object and frame, not a repeated verb alone.
2. Identify the exact asserted facts removed by the change. Populate removedFactIds with visible
   fact IDs. Populate addedMeanings only with the positive requested replacement meaning.
3. Determine scopeFrameIds: removing an actual delivery need not erase a remembered delivery.
   A request to change the selected scene applies to its selected frame unless a rule says otherwise.
4. Translate every protectedMeaning and controlling instruction into a PreservationRule. Classify
   its obligation as canon, knowledge, physical, motivation, ambiguity or scope; quote its authority.
5. Distinguish a hard restriction from a dramatic preference. A prohibition on new copies is physical;
   preserving why a rescue is credible is motivation; preserving competing motives is ambiguity.
6. Keep both sides of an explicitly protected uncertainty. Do not resolve it merely because one
   alternative makes the requested rewrite easier. State its required unresolved meaning as a rule.
7. Intersect the request's allowed paths with source-owned edit scope semantically. If their meanings
   conflict, record the conflict in unknowns. Exact path authorization is checked by the service later.
8. Give stable rule IDs grounded in visible clause IDs. Keep rule witnesses and delta witnesses
   separate so an audit can distinguish the changed fact from the authority requiring preservation.
9. Return requestId, delta, rules and unknowns. Missing event identity or an unstated replacement is
   explicit uncertainty; it is not permission to select an arbitrary story fact to change.
