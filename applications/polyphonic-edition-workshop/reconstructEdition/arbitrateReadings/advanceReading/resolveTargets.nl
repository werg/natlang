---
description: Judge source-supported attachment of editorial claims to one written location.
args:
  eventId: string
  readings: WrittenEvent[]
  claims: EditorialClaim[]
  protocol: NotationProtocol
returns: TargetDecision[]
---
1. Bind location to the actual original eventId and gather its source readings. Check they agree on
   voice/section/location identity before evaluating a correction; a contradictory identity stays unresolved.
2. Initialize decisions empty. Visit each actual EditorialClaim once and retain its claimId, authority
   status, operation, explicit candidate IDs and exact attachment witnesses in this pair's scope.
3. Declare TargetDecision and create a scoped literal for the actual pair. Directly await it:
   const judgeTarget = nl<TargetDecision>`1. Read this eventId/readings and the actual claim plus editorialPolicy. Establish whether original location identity is known.
   2. If the source establishes no authorization for this operation or excludes this location, return applicable=false with basis=excluded and the actual exclusion. Unknown authority returns null with basis=unknown-authority.
   3. Join named section/voice/order and attachment description to the actual readings. An explicitly unique supported target gives true with basis=established-target. Several authenticated supported exclusive target candidates preserve null with basis=finite-exclusive-candidate. Missing decisive attachment evidence returns null with basis=unresolved-premise.
   4. Return the original claimId/eventId, actual applicability and basis, reason distinguishing authority from attachment, and nonempty exact witnesses drawn from these supplied objects.`;
   const decision = await judgeTarget({eventId,readings,claim,editorialPolicy:protocol.editorialPolicy});
4. Validate returned claimId/eventId against this actual pair and every quote against the supplied
   claim/readings/protocol witnesses. Preserve returned basis and reason. Validate the basis/applicability relation: finite-exclusive-candidate
   is null under known authority with this original event among supported exclusive targets; excluded
   is false; established-target is true. Unknown authority and unresolved premises remain null.
5. Append the actual decision exactly once. Never interpret the fact that two targets share a pitch
   or duration as evidence that either target was intended; source attachment governs this stage.
6. Return all actual pair judgments. compareSupport owns constructing finite local alternatives,
   and checkClaimConsistency owns one exclusive source-level assignment across locations.
