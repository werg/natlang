---
description: Repair a selected story event through its causal and epistemic consequences.
kind: directory-reducer
args:
  request: RewriteRequest
  snapshot: SourceSnapshot
uses: [world, epistemics, props, motifs]
returns: PatchPacket
---
1. Call intake(request, snapshot). Keep the returned interpretation and inventory together as intakeResult.
   The exact snapshot revision identifies the base of every candidate. Rules supplied by the director
   govern the requested revision; a character's assertion is not a director rule.
2. Call world(intakeResult). This branch returns scoped frames, entity distinctions, partial event order
   and authorized facts. Preserve unresolved relations instead of inventing a complete chronology.
3. Call epistemics(intakeResult, worldResult), props(intakeResult, worldResult), and
   motifs(intakeResult, worldResult, knowledgeResult). Each returned view owns its distinct obligations.
   An information transfer and a physical transfer remain different contributions even in one scene.
4. Assemble analysis from these returned values. Call impact(analysis) to compute the connected
   consequences of the requested delta. Use its components as the repair frontier, not all scene files.
5. Initialize RepairState with remainingComponentIds from impactResult.components, empty settled IDs,
   decisions, selectedEdits and unresolved. An empty component list is a settled result, not a retry.
6. Invoke folder.iterateOn(repairStep, initialState, analysis, impactResult) with measure equal to
   remainingComponentIds.length and termination when that list is empty. Consume the returned state
   and private folder. Every reducer invocation must settle precisely one remaining component.
7. Call packagePatch(request, analysis, impactResult, finalState). Return that child's packet.
   A failed component remains explicitly unresolved in coverage; another component can still succeed.
8. This invocation proposes edits. Keep original evidence unchanged and return effects as packet data.
   The private folder holds only review decisions. Applying the story packet is an exact host operation.
9. Do not infer a replacement result from raw text after a specialist returns. Contradictions between
   views belong in the component audits and packet unresolved field, with both source witnesses.
