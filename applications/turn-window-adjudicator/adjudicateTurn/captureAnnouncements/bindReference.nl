---
description: Interpret one consumer's captured, live or last-known reference to an announced target.
args:
  part: IntentPart
  consumerRule: RuleReading
  initialBoard: BoardSnapshot
  clauses: SourceClause[]
returns: ReferenceBinding
---
1. Find the original target instance in initialBoard. Keep its cardId and instanceId distinct; an
   explicitly permitted future instance is not another spelling of the current physical target.
2. Gather only source clauses governing this consumer's reference basis and the relevant identity
   convention. Capture that context and the original target, leaving the changing input as the concrete
   part/consumer pair. In eval declare a local reference judgment with controllerBasis, reason and
   evidence:{clauseId:string;exactQuote:string}[]; controllerBasis uses ReferenceBinding's exact union.
3. Create a scoped NL literal and directly await it. Ask which reference this consumer uses: the
   controller at announcement, the current controller when its condition is tested, the departing
   instance's last-known controller, no controller reference, or a genuinely unresolved source basis.
4. Consume the actual returned basis. For captured, copy the original target's controller into
   capturedController. For current or last-known, leave capturedController null and retain the tracked
   instance identity so later windows use their own correctly timed board or departure record.
5. Validate each nonempty returned quote within its original supplied clause witness, then bind source
   identity/path from that parent-held witness. A missing or failed quote makes this contribution fail
   for repair; a semantic basis is not accepted merely because its label is a valid enum member.
6. Return consumerRuleId, partId, original trackedInstanceId, actual controllerBasis, capturedController,
   reason and checked witnesses. Later preparation and trigger binding consume this returned choice;
   they cannot replace it with the current controller just because that actor appears in a later file.
