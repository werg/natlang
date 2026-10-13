---
description: Verify the requested event change and selected-world scope.
args:
  original: Analysis
  replay: Replay
  obligations: Obligation[]
returns: SemanticFinding[]
---
1. Select requested-change and canon/scope obligations from the original rule interpretation.
   Read the original FactDelta and the reconstructed replay.world.authority.
2. Check that each removed selected-world fact is absent from the edited selected event. A memory
   retaining the same action is allowed when the request did not target that memory frame.
3. Check positive replacement meanings actually occur in the intended scene and scope. Replacing
   physical delivery with no event at all does not satisfy an explicitly requested oral communication.
4. Reject a candidate that shifts the requested event to an unselected take or hides it under quotation
   while leaving the actual selected delivery intact.
5. Check protected participant, place and edit-scope clauses against resulting event views. A new
   informed messenger may fix dialogue but violate the scene's stated participant constraint.
6. For each checked obligation return a finding with satisfied, violated or unresolved, a precise
   reason, required source IDs and exact witnesses from rules and resulting scene.
7. Keep absence reasoning source-grounded: cite the complete target selected event and its replaced
   span, not a fabricated empty quotation claiming an event does not exist anywhere.
8. Return only owned findings. Knowledge, prop and ambiguity obligations remain for their own
   specialists; do not mark the whole component satisfied after checking the target event alone.
