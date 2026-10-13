---
description: Check a proposed coordinate binding against identity evidence rather than appearance alone.
args:
  localMention: LandmarkMention
  globalMention: LandmarkMention
  transformedPoint: Point
  landmarks: LandmarkPartition
returns: AnchorAssessment
---
1. Interpret compatible as consistency of this proposed binding conditional on the supplied transform.
   The question is whether these places can be identified in this candidate world; which transform is
   the actual one remains a separate unresolved choice across the surviving world collection.
2. Find the landmark decision for these mentions, preserving their frame identities. If both global
   coordinates and transformedPoint are established, compare them exactly before judging the binding.
3. Return compatible=false for a coordinate mismatch or a source-established different-place decision.
   Include the specific coordinate or identity contradiction and its exact source witnesses.
4. Return compatible=true when the transformed point matches the stated global coordinate and no
   source-backed different-place constraint excludes this binding. This includes an unresolved actual
   reflection: the same local place may consistently bind a different global label in another world.
5. Return compatible=null only when a missing coordinate, identity-scope premise or frame ownership
   prevents assessing consistency even after conditioning on this candidate transform. State that premise.
6. A unique junction anchor may establish translation. A reused color mark contributes no new
   handedness constraint; preserve every candidate transform consistent with the established anchors.
7. Return the conditional assessment and precise witnesses. The parent consumes true bindings within
   this hypothesis, retains null assessments for further source review, and rejects only a contradiction.
