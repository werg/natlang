---
description: Check a proposed coordinate binding against identity evidence rather than appearance alone.
args:
  localMention: LandmarkMention
  globalMention: LandmarkMention
  transformedPoint: Point
  landmarks: LandmarkPartition
returns: AnchorAssessment
---
1. Find the landmark decision for these mentions, accounting for which frame each identifier belongs to.
   A fixed same-place anchor must agree with the global mention's exact coordinates when they are stated.
2. If a source decision says the places are different, reject an attempted identity binding even when
   their visual descriptions or exported coordinate labels resemble one another.
3. If identity remains unresolved, a matching point can support a transform-specific possible binding,
   but it does not prove that the transform is the actual one. State that distinction in reason.
4. An explicitly unique junction anchor may establish translation. A repeated color mark cannot
   independently establish handedness or disambiguate the lost export convention.
5. Return compatible, reason and the precise identity/coordinate witnesses. A missing coordinate
   cannot be replaced by a guessed location just to make an alternative fit.
