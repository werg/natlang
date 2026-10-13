---
description: Apply finite basis alternatives and test them against independent anchors.
args:
  survey: SurveyEvidence
  landmarks: LandmarkPartition
returns: FrameHypothesis[]
---
1. Take each local FrameAlternative from survey.frames.alternatives. Work on one alternative at a
   time and initialize a fresh ConditionalAnchorBinding[] named conditionalBindings for it. The fixed
   global coordinate convention supplies comparison landmarks, not another uncertain world.
2. Inventory every local landmark in this alternative's source frame. For each positioned mention,
   compute transformedPoint as origin plus its local x times uVector plus its local y times vVector.
   This arithmetic applies the interpreted alternative without deciding which alternative is actual.
3. For this local mention, propose independently named global counterparts whose established position
   equals transformedPoint, together with any counterpart required by a source-established same-place
   anchor. Preserve the local and global frame identities while constructing this finite pair collection.
4. For every proposed pair, await assessAnchorPair with these actual mentions, transformedPoint and
   landmarks. Validate every returned witness against the original survey document ID, path and
   nonempty exact quotation. A failed source check fails this contribution and requires evidence repair.
5. Immediately append one ConditionalAnchorBinding to conditionalBindings. Copy alternativeId and
   both mention IDs/frames from this pair, transformedPoint from this calculation, and assessment from
   the actual returned child value. Each completed pair contributes exactly one row before the next pair.
6. After all pairs for this alternative complete, group their rows by local frame and local ID. Read
   assessment.compatible from those rows. A true row supports its named global counterpart within
   this alternative; it leaves the actual reflection choice to the surviving world collection.
7. If one local mention has exactly one supported global identity, project that row to a binding with
   its localId, globalId and transformedPoint. Equivalent evidence rows for the same global identity
   contribute combined witnesses. Multiple distinct supported identities require a witnessed ambiguity
   report before this transform can become a complete physical-world hypothesis.
8. A false row rejects that proposed pair. Reject the transform when this contradicts a required
   same-place anchor or leaves an independently named positioned landing without a consistent
   counterpart. Preserve null rows as missing conditional premises requiring source review; complete
   their evidence before claiming a resolved binding for that local mention.
9. Only after processing the table, consider local mentions with no proposed named counterpart.
   Read the source naming and identity scope: a chamber named only in the local survey may retain
   its source local label as its physical identity. Give that binding its computed position and source
   witnesses. A mention already represented by a candidate row stays governed by that row's result.
10. Assemble each surviving FrameHypothesis from the table-derived named bindings and the separately
    justified locally named chambers. Verify every positioned local mention has exactly one binding,
    every accepted named row appears in it, and its globalId and position agree with that row.
11. Attach transform witnesses, consumed assessment witnesses and the clause leaving the export
    selector unknown. Return all supported complete hypotheses. If missing identity premises block
    completeness, report those premises for source repair instead of claiming a complete world set.
