---
description: Reject routes that lose a required dramatic beat or disclose facts to an unauthorized group.
args: { candidate: VariantCandidate; dramaturgy: DramaturgyMap; resources: ResourceTimeline }
returns: RouteCheck
---
Compare the candidate's stated preserved beat with the scene's qualified beat, then trace each changed dependency through the reveal graph. A route fails if it reveals a fact early, tells one audience group what only another learned, or requires a declined participant. It remains unresolved if the new route lacks rehearsal evidence or the resource ledger lacks current status. A passing check means the route is eligible for cue planning; it does not authorize performance.
