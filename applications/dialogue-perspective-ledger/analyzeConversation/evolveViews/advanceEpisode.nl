---
description: Apply one source episode through separate endorsement, belief, commitment and revision deltas.
args:
  frontier: PerspectiveFrontier
returns: PerspectiveFrontier
---
1. Remove the next original episode and copy current participant views. Apply its actual availability
   rows first, adding only returned turn/act/verified-fact IDs to the named participant's access sets.
2. For each actual episode act and relevant named participant call judgeEndorsement(act, episode,
   that view, frontier.source). Relevant means actual speaker or an access-bearing participant, not everybody.
3. Call updateBelief(actualDecision, act, episode, view, source). Consume its add/defeat arrays. Apply
   exactly those actor-owned belief effects, preserving independent contradictory beliefs unless defeated.
4. For performed promise acts at their actual original event, call updateCommitment(act, episode,
   issuerView, currentViews, source, scoped). Only the actual issuer view owns this commitment. For previously
   established commitments whose condition facts changed, call the same stage with their original act.
   Consume actual commitment additions/replacements; report-only acts cannot invent a new issuer promise.
5. For an actual correction or explicit adopted retraction call resolveRetraction(act, episode, view,
   actualDecision, source). Consume targeted defeat/retraction IDs; a sender's private intention does not
   cancel another audience's communicated commitment. Receipt alone does not apply adoption effects.
6. Each child sees the relevant original evidence and the actual current view. Do not pass a fabricated
   empty source or a synthetic participant who conveniently already knows the complete physical record.
7. Check every returned participantId and target ID belongs to this original view/act domain. Apply
   only typed effects to that participant; copy source witness identity exactly without semantic re-judgment.
8. Record EpisodeDelta containing the actual endorsement/belief/commitment/revision results and access.
   Append an immutable ViewSnapshot after all this episode's independent effects have been applied.
9. Return frontier with remaining original episodes, updated views/history/deltas and unresolved child
   reasons. Exact copy/set application is mechanism; the named child decisions own semantic changes.
