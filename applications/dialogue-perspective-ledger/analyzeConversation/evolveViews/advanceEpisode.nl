---
description: Apply one source episode through separate endorsement, belief, commitment and revision deltas.
args:
  frontier: PerspectiveFrontier
returns: PerspectiveFrontier
---
1. If frontier.pendingEpisodes is empty, return frontier. Otherwise bind source to frontier.source and scoped to frontier.scoped, preserving those actual original
   objects. Remove the next original episode as episode and copy current participant views as currentViews. Apply its actual availability
   rows first, adding only returned turn/act/verified-fact IDs to the named participant's access sets.
2. Build this episode's posture work items from its actual new act IDs and the actual newly available act IDs in returned access rows. Resolve those IDs against scoped.acts, deduplicate each participant/act pair, and retain whether it is new authorship or encountered original content. For each item and relevant named participant call judgeEndorsement(act, episode,
   that view, source, scoped). Name the actual returned decision actualDecision, and preserve this
   accessed actor view as beforeBeliefView before belief changes. For a source turn whose actual occurredAt matches this episode and whose original ID occurs in episode.turnIds, relevant actors are its actual author and source-established present audience. This is new authorship even when a simultaneous physical/receipt event gives the coalesced episode that kind. At a later receipt episode, relevant actors are only the participants receiving that original content now. Use actual original turn membership and time to distinguish these cases; kind alone does not determine authorship. the attributed original speaker does not act again at a later receipt. A quoted historical speaker is not the author of the forwarding utterance.
3. From actualDecision.adoptsActIds retain only actual correction/retraction acts from scoped, then add
   those exact IDs to that actor's adoptedRevisionActIds after checking source
   access and actual scoped ID membership. Call updateBelief(actualDecision, act, episode, beforeBeliefView, source). Consume its add/defeat arrays. Apply
   exactly those actor-owned belief effects, preserving independent contradictory beliefs unless defeated.
4. For performed promise acts at their actual original event, call updateCommitment(act, episode,
   issuerView, currentViews, source, scoped). Only the actual issuer view owns this commitment; preserve already retracted original commitments
   when later facts or due boundaries are reevaluated. For previously
   established commitments whose condition facts changed or whose source due boundary this episode crosses, call the same stage with their original act.
   Consume actual commitment additions/replacements; report-only acts cannot invent a new issuer promise.
5. For an actual correction or explicit adopted retraction call resolveRetraction(act, episode, beforeBeliefView,
   actualDecision, currentViews, source, scoped). Consume targeted defeat/retraction IDs; a sender's private intention does not
   cancel another audience's communicated commitment. Receipt alone does not apply adoption effects.
6. Each child sees the relevant original evidence and the actual current view. Do not pass a fabricated
   empty source or a synthetic participant who conveniently already knows the complete physical record.
7. Belief defeats apply to RevisionDelta.participantId. Commitment retraction effects apply only to
   its returned commitmentOwnerId, whose actual issuer ledger contains those exact commitment IDs.
   This cross-view effect requires actual audience adoption evidence; recipient access alone is insufficient.
   Check every returned participantId and target ID belongs to this original view/act domain. Apply
   only typed effects to that participant; copy source witness identity exactly without semantic re-judgment.
8. Record EpisodeDelta containing the actual endorsement/belief/commitment/revision results and access.
   Append an immutable ViewSnapshot after all this episode's independent effects have been applied.
9. Bind next to the shorter typed frontier with remaining original episodes, actual updated views,
   history/deltas and unresolved child reasons. Finish this invocation immediately with
   eval({code:"return next;",finish:true}); the parent iterator owns the next episode. Exact copy/set
   application is mechanism; the named child decisions own semantic changes.
