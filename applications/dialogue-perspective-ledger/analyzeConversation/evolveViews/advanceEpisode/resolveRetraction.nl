---
description: Target source-backed corrections in the actor who actually adopts them.
args:
  act: ActReading
  episode: Episode
  view: ParticipantView
  decision: EndorsementDecision
  views: ParticipantView[]
  source: DialogueSource
  scoped: ScopedConversation
returns: RevisionDelta
---
1. Select the actual correction/retraction content and its named superseded original turn/claim. Retain
   a private intention as intention without promoting it to a communicated audience revision.
2. Join decision to this actor and episode. A sender may endorse their correction at sending,
   while a recipient who only receives it has no adoption-based defeat yet under the visible rule.
3. Interpret the stated scope: correcting collection time need not retract the original collection
   promise, versioned acknowledgment or every proposition in the old utterance. Record the exact target.
4. Require actual access and explicit adoption before defeating this actor's endorsed belief. Select
   only their belief IDs supported by the actual superseded act/proposition lineage.
5. For a commitment retraction join the actual revision act to its named original promise using scoped.
   Locate that promise in its issuer's actual view, separately from the adopting participant's own view.
   Inspect each relevant beneficiary's adoptedRevisionActIds and its actual access to this same revision.
   Current decision.adoptsActIds supplies only this actor's adoption; it cannot invent another adopter.
6. Under this source model, all named promise beneficiaries must explicitly adopt the communicated
   retraction before the issuer-owned commitment changes status. Missing audience state remains
   unresolved; partial adoption records its actual adopter IDs without globally retracting the promise.
   A diary that says nobody was told supplies no communicated revision or audience adoption.
7. Return a generated revisionId only for an actual applied revision; otherwise null with a witnessed
   reason. Keep unchanged/doubted/unknown scope explicit and never invent global cross-actor defeats.
8. Return the actual RevisionDelta including actual adoptingParticipantIds and a commitmentOwnerId only for licensed issuer effects,
   plus original witnesses and unresolved missing premises. Parent
   applies belief IDs to this participant and commitment IDs only to the returned actual issuer owner and records the result in history, preserving before/after lineage.
