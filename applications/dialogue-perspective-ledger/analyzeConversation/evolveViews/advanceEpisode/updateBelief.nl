---
description: Transform actual endorsement into participant-owned belief evidence and defeats.
args:
  decision: EndorsementDecision
  act: ActReading
  episode: Episode
  view: ParticipantView
  source: DialogueSource
returns: BeliefDelta
---
1. Confirm actualDecision.participantId matches the supplied view and actId identifies the actual act.
   Require that actor had the act available at this episode or authored it; otherwise return unresolved.
2. For endorse posture create one witnessed BeliefAtom for each actual returned proposition/value,
   with this actor, acquisition time, actual act IDs and actual availability episode IDs.
3. Join corroborating verifiedFactIds only from this actor's currently availableVerifiedFactIds and the
   proposition being endorsed. Application-visible truth can test correctness, but cannot grant access.
4. A verified fact contradicting the endorsed value is not corroboration. Preserve that false belief
   and its actual access separately; do not attach the contrary fact as a knowledge witness or silently
   revise the endorsement. Later knowledge queries inspect actual currently available matching evidence.
5. When actual posture is doubt, retain a doubted atom rather than an endorsed negative belief. For
   report-only/none add no endorsed belief; preserve the source posture in the episode decision trace.
6. For unresolved posture retain an unresolved atom if a proposition domain is known, otherwise return
   a typed unresolved reason. Never fill unknown polarity with false to simplify downstream questions.
7. Translate actualDecision.defeatsActIds into this actor's existing belief IDs only. Preserve unrelated
   propositions, another participant's atoms and a mistaken belief the actor has not actually revised.
8. Return actual add/defeat arrays. Source wrappers and generated belief IDs are exact mechanics;
   endorsement and source entailment were owned by the scoped child, not a case-specific table here.
