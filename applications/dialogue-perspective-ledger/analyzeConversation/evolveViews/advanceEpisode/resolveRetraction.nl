---
description: Target source-backed corrections in the actor who actually adopts them.
args:
  act: ActReading
  episode: Episode
  view: ParticipantView
  decision: EndorsementDecision
  source: DialogueSource
returns: RevisionDelta
---
1. Select the actual correction/retraction content and its named superseded original turn/claim. Retain
   a private intention as intention without promoting it to a communicated audience revision.
2. Join actualDecision to this actor and episode. A sender may endorse their correction at sending,
   while a recipient who only receives it has no adoption-based defeat yet under the visible rule.
3. Interpret the stated scope: correcting collection time need not retract the original collection
   promise, versioned acknowledgment or every proposition in the old utterance. Record the exact target.
4. Require actual access and explicit adoption before defeating this actor's endorsed belief. Select
   only their belief IDs supported by the actual superseded act/proposition lineage.
5. For a commitment retraction require an actual source communicated revision and relevant audience
   adoption under the closed model. A diary that says nobody was told cannot satisfy this condition.
6. Return a generated revisionId only for an actual applied revision; otherwise null with a witnessed
   reason. Keep unchanged/doubted/unknown scope explicit and never invent global cross-actor defeats.
7. Return the actual RevisionDelta including original witnesses and unresolved missing premises. Parent
   applies these IDs to the supplied view and records the result in history, preserving before/after lineage.
