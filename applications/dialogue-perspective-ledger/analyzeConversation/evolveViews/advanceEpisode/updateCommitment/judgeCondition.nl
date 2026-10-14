---
description: Judge actual fulfillment separately from issuer knowledge and beneficiary belief.
args:
  predicate: ConditionPredicate
  act: ActReading
  episode: Episode
  views: ParticipantView[]
  source: DialogueSource
returns: ConditionReading
---
1. Capture the actual condition predicate with its strictness, version, actor/object and source witnesses,
   this episode horizon, source physical facts and the actual issuer/beneficiary views selected by original actor IDs.
2. Directly invoke a scoped NL ConditionReading literal. Its local policy first interprets the exact
   required event, then joins actual bounded physical evidence, then evaluates perspective fields separately.
3. For collection before a strict cutoff compare actual source occurrence with that established cutoff.
   Inspect occurrences only at or before this episode. Before that window closes, no occurrence leaves
   actual null with a source-resolved pending reason, rather than a failed condition or unknown meaning.
   A reported claim can be believed without establishing actual collection. A complete negative log can
   settle false only within its actual completeness horizon; missing event evidence otherwise stays null.
4. For personal acknowledgment join the required actor and exact brief version. Acknowledgment of one
   version is not transferable to another. Keep different predicates separate rather than one vague consent.
5. For role fulfillment interpret the actual promised consequent independently from activation.
   Apply its source day/hour due scope: an unspecified hour remains null, while a later-day event is late.
   Complete bounded absence can establish false timely fulfillment; incomplete absence remains null.
6. issuerKnows needs actual issuer endorsement plus available verified support. beneficiaryBelieves
   examines actual beneficiary endorsement in its own supplied view. Absence of a required view stays
   null rather than borrowing another actor's belief. Multiple beneficiaries remain unresolved if they differ.
7. Return the actual three-channel reading and source witnesses. A child false on actual fulfillment
   does not erase a participant's separately true but mistaken belief in the prerequisite.
