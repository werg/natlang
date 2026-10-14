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
   A reported claim can be believed without establishing actual collection. A complete negative log can
   settle false only within its actual completeness horizon; missing event evidence otherwise stays null.
4. For personal acknowledgment join the required actor and exact brief version. Acknowledgment of one
   version is not transferable to another. Keep different predicates separate rather than one vague consent.
5. issuerKnows needs actual issuer endorsement plus available verified support. beneficiaryBelieves
   examines actual beneficiary endorsement in its own supplied view. Absence of a required view stays
   null rather than borrowing another actor's belief. Multiple beneficiaries remain unresolved if they differ.
6. Return the actual three-channel reading and source witnesses. A child false on actual fulfillment
   does not erase a participant's separately true but mistaken belief in the prerequisite.
