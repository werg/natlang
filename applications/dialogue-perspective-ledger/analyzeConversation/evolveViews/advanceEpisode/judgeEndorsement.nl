---
description: Judge one actor posture without conflating receipt, quote and belief.
args:
  act: ActReading
  episode: Episode
  view: ParticipantView
  source: DialogueSource
  scoped: ScopedConversation
returns: EndorsementDecision
---
1. Capture the actual act meaning, original speech substring, participant's actual access, episode clock
   and source endorsement rule and actual scoped act inventory. Exclude final expected belief answers and application-global truth repair.
2. Construct and directly invoke a scoped NL EndorsementDecision literal for this participant/act pair.
   First determine whether the actor authored this act or only encountered its content at this episode.
3. Return basisActIds for the actual accessible assertion/utterance this actor endorses, alongside
   its own current act. A private note adopting an earlier claim retains that earlier actual scoped act
   identity, rather than losing the claim-source link when the note becomes a belief atom.
   Explicit adoption or a signed private belief can endorse actual per-proposition claims, including its owner's private intention. That does not communicate
   cancellation to an audience or replace a commitment. Merely hearing, receiving,
   quoting or forwarding preserves access without endorsing the original assertion.
4. Doubt returns posture doubt and the actual questioned proposition, not belief in its negation.
   Irony alone does not certify a literal proposition; unsupported attribution stays unresolved.
5. A correction adopted by this actor can identify superseded actual act IDs. Join original turn
   references to the actual scoped inventory and return adoptsActIds for the exact revision being adopted. Sender endorsement is
   not recipient adoption; an arriving correction leaves an earlier endorsement active until actual revision.
6. A conjunction can affirm one proposition and deny another. Copy each actual supported claim/value
   separately; a mentioned future promise consequent does not become an endorsed present-event fact.
7. Return actual per-proposition claim values, posture, basis/adoption IDs and defeat targets with nonempty exact original quotes.
   The parent binds source identity; updateBelief consumes the result without repeating the posture judgment.
