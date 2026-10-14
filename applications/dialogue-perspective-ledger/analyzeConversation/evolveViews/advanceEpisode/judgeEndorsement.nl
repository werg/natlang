---
description: Judge one actor posture without conflating receipt, quote and belief.
args:
  act: ActReading
  episode: Episode
  view: ParticipantView
  source: DialogueSource
returns: EndorsementDecision
---
1. Capture the actual act meaning, original speech substring, participant's actual access, episode clock
   and source endorsement rule. Exclude final expected belief answers and application-global truth repair.
2. Construct and directly invoke a scoped NL EndorsementDecision literal for this participant/act pair.
   First determine whether the actor authored this act or only encountered its content at this episode.
3. Explicit adoption or a signed private belief can endorse a proposition. Merely hearing, receiving,
   quoting or forwarding preserves access without endorsing the original assertion.
4. Doubt returns posture doubt and the actual questioned proposition, not belief in its negation.
   Irony alone does not certify a literal proposition; unsupported attribution stays unresolved.
5. A correction adopted by this actor can identify superseded original act IDs. Sender endorsement is
   not recipient adoption; an arriving correction leaves an earlier endorsement active until actual revision.
6. Return actual proposition IDs, value, posture and defeat targets with nonempty exact original quotes.
   The parent binds source identity; updateBelief consumes the result without repeating the posture judgment.
