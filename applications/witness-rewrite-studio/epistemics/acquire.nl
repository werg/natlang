---
description: Fold observation and communication into scoped knowledge states.
args:
  intake: IntakeResult
  world: WorldIndex
returns: KnowledgeIndex
---
1. Inventory every explicitly described perception, reading, utterance and report in selected events.
   Resolve actor identity through world.entities and preserve the event/frame ownership of the evidence.
2. Make InformationEvents with factId, sender, receivers, mode, credibility and exact witness spans.
   Distinguish receiving the physical letter from reading it and hearing an accurate oral message.
3. For a read, require access to its contents. For observation, use the observer's stated viewpoint.
   For hearing, require an utterance and reception; merely being nearby is not always stated listening.
4. For a report, establish what the sender could know before sending. A speaker can intentionally
   mislead, but the recipient's resulting belief must not be recorded as independently verified knowledge.
5. Fold the finite event partial order into actor-local states before each dialogue/action event. Carry
   knownFactIds only across source-supported earlier events in the same actual narrative frame.
6. When two possible orders give different knowledge, intersect definite known facts and retain
   alternative beliefs and an unresolved explanation. Do not turn a possible prior read into a certain one.
7. Recollections supply remembered beliefs in their own frames. They do not give a selected-world
   listener information unless the selected scene actually recounts that recollection to the listener.
8. A transfer chain terminates at observation, reading or an explicitly given premise. Repeated
   reports forming a circle do not manufacture an informed origin.
9. Return acquisitions, states, empty dialogue and localDecisions arrays. Each state names actor and beforeEventId;
   this explicit location prevents using information learned after the line being checked.
