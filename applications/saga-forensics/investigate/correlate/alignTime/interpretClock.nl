---
description: Resolve one clock mapping and its scope across restarts.
args:
  clock: string
  clauses: EvidenceAtom[]
returns: ClockRelation
---
1. Select the clauses that actually state a relationship involving clock. Keep clauses about rate, restart and
   epoch only when they change or affirm the applicability of that relationship.
2. Read equality language carefully: if local zero corresponds to global G, the global offset is G.
   If the clause says the local display is ahead of global time by D, the global offset is negative D.
3. A direct global clock has offset zero. A clock with only relative ordering information has null offset.
4. State the exact applicableScope in ordinary language, naming hosts or periods where the source does.
5. Set resetChangesOffset from explicit evidence; leave it null if the source is silent. A key epoch is not
   automatically a clock reset, and a process restart is not automatically an idempotency-domain change.
6. If compatible clauses jointly establish the mapping, carry all needed witnesses. If they conflict, return a
   null offset and explain both alternatives in applicableScope so the caller cannot fabricate a timestamp.
7. Return only this clock's relation. Event normalization and event-to-event comparisons belong to alignTime.
