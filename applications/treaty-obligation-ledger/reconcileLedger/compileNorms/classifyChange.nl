---
description: Classify amendment, reservation and interpretation by their actual legal target and authority.
args:
  instrument: Instrument
  archive: Archive
  authority: AuthoritySet
returns: ChangeRelation[]
---
1. Read the instrument's clauses and separate each proposed change relation. An accepted reservation,
   a prospective amount replacement and a suggested notice heading are different operations.
2. Call resolveTarget(instrument, clause, archive) for every proposed relation. Consume target clause
   IDs, actorScope and actionScope; unresolved referents prevent pretending the scope is universal.
3. Match the claimed act to issuer competence and the instrument's effective interval. Clip temporary
   orders to their parent competence; keep publication and adoption from changing the selected event boundary.
4. A wording interpretation preserves the underlying duty. A reservation changes only the accepting
   party's own undertaking as stated. An amendment changes only its named target from effective time.
5. Return ChangeRelations with authorityProofIds and exact witnesses. Use kind none for an issued
   text that merely describes a rule; do not manufacture a repeal from thematic similarity.
