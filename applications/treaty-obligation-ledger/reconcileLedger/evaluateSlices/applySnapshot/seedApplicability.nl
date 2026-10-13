---
description: Join each norm with the requested trip, party roles and available evidence.
args:
  snapshot: Snapshot
  trip: Transit
  compiled: CompiledNorms
  archive: Archive
  authority: AuthoritySet
returns: Applicability[]
---
1. Enumerate norms whose action and named participants can concern this trip, including possible
   defeaters. Do not discard an exception merely because its decisive fact is presently unknown.
2. For each candidate call judgeScope(norm, trip, snapshot, archive, authority). Consume its actual
   key, applies value, condition facts and authority proof references; do not summarize away false reasons.
3. Join operator/destination roles separately from named parties. A reservation about Birch as debtor
   cannot match an obligation whose beneficiary happens to be Birch.
4. Keep the shared business key for alternative notice justifications while retaining distinct normId
   and applicationId. Two applicable clauses can support one required performance.
5. Return every candidate assessment. Unknown source scope must remain distinguishable from a
   definite mismatch in reach, effective interval, role or evidence condition.
