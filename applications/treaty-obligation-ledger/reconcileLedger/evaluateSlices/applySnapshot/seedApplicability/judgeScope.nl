---
description: Judge one norm's event-time scope and snapshot-visible conditions with a scoped NL literal.
args:
  norm: Norm
  trip: Transit
  snapshot: Snapshot
  archive: Archive
  authority: AuthoritySet
returns: Applicability
---
1. Gather this norm's scope clauses, actual authority interval, accession/role facts and relevant
   trip evidence. Restrict evidentiary premises to those available by snapshot.knowledgeTime.
2. Capture the norm and source identity convention. In eval declare a local ScopeJudgment matching
   Applicability, create nl.with<ScopeJudgment>, and directly await it on the changing trip/snapshot premises.
3. Ask for separate evaluation of membership, debtor/beneficiary role, reach, event-time interval and
   condition facts. A definite failed scope condition makes applies=false; an unresolved decisive condition
   makes applies=null; all required supported conditions make applies=true.
4. Require witnesses and fact IDs for each condition, including the premise establishing when late
   evidence became available. Source content and its knowledge timestamp contribute distinct premises.
5. Consume the returned key, conditionFacts and applies value in the caller's candidate set. The
   literal must not choose a convenient event time or use certification unavailable in this snapshot.
6. Return the actual assessment with authorityProofIds. A valid low-scope grant does not authorize
   an unrelated action, and a later rate applies to later transits rather than later reviews of earlier ones.
