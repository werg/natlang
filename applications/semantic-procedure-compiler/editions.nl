---
description: Retain governed historical clauses and apply only supported scoped replacements.
args:
  intake: Intake
returns: EditionSelection
---
1. Extract clause identities, source issuer and stated adoption/effective intervals from original
   edition and substantive rule records. Keep document publication time distinct from effective time.
2. Call selectAuthority(intake.editions, intake.context). Consume its actual adoption/competence
   decisions with scope; a confident actor's recollection of a habit is not a adopted procedure.
3. Call resolveSupersession(clauses, authorityDecisions, intake.exceptions). Consume its selected
   temporally scoped clauses, replaced IDs and preserved IDs. A replacement of one prerequisite
   cannot erase unchanged custody/safety rules in the surrounding edition.
4. Retain each governing clause for the event interval in which it applies. A clause replaced from
   day5 still explains behavior on day4; keep that historical rule with effectiveUntil=5.
5. Exclude unsupported drafts/habits from normative selection while retaining their source evidence
   for scenario diagnosis. Source actors can act on mistaken rules without changing actual permission.
6. Return the selected clauses, replacement/preservation identities and authority decisions. Any
   unresolved decisive authority or replacement target remains explicit for downstream guard judgments.
