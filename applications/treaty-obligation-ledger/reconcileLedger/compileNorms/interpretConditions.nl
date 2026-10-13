---
description: Preserve conjunctive prerequisites, open evidence and source-declared exceptions.
args:
  clause: Clause
  archive: Archive
returns: Condition[]
---
1. Separate the clause's membership, actor, reach, time and factual prerequisites. Keep each condition's
   subject, predicate, argument references and polarity instead of flattening all conditions into a slogan.
2. Distinguish an event condition from the evidence that establishes it. A valid rescue concerns the
   trip time, while availability of its certification concerns the knowledge snapshot.
3. Label evidenceMode affirmative when a positive or negative source assertion is required; use
   closed-world only where the source explicitly closes that docket. Otherwise retain open evidence mode.
4. Preserve exception nesting. A rescue exception can defeat a toll and override a flood closure
   without changing the separate notice or translation duty.
5. Return conditions with source-derived IDs and exact witnesses. Later applicability may evaluate
   them true,false or null; unknown is not silently converted into a negated predicate.
