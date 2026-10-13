---
description: Consume one rooted grant or one unsupported finite dependency component.
args:
  state: AuthorityFrontier
returns: AuthorityFrontier
---
1. If pending is empty return state. Select a root claim or a claim whose explicit parent claims
   already have supported proofs. Preserve source order only as a stable tie-break among ready claims.
2. For a charter root, verify that its witnesses actually establish a root grant, then append a proof
   with no parent proof IDs and its exact source scope. Remove this claim from pending.
3. For a nonroot ready claim, call assessCompetence(claim, parentProofs, archive). Consume permitted
   and narrowedInterval. A true result creates a proof referencing the actual parent proof IDs.
4. A false result records outside-scope; a null result records unresolved with its missing premise.
   Both consume the claim without inventing a grant. Its unsupported children cannot use it as authority.
5. If no claim is ready, call detectUnsupportedCycle(pending, proofs, archive). Consume its finite
   unsupported components or missing-root claims, record their reasons and remove precisely those claims.
6. Require pending.length to decrease. Return copied state with accumulated proofs/rejections; no
   citation can make itself ready by using the proof currently being constructed.
