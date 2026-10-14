---
description: Judge authorization and scope of one original editorial act.
args:
  claimClauses: SourceClause[]
  roles: RoleIndex
  policyClauses: SourceClause[]
returns: AuthorityDecision
---
1. Capture the actual source roles and governing policy. Create a scoped NL AuthorityDecision literal
   and directly await it on the original claimClauses with those actual captured inputs.
2. The literal joins stated issuer/authentication to the source's correction policy. Separate signed
   editorial instruction, unsigned player aid and later copyist interpretation without inferred prestige.
3. It determines which operations/locations the authority actually covers. A general source role alone
   cannot broaden a specific slip's duration-only instruction into an unrelated pitch correction.
4. Return authorized=true for established scope, false for established lack of authority, null for
   missing decisive attribution, with a reason and nonempty exact clause-keyed support.
5. Consume the actual return. The parent binds original witness identities before constructing the
   claim; authority does not resolve an attachment ambiguity left in the source.
