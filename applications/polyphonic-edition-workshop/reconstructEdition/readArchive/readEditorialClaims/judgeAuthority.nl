---
description: Judge authorization and scope of one original editorial act.
args:
  claimClauses: SourceClause[]
  roles: RoleIndex
  policyClauses: SourceClause[]
returns: AuthorityDecision
---
1. Capture the actual role index and governing policy clauses. Create a scoped NL AuthorityDecision
   literal over one actual claim's clauses and directly await its typed result. The literal can read
   the captured roles and policies; neither its instruction nor its answer invents an issuer.
2. Inside that literal, inventory each source-stated issuer, authenticated role or form, signature,
   authentication act and copying relationship for this claim. Associate every candidate attribution
   with its exact clause and quotation. An authenticated signed instruction can satisfy a policy
   without a personally named issuer when its established role and form meet the source rule; never
   invent that person's name. If a role is asserted by a later copy rather than established by the
   original act, retain that distinction in the candidate set.
3. For each candidate issuer, compare the governing policy's conditions to the claim's actual form:
   who may order a correction, whether a signature or adoption is required, and whether the document
   is an instruction, an unsigned player aid, or a report of an earlier performance. Use the source's
   own authority order; do not infer prestige from a voice name or a later clean copy.
4. Determine the operation and location scope that each still-supported authorization actually
   grants. A signed duration-only slip does not authorize a pitch correction; a general source role
   does not extend one slip to another event or section. Preserve narrow scope even if a later editor
   proposes a wider, musically plausible reading.
5. If one decisive attribution establishes authorization under the governing policy, or a nonempty
   set of all still-supported attributions establishes authorization for the same scope, return
   authorized=true with that scope, a reason and exact clause-keyed support. If a nonempty supported
   candidate set all establishes lack of authority, return false with the excluding rule and claim
   evidence. If the decisive attribution
   or policy is missing, or live candidates disagree about authority or scope, return null and name
   the unresolved premise; do not turn uncertainty into an authorization.
6. Check that the evidence array contains a nonempty quotation for every decisive claim and policy
   premise. Return the typed judgment to the parent. The parent consumes its actual authorized,
   scope and reason fields while constructing the claim; target attachment remains a separate
   decision even when authority is established.
