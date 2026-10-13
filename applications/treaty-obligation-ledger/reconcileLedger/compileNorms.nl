---
description: Compile source-supported duties and scoped changes while retaining rejected instruments.
args:
  archive: Archive
  authority: AuthoritySet
returns: CompiledNorms
---
1. For each issued instrument, identify its required act power and source-time scope. Match only
   authority proofs that grant that issuer the relevant competence; preserve partial scope clipping.
2. Reject draft acts and unsupported office tariff claims as controlling norms. Keep their exact
   source clause IDs in rejectedClauseIds. A mixed circular can contain an unauthorized tariff clause
   and a valid notice-wording suggestion; classify each act without rejecting the whole document.
3. Call classifyChange(instrument, archive, authority). Consume precise target clauses and actor
   scopes; a reservation about Birch's own acts does not target another debtor's undertaking.
4. For every supported substantive clause call interpretConditions(clause, archive). Construct a
   Norm with witnessed roles, action, effect, amount, conditions, authority proofs and effective interval.
5. Represent a rate amendment as a scoped replacement of the target amount, a rescue exception as
   conditional exemption/permission and a closure as a defeater of ordinary passage permission.
6. Preserve N_NOTICE and an effective board notice as alternative justifications of the same business
   obligation. Clause provenance stays separate even when one performance discharges both duties.
7. Return norms and changes, rejected IDs and unresolved interpretation. Every normative consequence
   requires a supported source act; repeated words and document confidence supply no missing authority.
