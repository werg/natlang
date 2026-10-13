---
description: Build request identities by combining payload meaning with the correct deduplication scope.
args:
  index: EvidenceIndex
returns: IdentityPartition
---
1. Extract every distinct named request attempt from the atoms. Populate Attempt fields from the source:
   order, system, operation kind, tenant or warehouse, epoch, key, quantity, unit, referenced effect and cause.
2. A retry described by reference may inherit the referenced request's expressly unchanged fields. Attach both
   the reference clause and the original request witness. Unknown scope fields remain null, never empty keys.
3. Collect explicit rejected attempts separately. Retain them in attempts but do not count their proposed amount
   as a successful effect. The rejection itself may confirm that a previous payment remains valid.
4. Construct the finite set of unordered pairs sharing a raw key within the same system, plus any pair the
   source explicitly calls a retry. Initialize groups to singleton nonrejected attempt IDs and unresolved empty.
5. Visit each pair exactly once. Call compareRequestMeaning(left, right, index.atoms). Consume its returned
   IdentityDecision: duplicate merges groups, independent keeps them separate, incompatible forbids a merge,
   unresolved records the missing proposition. Retain every decision, including cross-tenant/epoch separations.
6. Before merging, check all pair decisions already known across the proposed groups. A contrary decision
   prevents the merge and becomes unresolved; transitivity does not erase an explicit incompatible payload.
7. A replayed receipt is not an additional attempt unless the source names another attempt. The same effect ID
   can support several attempts without multiplying the effect. Adjudication will join those receipts later.
8. Return attempts, groups, pairDecisions, rejectedAttemptIds and unresolved. Keep group order semantically
   irrelevant. Source IDs within a group also form a set, with no preferred representative required.
