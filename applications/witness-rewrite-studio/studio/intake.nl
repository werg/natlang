---
description: Separate immutable evidence from task authority and candidate proposals.
args:
  request: RewriteRequest
  snapshot: SourceSnapshot
uses: [readRecords]
returns: IntakeResult
---
1. Call readRecords(snapshot). Consume its inventory rather than classifying the same records again.
2. Call interpretRequest(request, inventory.instructions, inventory.canon, inventory.selectedScenes).
   The returned fact delta and preservation rules are the semantic authority for revision.
3. Check that every source ID has one path and every path has one source identity in this snapshot.
   Report conflicting identities as unknowns; do not silently prefer the last file in array order.
4. Keep proposal documents in inventory.proposals. Their desired repairs are alternatives to judge;
   they are not assertions that a scene already contains the proposed text.
5. Keep unclassified records visible in the inventory. A missing ownership cue is uncertainty about
   evidence scope. It does not authorize dropping a potentially relevant director restriction.
6. Check that the requested fact can be located in the selected scene or a controlling canon clause.
   If no matching fact is supported, append an explicit source-grounded unknown to meaning.unknowns.
7. Return {snapshot, inventory, meaning}. Preserve the exact original records and revision.
   Later branches may build normalized views but must retain paths and quotations for every join.
8. Identity checks concern input coherence. They do not settle whether a statement is true in-world,
   remembered by a character, quoted by another character, or introduced only as a proposal.
