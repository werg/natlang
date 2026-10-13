---
description: Reconstruct only the before-announcement board and original turn-use state.
args:
  clauses: SourceClause[]
  rules: RuleIndex
returns: BoardSnapshot
---
1. Select explicit before-announcement declarations. Inventory current instanceId/cardId pairs,
   creature versus artifact kind, zone and controller. A later return rule describes a permitted future
   instance and must remain separate from an object currently present in the garden.
2. For each current instance, join its integrity, damage, growth and attachment declarations. Keep
   integrity null for a source artifact without an integrity value. Missing values remain unresolved;
   a convenient zero is allowed only when the initial-state contract explicitly supplies it.
3. Bind each attachment to the current instance named by its source. Ownership of the attachment
   and controller of the attached creature are independent fields. A future submitted move is not yet
   applied to this snapshot, even if both clauses occur in one file collection.
4. Join turn-use tokens to their source holders and distinguish available from already spent state.
   Keep a once-per-event-lineage replacement history separate from a shared one-use card token; they
   are different capacities and will be owned by different reducers later.
5. Read each player's initial resource balance and cite the declaration. Use source names exactly.
   Preserve only original before-announcement state; costs explicitly already paid do not deduct again.
6. Return revision, objects, resources, tokens and unresolved with nonempty exact original witnesses.
   These values become the immutable capture snapshot; downstream window changes create later state.
