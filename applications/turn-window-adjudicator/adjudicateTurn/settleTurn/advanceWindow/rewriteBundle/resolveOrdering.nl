---
description: Resolve replacement precedence and shared capacity without treating simultaneous outcomes as sequential.
args:
  bundle: EventBundle
  beforeBoard: BoardSnapshot
  source: TurnSource
  protocol: WindowProtocol
returns: ReplacementOrder
---
1. Build candidate rule/lineage applications from the bundle and actual standing/replacement meanings.
   Preserve explicit source precedence between rules within each lineage. Every condition reads the
   same beforeBoard unless its own rule explicitly reads the current rewritten quantity or spent use.
2. Distinguish separate capacities: a once-per-lineage standing rule may apply independently to sibling
   parts; one artifact's turn-use token is shared across those parts. The application frontier tracks
   those declared scopes rather than resetting every capacity at each part.
3. For unordered applications, judge whether swapping them could change effect kind, quantity, target
   or shared-token availability. Independent reductions on distinct lineages can commute; competition
   for one shared token needs a source priority or a submitted legal choice when it changes the result.
4. Use only supplied before/after rules and actual submitted choices to resolve consequential order.
   Retain a missing choice as unresolved. A deterministic storage order is allowed only after the
   source judgments establish semantic independence, not as a way to decide who receives protection.
5. Return one finite ordered application list respecting those findings, plus exact ordering witnesses
   and unresolved choices. Planning this replacement order does not commit any damage, destruction,
   token use or after-batch consequence; applyReplacement still judges each actual current part.
