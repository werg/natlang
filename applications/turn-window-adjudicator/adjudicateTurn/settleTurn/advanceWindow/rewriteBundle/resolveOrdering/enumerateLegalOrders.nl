---
description: Interpret ordering obligations and enumerate finite source-legal replacement orders.
args:
  applications: ReplacementApplication[]
  bundle: EventBundle
  beforeBoard: BoardSnapshot
  source: TurnSource
  protocol: WindowProtocol
returns: OrderPlan
---
1. Join named applications with actual rule meanings. Inspect pairs sharing a lineage, capacity or
   stated precedence. Different targets still need review when one turn-use token is shared.
2. For each pair capture its governing rules, capacity owners and timing clauses. In eval declare
   PairOrder as {relation:'first-before-second'|'second-before-first'|'unconstrained'|'independent'|
   'unresolved';reason:string;witnesses:Witness[]}. Directly await a scoped NL literal on the pair.
3. The literal interprets whether the source fixes precedence or permits choice. It returns
   independent only when effect and capacity scopes establish interchangeable execution. A same-lineage
   quantity or kind dispute remains unconstrained until actual previews show its consequences.
4. Parent-bind the actual relation to supplied application identities. Require nonempty exact source
   witnesses for established constraints or independence. Record source precedence as directed edges.
   An independent pair may use stable display order with basis independent-display; an unconstrained
   pair retains both legal possibilities. Missing source meaning remains unresolved.
5. Enumerate legal orders by structural recursion from an empty prefix and all identities remaining.
   At each step choose every remaining application whose predecessors are in the prefix, then recurse
   on the strictly smaller remainder. Empty remainder yields a candidate; no ready identity with a
   nonempty remainder exposes a cycle. Do not discard a source edge merely to complete enumeration.
6. Copy each branch's prefix and remainder. Bundle, beforeBoard and live tokens remain untouched.
   Deduplicate exact application sequences only; retain different orders for actual preview comparison.
7. Return applications, constraints, witnessed candidates, relevant original choice clauses and
   unresolved findings. Legality uncertainty cannot become a chosen order because it appeared first.
