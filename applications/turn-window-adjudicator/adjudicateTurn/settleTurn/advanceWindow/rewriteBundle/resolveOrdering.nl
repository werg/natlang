---
description: Resolve replacement precedence and shared capacity without treating simultaneous outcomes as sequential.
args:
  bundle: EventBundle
  beforeBoard: BoardSnapshot
  source: TurnSource
  protocol: WindowProtocol
returns: OrderPlan
---
1. Inventory standing and replacement readings relevant to the actual effect kinds and original
   lineages. Allocate one application identity per source-permitted rule/lineage use. Keep potentially
   false applications: their returned negative decisions explain why an effect survived unchanged.
2. Interpret each capacity owner. Once-per-lineage history and an artifact's shared turn token are
   different scopes. Planning neither spends the token nor decides who will obtain it.
3. Gather clauses establishing precedence, chooser authority and actual submitted relative-order
   choices separately. Permission to choose does not supply a choice; an example is not a submission.
4. Call enumerateLegalOrders(applications, bundle, beforeBoard, source, protocol). Consume its actual
   constraints, complete alternatives and unresolved legality findings. Source-required chronology
   remains distinct from proven independent display order.
5. Check candidate identities against the inventoried finite set: each application appears once and
   every returned directed constraint holds. These exact checks cannot invent a missing priority.
6. Return the actual OrderPlan with source choice clauses. Later actual previews establish whether
   legal alternatives differ; do not predict a selected order from an expected final board.
