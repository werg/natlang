---
description: Construct and settle the finite source-backed prerequisite and invalidation graph.
args:
  requirements: RequirementSet
  schema: StateSchema
returns: DependencyGraph
---
1. For each rule identify effects used by another rule's prerequisites. Create a requires edge only
   when that effect actually establishes the required proposition for the same actor and temporal scope.
2. Create invalidates edges for source-backed retractions or exclusive ownership changes. Preserve
   alternative support as alternative edges so removing one witness need not retract a surviving fact.
3. Inventory finite strongly connected components of these named rules using exact graph reachability.
   This graph mechanism does not decide whether the source establishes a semantic dependency.
4. Initialize ComponentFrontier with the inventoried components and original requirements. Invoke
   settleComponent.iterateOn(frontier) with measure remaining.length until no component remains.
5. Consume each returned orderedRuleIds and component reason. Positive source-supported recursion
   may have a finite readiness order; mutually conditional negatives without priority remain unresolved.
6. Return edges, settled components and unresolved. A topological ordering is an executable schedule
   only after its semantic edges have been established; rule/file ordering supplies no missing premise.
