---
description: Settle one dependency component without turning an unresolved policy cycle into a default.
args:
  state: ComponentFrontier
returns: ComponentFrontier
---
1. Select one original remaining component and remove it from a copied frontier exactly once.
   Inspect its internal requires, invalidates and alternative edges with their actual rule predicates.
2. For an acyclic component, derive orderedRuleIds from established dependencies, keeping independent
   branches independent rather than inventing a total source priority.
3. For a component with a cycle, read source exceptions, explicit priorities and finite base premises.
   Decide whether those meanings break the cycle or provide a monotone finite derivation order.
4. If no source-supported resolution exists, set status=unresolved-cycle, keep orderedRuleIds empty
   and explain the missing priority/base premise. The component still leaves the frontier, terminating
   iteration with a diagnostic rather than repeated truth toggles.
5. Append the actual settled component with its witnessed reason to settled. Keep alternative support
   distinct so a cyclic rejected path does not erase an independently grounded prerequisite producer.
6. Return copied remaining, settled, requirements and unresolved. The parent consumes this component's
   actual ordering into ModulePlans and scenario readiness selection, not a cosmetic depth receipt.
