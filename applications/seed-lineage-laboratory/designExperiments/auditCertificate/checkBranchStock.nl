---
description: Check each chosen policy branch against original material and experiment capacities.
args:
  stocks: StockIndex
  policy: PolicyResult
returns: AuditFinding[]
---
1. Traverse the chosen reachable policy from rootNodeId using a finite visited-node frontier. Resolve
   each actual selection and transition; all positive-probability answer branches must be represented.
2. For each selected experiment, check its original slot is present in the parent and absent in each
   child. Generated experiment or node names cannot replenish that original capacity.
3. Compare each child's actual stock with a fresh parent snapshot minus the selected experiment's
   interpreted requirements. A sibling branch must not observe another hypothetical child's subtraction.
4. Interpret any returned restoration or reuse claim against the source stock rules. Exact arithmetic
   checks the selected meaning, while a missing material permission remains a witnessed unresolved issue.
5. Preserve source-based findings for negative stock, duplicate original slot use, missing branches
   and unreachable references. Return all findings without changing the proposed policy.
