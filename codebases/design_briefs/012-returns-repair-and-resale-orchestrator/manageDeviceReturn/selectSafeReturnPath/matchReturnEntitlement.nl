---
description: Match item purchase scope and approved handling policy to allowed paths.
args:{scope:GoalContract;policy:PolicySelection;reports:EvidencePage[]}
returns:PurchaseHandlingMatch[]
---
Bind item and purchase to the exact effective policy edition. Separate consumer entitlement from safety restrictions and marketplace rules. Identify specialist checks, customer consent requirements and delegated authority. Missing purchase date or policy precedence remains unresolved; do not select the most permissive rule.
