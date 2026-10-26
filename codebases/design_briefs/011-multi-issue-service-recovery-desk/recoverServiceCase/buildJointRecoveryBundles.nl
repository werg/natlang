---
description: Compose jointly valid remedies without duplicate compensation or unauthorized promises.
args:{scope:GoalContract;policy:PolicySelection;issues:ServiceIssue[];entitlements:EntitlementFinding[];ledger:EvidencePage[];choices:EvidencePage[]}
returns:StageResultRemedyBundle
---
Bind alternatives = `evaluateIssueRemedyCombinations(issues, entitlements, ledger, choices)`. Compare complete bundles so a credit for one loss is not repeated, while separately entitled outcomes remain available. Enforce delegated amount limits and mutually exclusive remedies before presentation. A scarce replacement date requires a live reservation; a reservation expiring before customer consent removes that promise and reopens negotiation. Preserve each customer's explicit choice or pending status.
