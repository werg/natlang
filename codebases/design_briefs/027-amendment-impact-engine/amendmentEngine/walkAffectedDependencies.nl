---
args: { composition: ObligationComposition; evidence: EvidencePage[]; cutovers: EvidencePage[] }
returns: AmendmentImpactMap
---
Start from each changed obligation facet and traverse only evidenced dependency edges. Identify affected open tasks, future schedules, approvals and artifacts; record an explicit unaffected reason for adjacent items outside the reachable scope. For each unexecuted task after cutover, compare its approval scope to the current amendment interval and require revalidation when changed. Keep completed historical work and billing linked to its historical rule edition.