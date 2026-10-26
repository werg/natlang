---
description: Trace accepted terminology changes into released strings and dependent workflows.
args: { dossier: TerminologyDossier; change: TerminologyChange; decision: TerminologyDecision }
returns: MigrationPlan[]
---
Search released content by concept ID and prior terminology version, not string alone. Identify each workflow and dependent artifact that uses the old term. Group updates into a coherent version; flag mixed-version flows and require a complete review set. Preserve unaffected releases.
