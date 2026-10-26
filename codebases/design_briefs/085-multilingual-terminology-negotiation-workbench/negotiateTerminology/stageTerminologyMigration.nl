---
description: Stage a versioned migration of an accepted terminology decision into released content.
args: { dossier: TerminologyDossier; change: TerminologyChange; decision: TerminologyDecision }
returns: MigrationPlan[]
---
Call `traceReleasedTermUsage(dossier, change, decision)` for every affected product and workflow. Stage one coherent versioned change and identify any workflow that would contain mixed old/new terminology. Require review of the complete migration before altering published content. Keep prior versions available for rollback and audit.
