---
description: Relate repeated templates and duplicate ingestion while preserving participant count and unique contributions.
args: { docket: ConsultationDocket; arguments: ScopedArgument[] }
returns: CampaignRelationGraph
---
Call `compareSubmissionText(docket, arguments)` and bind exact-text and template-language relations separately. Reconcile each participant identity once for participation volume, while retaining all independently submitted positions. Count template clusters as a separate statistic. Preserve unique reasons added to campaign text, and never discard a participant's statement because its opening language is repeated.
