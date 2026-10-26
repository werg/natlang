---
description: Convert verified and pending pathway requirements into truthful, consent-bounded application tasks.
args: { dossier: NavigationDossier; change: ServiceChange; pathways: PathwayGraph; questions: NavigationQuestion[] }
returns: ApplicationWork
---
Call `bindHelperConsent(dossier, change)` before assigning any representative task. Keep access limited to the applicant's explicitly approved purpose and channel. For each form requirement, map only established applicant facts. Send unrepresentable or unsupported facts to clarification; never fill a false answer. Keep every document request tied to a pathway and completion condition.
