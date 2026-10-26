---
description: Reconcile application receipt and agency status before proposing any resubmission.
args: { dossier: NavigationDossier; change: ServiceChange; tasks: ApplicationTask[] }
returns: ReceiptReconciliation
---
Check the agency status evidence for the exact application reference and submission time. A delayed receipt is unknown, not a failed submission. If status is unknown, keep the application on hold and request recipient readback. Mark safe to resubmit only when an authoritative agency response proves no submission was received and the retry is within current policy. Preserve every acknowledgment and ambiguous delivery outcome.
