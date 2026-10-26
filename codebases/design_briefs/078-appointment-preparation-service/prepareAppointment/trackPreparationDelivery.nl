---
description: Track checklist delivery and acknowledgment under the patient's current consent and approved fallback channel.
args: { dossier: PreparationDossier; change: AppointmentChange; checklist: PreparationChecklist[] }
returns: DeliveryLedger[]
---
Call `bindPreparationConsent(dossier, change)` before every recipient or representative transmission. Keep delivery attempts separate from acknowledgments. When delivery is undelivered, use only an approved fallback channel within the same consent scope; if no fallback is authorized, retain the status and ask staff for a safe route.
