---
description: Verify appointment-specific consent and communication scope before disclosure.
args: { dossier: PreparationDossier; change: AppointmentChange }
returns: DeliveryLedger[]
---
Verify current consent names the recipient, appointment, message content and channel. A representative's prior consent is insufficient when scope or version changed. With missing, expired or withdrawn consent, block disclosure and preserve the undelivered task for authorized staff.
