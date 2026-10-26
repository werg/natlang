---
description: Bind representative access to current applicant consent scope and purpose.
args: { dossier: NavigationDossier; change: ServiceChange }
returns: string
---
Verify the named helper, purpose, allowed records, contact channels and consent version. If consent is expired, withdrawn or narrower than the requested action, do not disclose or assign that work. Preserve the blocked task and route it to the applicant or authorized staff.
