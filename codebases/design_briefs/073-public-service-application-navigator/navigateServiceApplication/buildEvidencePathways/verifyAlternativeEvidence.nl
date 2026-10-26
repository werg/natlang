---
description: Verify that an alternative document is explicitly accepted for the exact service condition and current rule version.
args: { dossier: NavigationDossier; change: ServiceChange; situation: ServiceSituation[] }
returns: EvidenceRoute[]
---
Require a current rule source naming the alternative type, relevant condition and verification authority. Record the rule version and the source of the applicant's document. If any mapping is missing, retain the alternative as an unresolved route and ask the named authority; do not present the document as accepted.
