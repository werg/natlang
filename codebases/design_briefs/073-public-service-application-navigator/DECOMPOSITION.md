# 073 — Public Service Application Navigator

## Distinct algorithm: route a person through an evidence-permission graph, not an eligibility score

The navigator translates a person's stated situation into service pathways whose conditions each have explicit evidence routes. Alternative evidence is modeled as a permitted route for a particular condition, not as a general exemption. When a person lacks a standard document, the program follows only an alternative accepted by the current service rule and records which verification is still pending. It never declares ineligibility merely because a standard document is absent.

Application questions minimize burden by asking only for unresolved facts that change a pathway. A form field that cannot represent the person's situation becomes a truthful clarification task; the navigator never forces a false answer. Delegated helpers receive only the applicant-approved scope and access window. A service-rule change revalidates affected pathways and explains which requirement changed, without rewriting a completed application.

Submission is a receipt-reconciliation state machine. A delayed receipt leaves status unknown; the system checks the agency status channel before any resubmission to avoid duplicate filings. The resumable case preserves the rule version, evidence paths, delegation consent, question answers, application edition and agency receipt history.
