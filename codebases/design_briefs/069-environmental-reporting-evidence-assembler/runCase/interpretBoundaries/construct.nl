---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftReportingBoundary
---
Construct a ReportingBoundary draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read the supplied reporting framework, organizational boundary, and amendments to determine included activities and required measures. Distinguish ownership, operational control, and other supplied boundary concepts rather than choosing whichever yields a favorable result.
