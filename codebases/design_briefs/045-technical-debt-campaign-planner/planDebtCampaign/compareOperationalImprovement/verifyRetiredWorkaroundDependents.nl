---
args: { workaroundId: ID, runbooks: Evidence[], alerts: Evidence[] }
returns: WorkaroundRetirementCheck
---
Search each supplied runbook and alert record for references to the workaround. Confirm that every consumer has a reviewed replacement or an explicit removal decision. Keep retirement blocked while any active dependency remains unowned.
