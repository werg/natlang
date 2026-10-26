---
args: { prior: SupplierIncidentCheckpoint|null, request: SupplierIncidentRequest, trace: SupplierTraceFile, clusters: StageResultIncidentCluster, exposure: StageResultExposureFinding, containment: StageResultContainmentPlan|null, correction: StageResultCorrectiveAction|null, closure: StageResultIncidentClosure|null, notice: IncidentGuardrail }
returns: SupplierIncidentCheckpoint
---
Create a checkpoint for the identified incident and preserve all original report and lot identities. Link containment receipts to exact lots and exposed quantities; keep unresolved split-stock lineage precautionary. Record supplier responses separately from verified correction. A notice is delivered only with its actual recipient receipt; corrected notices supersede only the corresponding earlier notice. New recurrence evidence reopens causal review and affected closure without erasing prior incident history. Duplicate report and receipt identities are no-ops.
