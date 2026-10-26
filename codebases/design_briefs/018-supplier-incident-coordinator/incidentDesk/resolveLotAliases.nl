---
args: { request: SupplierIncidentRequest, evidence: EvidencePage[] }
returns: IncidentGuardrail
---
For a supplier lot renumber, bind old and new lot identifiers through an authenticated supplier mapping and receiving trace. Preserve both identifiers and original evidence; do not infer continuity from similar labels.
