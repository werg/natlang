---
args: { plan: RedactionPlan, receipts: RedactionReceipt[], inspections: DocumentInspection[] }
returns: RedactionReceipt[]
---
Require a sanitizer receipt binding input/output hashes, redaction map and engine version. Verify every text/OCR layer, filename, embedded metadata and attachment property was sanitized or safely renamed. A black rectangle in the visible page does not prove hidden text or metadata removal; without an exact receipt, withhold that artifact.
