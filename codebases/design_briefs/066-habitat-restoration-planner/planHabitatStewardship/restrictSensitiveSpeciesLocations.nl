---
args: { records: SensitiveSpeciesRecord[] }
returns: SpeciesDisclosurePlan
---
Apply each record's sensitivity and access rule independently. Keep exact coordinates only in an authorized conservation view; suppress or coarsen them for volunteers and public views. Preserve restricted record IDs so redaction does not look like ecological absence.
