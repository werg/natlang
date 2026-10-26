---
args: {"document": "EvidencePage", "mandate": "MandateBoundary"}
returns: "QuoteOption[]"
---
Partition one document by supplier, requirement, quote revision and validity interval. Distinguish unit price, total, freight, service and conditional discount. Bind each extracted value to exact evidence IDs; preserve unreadable or incompatible units as blockers. Separate a proposed capacity from an actual binding reservation. Return options only for mandate-authorized suppliers and list excluded scope in blockers rather than leaking it to their rivals.
