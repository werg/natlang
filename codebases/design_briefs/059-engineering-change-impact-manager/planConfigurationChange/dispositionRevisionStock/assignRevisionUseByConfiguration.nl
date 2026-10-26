---
args: { inventory: PartInventory[], applicability: ApplicabilityPartition }
returns: string[]
---
Produce the exact revision-to-configuration allowlist from supplied intendedConfigurations and the approved change partition. Preserve unrelated inventory without reassignment.
