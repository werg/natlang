---
args: { inventory: PartInventory[], shipments: SupplierShipment[], applicability: ApplicabilityPartition }
returns: InventoryDisposition
---
Bind allowlist = assignRevisionUseByConfiguration(inventory, applicability). Bind shipmentReview = classifyShippedObsoleteRevisions(shipments, allowlist). Use the exact revision-to-configuration allowlist for each inventory allocation. A shipment of an old revision stays explicitly disposition-pending until the owner selects accept, return, rework, or quarantine; do not infer what happened.
