---
args: { inventory: PartInventory[], shipments: SupplierShipment[], applicability: ApplicabilityPartition }
returns: InventoryDisposition
---
Bind rule = assignRevisionUseByConfiguration(inventory, applicability). Include rule as the useByConfiguration allowlist. Allocate stock only to configurations explicitly compatible with that revision. For a shipped obsolete part, retain shipment identity and require an owner-recorded accept, return, rework, or quarantine disposition; do not infer the supplier action.
