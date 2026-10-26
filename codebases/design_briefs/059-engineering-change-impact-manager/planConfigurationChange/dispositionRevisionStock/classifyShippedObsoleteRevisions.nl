---
args: { shipments: SupplierShipment[], allowlist: string[] }
returns: ShipmentDispositionReview
---
Compare each received or in-transit shipment's exact part revision with the configuration allowlist. Identify old revisions, retain shipment IDs, and require an accountable disposition for incompatible stock; never silently substitute a current revision label for the shipped revision.
