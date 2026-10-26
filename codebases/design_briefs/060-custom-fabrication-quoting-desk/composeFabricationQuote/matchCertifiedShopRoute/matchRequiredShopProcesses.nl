---
args: { concept: ConceptSelection, equipment: ShopCapability[] }
returns: CapabilityMatch
---
Match every concept operation to active shop equipment and capacity. Preserve unsupported operations as route gaps and keep the fabrication route in process order; do not infer capacity from an equipment name alone.
