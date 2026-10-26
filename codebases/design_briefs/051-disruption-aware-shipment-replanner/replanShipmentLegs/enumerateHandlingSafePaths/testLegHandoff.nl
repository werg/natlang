---
args: {"order": "ShippingOrder", "path": "LegAlternative", "nextOffer": "RouteOffer", "delta": "RoutingDelta", "reservations": "CarrierReservation[]"}
returns: "LegAlternative"
---
Compare continuity of origin/destination, transfer timing, item-class eligibility, required temperature/handling and explicitly available quantity. Call interpretHandlingCompatibility(order.handlingRequirements, nextOffer.handlingCapabilities, delta) before treating a capability label as equivalent. Check interval and quantity arithmetic exactly in compatible units. Preserve the path's distinct appointment consequences and reject a transfer that breaks an essential handling condition even if it arrives sooner.
