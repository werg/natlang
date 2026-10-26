---
args: {"order": "ShippingOrder", "offers": "RouteOffer[]", "delta": "RoutingDelta", "reservations": "CarrierReservation[]"}
returns: "LegAlternative[]"
---
Start a frontier at the shipment origin, carrying visited offer IDs and preserved appointments. Traverse the finite offered-leg inventory in departure order; never reuse a leg in a path. For each extension call testLegHandoff(order, path, nextOffer, delta, reservations). Keep only feasible handling and timing extensions, preserving rejected reasons. A destination-reaching path is an alternative, not a reservation. Retain unmet demand when no verified path survives; unknown capacity cannot complete a path.
