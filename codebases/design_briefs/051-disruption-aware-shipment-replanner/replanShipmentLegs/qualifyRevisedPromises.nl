---
args: {"allocated": "LegAlternative[]", "promises": "DeliveryPromise[]", "reservations": "CarrierReservation[]", "notices": "Evidence[]"}
returns: "DeliveryPromise[]"
---
For each appointment compare its approved deadline with the actual leg plan and matching accepted reservation receipts. A proposed capacity request is insufficient. Retract an unsupported promise after a carrier's class rejection and retain a recovery task. A changed appointment requires its actual customer/owner approval. Preserve the distinction between booked, departed, delivered and fulfilled; declare fulfillment only from the required delivery/appointment evidence.
