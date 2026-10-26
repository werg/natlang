---
args: {"operation": "CarrierReservation", "notices": "Evidence[]", "receipts": "CarrierReservation[]"}
returns: "CarrierReservation"
---
Order actual carrier events by their evidenced operation sequence and version. Distinguish request received, booking accepted, item rejected, physical departure and confirmed cancellation. Preserve contradictions and unknown outcomes rather than selecting the most convenient status. Attach matching receipt IDs without inventing acknowledgment. A rejected refrigerated class cannot be treated as available ambient capacity for the same shipment.
