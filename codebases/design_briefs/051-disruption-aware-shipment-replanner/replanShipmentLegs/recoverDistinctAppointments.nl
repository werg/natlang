---
args: {"orders": "ShippingOrder[]", "prior": "DeliveryPromise[]", "notices": "Evidence[]", "priorityRules": "string[]"}
returns: "DeliveryPromise[]"
---
Recover contractual deadlines, customer-approved changes, preferences and dependent appointments separately. Keep several deadlines on one shipment as distinct obligations. Apply priority only from supplied policy and actual commitment authority, never from an assumed customer's commercial importance. Preserve an at-risk promise until renegotiation or actual fulfillment evidence changes it; do not silently move its deadline.
