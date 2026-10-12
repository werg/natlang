---
description: Quote the shipping price for an order.
args:
  order: Order
returns: Quote
uses: [helpers/rates]
---
Quote the shipping for order. An order has items (each with weightKg and quantity), a zone and a
subtotalCents.

1. weight = the sum of item.weightKg * item.quantity over order.items.
2. cents = rates.shippingCost(weight, order.zone, order.subtotalCents). The helper is crisp and
   follows these rules exactly:
   a. The billable weight is weight rounded up to the next whole kilogram, and at least 1 kg.
   b. Zone "domestic" costs 500 cents plus 150 cents per billable kilogram.
   c. Zone "international" costs twice the domestic price for the same billable weight.
   d. Domestic shipping is free (0) when subtotalCents is 5000 or more. International never is.
   e. Any other zone throws an Error with the message "unknown zone: <zone>".
   f. The result is a whole number of cents.
3. Return { weightKg: weight, cents, free: cents is 0 }.
4. Describe the quote to the customer in one friendly sentence, mentioning free shipping when it applies.
