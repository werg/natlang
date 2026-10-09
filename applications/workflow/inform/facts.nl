---
description: Choose the facts about an order that the customer may be told, in plain words.
args:
  report: Report
  kind: MessageKind
returns: Facts
---
Pick what the customer needs to hear about report.after for a message of this kind.
1. order: report.after.order_id. amount: report.after.amount, as given.
2. summary: what happened to the order, in plain words about the order and its goods: kind "paid" says the payment went
   through; "shipped" that the order is on its way; "delayed" that confirmation of a step is taking longer than usual and is
   being checked; "problem" that a step could not be completed and is being put right; "cancelled" that the order is
   cancelled and the payment and reserved goods are returned.
3. next: what the customer can expect next, from after.phase and after.obligations: shipping follows payment; the check
   continues while an outcome is unconfirmed; the return is completed when an obligation is open. Give no date or
   time unless after.history holds one.
