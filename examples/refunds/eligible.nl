---
description: Whether a refund request may be settled at all.
args:
  request: Request
returns: boolean
readout: decision
---
true when the order was delivered at most 30 days ago, or when the customer reports that it arrived damaged or is not
what they ordered; false otherwise.
