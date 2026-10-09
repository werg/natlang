---
description: Decide whether what just happened to an order is something the customer should hear about.
readout: decision
args:
  report: Report
returns: Moment
---
Compare report.before with report.after (the order before and after the event) and decide what the customer should hear,
taking the first that holds.
paid: the payment became done (after.history has a done charge that before.history lacks).
shipped: after.phase is shipped and before.phase was not.
cancelled: after.phase is released (the payment and the reserved goods are both returned) and before.phase was not.
problem: an operation failed for good, or a compensation failed (after.obligations names one).
delayed: after.pending is not empty and before.pending was empty, so an outcome is waiting to be confirmed.
none: anything else, including internal steps such as reserving stock, looking again, and waiting.
