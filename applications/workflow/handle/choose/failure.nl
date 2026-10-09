---
description: Decide whether the remote's refusal of an operation may succeed on another try.
readout: decision
args:
  operation: Operation
returns: Failure
---
operation is an attempt the remote refused; operation.detail says why. transient: the refusal came from load or timing
(a rate limit, a timeout, "try again later"), so the same request may succeed later. definite: the remote declined the
request itself (a declined card, no stock, an invalid address), so repeating it gets the same answer. A detail that
names neither is definite.
