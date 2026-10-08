---
readout: decision
args:
  first: string
  second: string
returns: Sameness
---
first and second are code that an executor ran for two calls of the same function, normalized: $in.<path> is that call's input, v0, v1 ... are local variables, $h0, $h1 ... are literals.

Answer "same" when both do the same work on their inputs: the same service calls with arguments taken from the same inputs, and the same result computed from them, even if written differently. Answer "different" otherwise.
