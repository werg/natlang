---
description: Compact a large call argument into a Neuralese state tailored to the receiving function.
args:
  value: string
  name?: string
  type?: string
  purpose?: string
returns: Neuralese<string>
---
Write a compact Neuralese view of the argument value for the receiving call.

When name, type, or purpose are given, keep every fact, identifier, structural boundary, and constraint that the
receiving call needs from this argument, and omit boilerplate or unrelated passages. When they are omitted, preserve
value faithfully while dropping only repetition.
