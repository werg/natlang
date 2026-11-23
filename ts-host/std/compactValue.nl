---
description: Compact a large tool or intermediate value into a Neuralese state for downstream calls.
args:
  value: string
  instructions?: string
returns: Neuralese<string>
---
Write a compact Neuralese view of value that downstream calls can use in place of the full text.

Without instructions, preserve every detail needed to reproduce value faithfully—exact names, numbers, identifiers,
structure, and code—dropping only repetition and redundant phrasing.

With instructions, keep everything that purpose needs, including how value is organized and where any omitted details
live, and drop what the purpose does not need.
