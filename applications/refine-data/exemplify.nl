---
description: Write realistic values that satisfy a property, to seed near-miss pairs.
args:
  predicate: string
  base: string
  slot: string
  count: number
returns: Examples
---
Write count different values of type base, each of them predicate. They are values that the slot slot of an application
would hold, so make them realistic, and let them differ from each other in length, wording and detail. Return
{ values }.
