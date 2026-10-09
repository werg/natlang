---
description: "A view of a value: a shorter form a reader uses in its place. Without instructions it is faithful (the value can be reproduced from it); with instructions it keeps what their purpose needs."
args:
  value: string
  instructions?: string
generic:
  R: string | Neuralese<string>
returns: R
---
Write a view of value: a shorter form of it that a reader uses in its place.

Without instructions, the view is faithful: keep everything needed to reproduce value exactly, its content, its structure and its exact details (names, numbers, identifiers, code), and drop only what repeats or can be restated more briefly without loss.

With instructions, they say what the view is for: keep what that purpose needs, including how value is organised and where in it the rest can be found, and leave out what the purpose does not need.
