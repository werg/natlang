---
description: Edit a value that satisfies a property so that it just fails it, changing as little as possible.
args:
  predicate: string
  value: unknown
returns: NearMiss
---
value is a value that is predicate: it satisfies that property.

Return a copy of value with the smallest change that makes it no longer satisfy the property, in edited. Change one detail
and keep everything else: the length, wording, format and structure of value stay as they are, so that a careful reader
has to check the property to tell the two apart. edited has the same type as value. In edit, write one short sentence
naming the change you made.
