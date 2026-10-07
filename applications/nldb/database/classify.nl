---
description: Decide what kind of request a database received.
readout: decision
args:
  request: string
  catalog: Catalog
returns: Kind
---
Decide what request asks of the database whose schema is catalog. question: it asks for information and changes
nothing. change: it records, corrects or removes facts in existing tables. schema: it asks to keep track of a new
kind of thing, or to add, rename or drop tables or columns. unclear: careful readers could take it more than one
of these ways, or it names things the database cannot hold.
