---
description: Carry out a change request against a database folder, as one transaction.
kind: directory-reducer
args:
  request: string
  catalog: Catalog
returns: Report
---
Carry out request as one transaction on the database in folder; catalog is its current catalog.json. First write
the request as relational statements (INSERT, UPDATE, DELETE, with the rows or conditions each means). Then do each
one: find the affected rows (through an index when one fits, otherwise by reading the pages), and change the pages:
a new row gets the table's nextId as _id and goes on the last page, or a new page once that one holds 256 rows.
Keep every index file sorted and in step with the rows, and the catalog's rows, pages and nextId current.

Before finishing, check every new or changed row against its table: types, not-null, unique, check conditions and
foreign keys (the referenced row must exist; a delete must not leave rows referring to a deleted one). If anything
fails, or the request cannot be carried out as a whole, fail the call with the reason: none of its changes are kept.
Return the statements, the rows inserted, updated and deleted per table, any assumptions, and a one-sentence summary.
