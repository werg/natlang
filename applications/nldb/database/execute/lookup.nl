---
description: Index lookup. Find a table's rows by the values of an indexed column, reading only the pages that hold them; then keep those that meet where.
kind: directory-reducer
args:
  table: string
  alias: string
  column: string
  values: Value[] | null
  range: "{ low: Value, high: Value } | null"
  where: string | null
returns: Row[]
---
Read indexes/<table>/<column>.json in folder: a JSON array of [value, _id] pairs sorted by value, then _id. Find the
matching pairs by binary search: each of values exactly, or every value from range.low to range.high inclusive (a
null bound is open on that side). Their _ids are the rows wanted.

Fetch those rows from tables/<table>/ without reading every page. Pages hold rows in _id order, so a page holds the
_ids from its first line's to its last line's. Find the pages whose span covers a wanted _id, read only those, and
take the wanted rows. Rename each row's fields alias.<field>, _id included. Keep the rows that meet where (as scan
does), in _id order. Change no file.
