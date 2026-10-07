# Natural-language database format

A database is a folder. Everything in it is plain JSON or JSON Lines, readable by people and by natlang.

| Path | Holds |
|---|---|
| `catalog.json` | The schema: `{ "version": n, "tables": { name: Table } }`. A table has a `description`, its `columns` (name, type, description, nullable, unique, references, check), its `key`, the columns that have `indexes`, and `pages`, `rows` and `nextId`. |
| `tables/<table>/<page>.jsonl` | Rows, one JSON object per line: `{"_id": 7, "name": "Ana", "city": "Lisbon"}`. Pages are numbered 0, 1, 2, … and hold at most 256 rows each, ordered by `_id`. Every row has a unique integer `_id` and one field per column (`null` when absent). |
| `indexes/<table>/<column>.json` | A JSON array of `[value, _id]` pairs sorted by value, then `_id`: every row of the table, so a value can be found without reading the pages. |
| `log/<sequence>.json` | One record per committed transaction, written by the host before it changes the files: the request, how it was read (statements), what changed, and every changed file's new content. A commit interrupted after its record is finished from it when the database opens. |

Column types: `integer`, `real`, `text`, `boolean`, `date` (`YYYY-MM-DD`), `timestamp` (ISO 8601), and `neuralese` (the
content ID of a stored neuralese block; see `NEURALESE.md` in the natlang repository).

A transaction is one natural-language reducer call over this folder: its changes are kept only if the call
finishes, and only the host writes `log/` and this file.
