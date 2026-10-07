# nldb: a database you talk to

Requests are plain language. A request can define a table ("keep track of customers: name, city, when they
signed up"), make a change ("Ana from Lisbon signed up today", "move 50 from Ana to Ben, all or nothing"), or ask
a question ("which cities have more than two customers?"). The engine reads each one the way a database would:
schema, constraints, transactions and query plans.

Two engines share one interface (`Database.ask(request)` returns a schema change, a committed report, an answer, or
a clarifying question):

- **Pure (`FolderDatabase`, `pure/`).** End to end in natural language over a documented folder format
  ([FORMAT.md](FORMAT.md), which is also copied into each database). `classify` decides what kind of request it
  is (a decision readout). `define` and `transact` are directory reducers: a transaction is one reducer call over
  the database folder, so its changes are kept only if it finishes. `query` is a read-only call on an overlay, and
  `clarify` asks back when a request is ambiguous. The host serializes writers. Before saving the files, it writes
  each commit's redo record to `log/` and replays an interrupted commit when the database is opened.
- **SQLite (`SqliteDatabase`, `sqlite/`).** `translate` compiles a request to SQL against the schema. A question
  must be one SELECT, and it runs inside a transaction that is rolled back. A change runs as one transaction, and
  every statement succeeds or none does. Conditions about meaning ("notes that mention a late delivery") become
  `meets(column, criterion)`. `judge` decides them once per distinct value before the query runs (`_nl_semantic`
  cache), because SQLite functions are synchronous.

Neuralese values (storage, dialect pinning, IVF similarity index) are designed in [NEURALESE.md](NEURALESE.md) and
implemented as `neuralese.ts`. Blocks are content-addressed safetensors files, and the index is per dialect.

## Running

```sh
natlang run applications/nldb -- shop.nldb                         # pure engine, one request per line
natlang run applications/nldb -- shop.sqlite "Which cities have more than two customers?"
natlang run applications/nldb -- shop.nldb --scenario scenarios/shop.json   # run and judge a scripted session
```

## Status

Both engines, the redo log, rollback, read-only questions, cached judgments and the neuralese store/index are
tested with scripted models (`ts-host/test/nldb.test.mjs`). The live scenario run is pending; results will be
recorded here.
