# nldb: a database you talk to

Requests are plain language. A request can define a table ("keep track of customers: name, city, when they
signed up"), make a change ("Ana from Lisbon signed up today", "move 50 from Ana to Ben, all or nothing"), or ask
a question ("which cities have more than two customers?"). The engine reads each one the way a database would:
schema, constraints, transactions and query plans.

Two engines share one interface (`Database.ask(request)` returns a schema change, a committed report, an answer, or
a clarifying question):

- **Pure (`FolderDatabase`, `database.nl`).** A database server written in natural language from end to end, over
  a documented folder format ([FORMAT.md](FORMAT.md), which is also copied into each database). Its parts follow a
  relational engine:

  ```
  database.nl             the server: classify the request, then hand it to the query processor or to schema definition
  database/classify.nl      question / change / schema / unclear (a decision; below p = 0.5 it asks back)
  database/clarify.nl       the one question that settles an unclear request
  database/parse.nl         front end: names resolved against the catalog, the SQL it amounts to, a first plan
  database/plan.nl          optimizer: push-down, index access paths, hash joins, join order, semantic filters last
  database/execute.nl       executor: runs plans; for a change, the writes, index maintenance and constraint checks
    execute/scan.nl           sequential scan over pages
    execute/lookup.nl         index lookup by binary search, reading only the pages that hold the rows
    execute/filter.nl         exact conditions, then conditions on meaning judged once per distinct value
      filter/meets.nl           does a value meet a criterion (a decision)
    execute/join.nl           inner, left, semi and anti joins; hash or nested-loop
    execute/aggregate.nl      groups and exact aggregates
    execute/project.nl        output expressions with SQLite's semantics; DISTINCT
    execute/sort.nl           ORDER BY, LIMIT, OFFSET
    execute/write.nl          page writer: _id assignment, 256-row pages, catalog counts
    execute/reindex.nl        index maintenance: sorted [value, _id] files
    execute/check.nl          constraint checker: types, not null, check, unique/key, foreign keys both ways
  database/define.nl        schema definition: catalog updates and migrations
    define/design.nl          schema design: tables, types, keys, constraints, indexes
    define/migrate.nl         rewrites stored rows and indexes for one change, and verifies new constraints
  ```

  The stages pass typed values to each other, declared in `types.ts`:

  - a `Statement`: a question's `Plan`, or a change's `Change`s;
  - a `Plan`: numbered `Step`s, one relational operator each;
  - `RowChange`s: a row's before and after images.

  A request is one directory-reducer call over the database folder. Inside it, each write stage runs through
  `folder.apply`, so a stage that fails keeps nothing. A question runs `execute` directly, so anything it writes is
  dropped. The only crisp part is the atomic commit. The host serializes requests, and it commits only a change that
  the server reports as done. Before saving the files, it writes the commit's redo record to `log/`, and it replays
  an interrupted commit when the database is opened.
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
tested with scripted models (`ts-host/test/nldb.test.mjs`). The pure engine's test follows each request through its
stages: dispatcher, define and design; parse, plan, execute, write, reindex and check; an index lookup chosen by the
optimizer. The live scenario run is pending; results will be
recorded here.
