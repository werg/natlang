# Storing and indexing neuralese

Neuralese values are what natlang programs pass between model calls when text would lose information: immutable
blocks of vectors written by one model channel (a *dialect*) and read by calls in the same dialect. A database that
holds them needs four things a text database does not.

## 1. Storage: content-addressed blocks, referenced by rows

- A column of type `neuralese` holds a block's content ID (`nz1_…`: SHA-256 over dialect, shape, dtype and bytes).
  The ID is the value: equal blocks are one block, and a row can be checked against its block at any time.
- Payloads live beside the tables: `blocks/<dialect>/<id>.safetensors` in the folder engine (the same layout the
  Python server's blocks use: natlang metadata in the header, the payload as tensor `payload`), or a
  `_nl_blocks(id PRIMARY KEY, dialect, length, width, dtype, data BLOB)` table in the SQLite engine.
- Rows are the references: deleting the last row that names a block makes it garbage; `FolderBlockStore.collect`
  removes blocks no row names and no running call has pinned. A transaction writes its blocks before its rows, so a
  committed row never names a missing block.

## 2. Dialect pinning

A block means something only to readers of its dialect. The catalog records the dialect of each neuralese column;
an insert of a block from another dialect fails the transaction, and an index answers only queries in its dialect
(`IvfIndex` refuses others, naming the fix: re-encode the query in the index's dialect). A backbone change is a new
dialect: per the foundation rules, the channel is requalified, stored blocks are re-encoded from their sources (the
log keeps the text a block was written from when there was one), and indexes are rebuilt.

## 3. Similarity indexing

`IvfIndex` (crisp, in `neuralese.ts`) pools each block to one unit vector (the mean of its rows) and files it under
the nearest of k centroids found by k-means; a search ranks the members of the closest few lists by cosine
similarity. It lives in `indexes/<table>/<column>.ivf.json` beside the ordinary indexes. Pooling loses order within
a block; it is a recall stage, and precise answers come from reading candidates (section 4). Larger collections would
swap IVF for HNSW or product quantization behind the same `add`/`search` contract.

## 4. Queries over meaning

Two kinds of question touch neuralese values:

- *Nearest neighbours* ("notes like this one", "orders whose instructions resemble this request"): encode the query
  (a stored block, or text written into the column's dialect by the serving model), search the index, return the
  rows that refer to the hits.
- *Conditions on meaning* ("notes that mention an allergy"): a decision call reads each candidate block as a
  `Neuralese<string>` argument; in the SQLite engine this is the same `meets(column, criterion)` mechanism text
  columns use, with the block read instead of the text, and judgments cached per (criterion, block ID). The index
  narrows the candidates first when the criterion can be encoded too.

Reading blocks needs a neuralese-capable model server in the column's dialect; until one is attached, the database
stores, references, collects and indexes blocks (tested with synthetic payloads), and conditions on neuralese columns
fail with a message naming the missing server.
