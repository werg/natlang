# Preserve the teacher catalog while reducing its storage

The supervisor keeps the same admitted source population, extraction decisions,
canonical program bytes, source occurrences, completed file cursors and program
order. Compression changes storage only. It does not grant admission, repeat
reference qualification or replace the active teacher batch.

1. At a durable writer boundary, stop the catalog writer and retain its exact
   command and input authority. Keep immutable reference packets running.
2. Copy the closed catalog to a distinct target with sufficient disk headroom.
   Store program bodies with the shared lossless codec. Preserve every rowid,
   table, index and non-body value. Authenticate every body's canonical SHA.
3. Independently read the target and compare complete logical table digests
   against the source snapshot. Check database integrity, fsync the target and
   its directory, and publish a durable storage-only receipt.
4. Resume the explicitly bound storage-only scanner continuation using the
   shared encoder and decoder. All body readers accept both original text and
   the versioned compressed representation. Preserve source/protection guards.
5. Switch the logical database only after verification and closure of old
   handles. Remove the redundant old database only after the new durable copy
   and its dependency checks succeed. A failed copy never replaces the source.

These numbered decisions are supervisor policy in natural language. The crisp
helper implements exact byte encoding, SQLite transactions, row/hash comparison
and durable receipts. It never stops processes, switches pointers or deletes
the source. Storage floors and operation scheduling remain with supervision.

The actual first copy independently restored all 43,059 programs and 490,325
source occurrences: 10,104,705,024 bytes became 2,826,858,496 bytes. The live
writer is bound to a byte-identical frozen decoder so later mechanism maintenance does
not invalidate its historical copy authority. The public copier publishes its
closed immutable temporary file without clobbering a competing destination and
removes the temporary link before permitting mutable database use. The shared
artifact-binding helper streams hashes; it does not allocate an entire catalog
in anonymous memory merely to record its identity.
