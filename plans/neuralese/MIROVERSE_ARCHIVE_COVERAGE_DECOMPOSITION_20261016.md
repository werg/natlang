# MiroVerse archive coverage verifier

The verifier answers one exact data question: whether the SFT rows in the
pinned `zip_sft/MiroVerse-v0_1-SFT.zip` are already represented by the
separately acquired `jsonl_sft/*.jsonl` files. The acquired archive currently
contains 15 split `.json` arrays, not `.jsonl` members; its splits therefore
must be compared as a global row multiset against all sibling JSONL files.

1. Bind the immutable acquisition receipt and verify every listed source file
   against its recorded size and SHA-256 before using it.
2. Stream JSONL sources and zip `.jsonl` members line by line. Stream top-level
   JSON arrays with an incremental decoder and bounded buffer. Preserve raw
   member hashes, row counts, canonical full-row hashes and row ordinals. Do
   not extract the archive or retain row payloads in memory.
3. Where filenames do not represent one-to-one files, compare the exact global
   multiset of canonical row hashes across archive shards and JSONL sources.
   This permits shard reordering while exposing duplicates or missing rows.
   Record same-identity/different-payload conflicts separately.
4. Keep unmatched members, rows, source files and unparsed archive payloads
   explicit. The verifier does not make training, quality, split or
   source-eligibility decisions; those remain with the existing acquisition
   and conversion pipeline.

The source-selection policy is natlang work. File hashing, zip/JSONL reading,
canonical comparison and durable receipts are crisp boundary and verification
mechanisms because they must preserve exact external bytes.
