# MiroVerse archive coverage verifier

The verifier answers one exact data question: whether the SFT JSONL members of
the pinned `zip_sft/MiroVerse-v0_1-SFT.zip` are already represented by the
separately acquired `jsonl_sft/*.jsonl` files.

1. Bind the immutable acquisition receipt and verify every listed source file
   against its recorded size and SHA-256 before using it.
2. Stream each JSONL source and each JSONL zip member one record at a time.
   Preserve raw-file hashes, row counts, canonical full-row hashes and source
   row ordinals. Do not extract the archive or retain row payloads in memory.
3. Compare each uniquely basename-matched archive member and source file as an
   exact multiset of canonical row hashes, so reordering is visible in the
   receipt but does not create false unmatched content. Record duplicate row
   identities and same-identity/different-payload conflicts separately.
4. Keep unmatched members, rows and ambiguous basename matches explicit. The
   verifier does not make training, quality, split or source-eligibility
   decisions; those remain with the existing acquisition and conversion
   pipeline.

The source-selection policy is natlang work. File hashing, zip/JSONL reading,
canonical comparison and durable receipts are crisp boundary and verification
mechanisms because they must preserve exact external bytes.
