# Exact compressed text corpus storage

The natural-language operation `Prepare the next admitted training union -> ImmutableCorpus` keeps source,
split and quality decisions in the existing union policy and admission receipts. Storage pressure changes the
representation, not those decisions or model-visible rows. D13 contributes5.5B/7.0B program-header tokens;
the R6 LFM program-header output exceeded43GB while other twins were still unwritten (2026-10-14 DGX).

1. Select storage locations from current machine headroom: supervisor policy. Preserve canonical logical paths,
   and route only unopened outputs through filesystem symlinks. Do not move files still owned by a writer.
2. Compress a completed immutable text file: crisp zstd service. Record original bytes/SHA-256 and compressed
   bytes/SHA-256, and independently decode and hash before replacing the raw representation. Register actual
   compressed paths in manifests and recipes. Keep one useful representation; old raw path compatibility must
   never disguise compressed bytes under a plain-text extension.
3. Read either representation: extend common/jsonio with one exact UTF-8/literal-LF stream opener. Reuse it in
   direct streamed reads, warm-up eager reads, common JSONL reads and union assembly. A complete read must
   reject a failed/truncated decompressor; a deliberately partial read closes its subprocess safely. Preserve
   embedded Unicode line/paragraph separators, CRLF and long rows.
4. Verify semantic identity: exact bytes after decoding, plus the existing tokenizer-twin, split/source-review and
   schema checks. Compression/copy/schema checks do not add training admission. A cache keys the physical
   input bytes, so a representation change cannot reuse a stale row index.
5. Publish the new immutable storage binding and transfer through existing corpus synchronization. This is
   exact artifact plumbing, not a new trainer, sampler or model-facing rendering. No live prompt changes occur.

Training proceeds only from registered actual file bindings. The R6 build remains raw until its closed outputs
are verified and their storage conversion is recorded; a symlink location by itself is not admission.
