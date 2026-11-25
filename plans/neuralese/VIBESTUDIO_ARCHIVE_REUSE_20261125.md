# VibeStudio seed archive reuse

The controller's `_drive` re-archives every admitted seed before collecting a new wave. Each `remote_archive` currently copies and then checksums the complete capture again, even when neither endpoint changed.

Mechanism inventory before editing:

- `vibestudio_campaign._drive` archives both seed episode directories and new raw waves through `remote_archive`; `transfer` owns rsync plus independent checksum verification. Reuse applies to seed episode directories; new waves retain their existing transfer and admission path.
- `remote_archive` also publishes immutable receipts and optionally launches export. A reused seed must still reach export when requested. Different existing receipts must fail before any capture is overwritten.
- `retained_state_archive.prepare_retained_state_packs` owns raw-state packing/exclusions, and `retire_ingested_native_state` owns recovery-pack retirement. Seed reuse does not restore or remove recovery state. Existing raw-wave behavior is unchanged.
- `retained_state_archive.inventory` supplies symlink-safe membership; `common.hashing.sha256_file_hex_cached(trust_archive_receipt=False)` already binds cached content to device/inode/size/mtime/ctime and detects restored-mtime rewrites. Reuse these implementations instead of adding a cache or custody index.
- `sync_index` only sends small mutable indexes and does not use seed reuse. `archive_transport` owns bounded idempotent IO retries; no teacher call is retried by this change.
- The native archive reader, student-fill consumer, registry admission and serving engines do not implement this transport mechanism. Their bytes and outcome checks remain unchanged.

Implementation: compare the exact receipt and content of both seed endpoints with strict cached digests, using one readonly remote helper invocation. Unchanged files need metadata reads, not content rereads. Missing or changed content takes the normal checksum transfer. `NATLANG_REUSE_SEED_ARCHIVES=0` disables reuse. Send only this small helper over SSH stdin, using the archive reader's existing Python/shared hashing implementation; do not copy a source tree or move collection.

The same inspection found that stat-only rsync sends could not repair same-size remote corruption with a restored mtime: the later checksum detected it but the preceding copy skipped it. Checksum sends now cover capture files, receipts and mutable indexes together. `NATLANG_ARCHIVE_CHECKSUM_TRANSFER=0` restores the previous sending behavior; independent post-transfer capture verification remains enabled.

Validation covers actual local rsync and helper execution, warm-cache reuse, same-size corruption with restored mtime, immutable receipt mismatch, the opt-out, and continued export after reuse.
