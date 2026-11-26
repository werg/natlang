# Publish release catalogs to their data consumers

Inventory before editing: `collect-live-workflow-teacher.mjs --list` produces the
source-bound catalog; campaign activation publishes its local current copy.
`vibestudio_campaign._drive` reads that copy through `campaign_scope`, but only
published `status.json` and `reused.json` through the existing `sync_index` archive
transport. The remote drainer/exporter reads `ARCHIVE/catalog.jsonl`, and
`live_campaign_ingest.holdout_registry` compares it with its immutable WORK
snapshot. `expert_iteration.live_holdout_registry` and `sft_sources` consume that
snapshot. Their goal/split/group validation and metadata compatibility mechanisms
are unchanged. Native reference registries are separate fixture inventories, not
the campaign catalog. All model engines share this data path; none implements a
second catalog publisher.

Acted on: after validating the source catalog and all completion seeds, publish
the current catalog through the same idempotent, checkpoint-aware archive
transport before archival/model work. This makes later review/entry metadata
available to exporters and lets the existing ingestion compatibility check record
its exact source/snapshot hashes. It does not rewrite the immutable held snapshot
or change goal IDs, text, splits, groups, captured trajectories or native IR.
`NATLANG_CAMPAIGN_SYNC_CATALOG=0` retains externally managed archive metadata.

Controls verify ordering and that explicit opt-out, with no model/network calls.
Current live immutable drivers enter this fix at the next normal resume; their
running captures are not modified. Archive metadata is synchronized separately
for the already-active Source37b cutover.
