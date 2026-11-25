# Live catalog held identity (2026-11-25)

A Source36 review/result-type metadata update preserves all 2,327 registered user goals, but the standing ingester required byte equality with WORK's immutable catalog snapshot. This would reject a sound new source at ingestion. The snapshot remains unchanged; a shared comparison now validates semantic task identity and records exact source/snapshot hashes, canonical identity hash, comparator source hash and changed execution metadata paths in `WORK/catalog-compatibility.json`.

## Inventory before editing

- `scripts/neuralese_data/live_campaign_ingest.py:holdout_registry`: copies the initial catalog to WORK. Replace byte equality with checked compatibility; never rewrite that initial snapshot.
- `scripts/neuralese_data/sft_sources.py:live_holdout_registry`: preserves task ID, split, source-group closure and goal hash. Shared parser additionally validates actual goal values wherever provided, including redundant entry encodings. Existing cross-split prompt/group rejection remains unchanged. Compressed registry input remains supported through its existing `zstd_lines` reader.
- `sft_sources.py:require_registered_open_goal`: independently binds actual episode goal and source groups to the immutable training registration. Unchanged.
- `scripts/neuralese_data/expert_iteration.py:live_holdout_registry`: multiple live WORKs previously compared raw hashes too; now uses the same semantic comparison before selecting an original immutable snapshot.
- `scripts/neuralese_data/vibestudio_campaign.py:campaign_scope` and `inventory_live_teacher.py`: consume the shared registry validation. `remote_archive` transfers original captures/receipts; export reads archive root catalog. No rewritten wave evidence or catalog needed.
- `scripts/vibestudio/qualify_open_goals.py`: byte-pins the full executable catalog for an exact reference resume. That is executable provenance, not held identity, so remains strict.
- Source release `activate.py` scripts outside this repository update current catalogs under the parent's controller lock. They do not own the learner's immutable snapshot; no edits made.

Actual 2,327-row field inventory: taskId, interaction, split, sourceGroups, goalSha256, scenario, family, sourceGroup, workspace; requiresReview; entry; optional devices, requiresAndroidEmulator, requiresDeploymentPackage, workspaceBaseline. Each entry has program, callable, resultType, skill, arguments, argumentJson, parameterNames and inputValues.

Only the five explicitly named top-level execution/review fields and four entry execution fields may change. Their types are checked. Source groups compare by exact membership. Actual goal encodings must match goalSha256; their formatting/redundant presence does not change identity. Every non-goal input and unknown field remains protected, including unknown nested entry fields. Added, dropped, duplicated or changed tasks fail. No blanket unknown-field exemption or held split rewrite exists.

## Evidence

`vibe-live-catalog-identity-20261125a`: 88 tests passed, then quick gate passed; 27.8s/1.1GiB peak. Tests cover immutable snapshot and unchanged held closure/registration across metadata updates; added/dropped/renamed held tasks, changed split/group/interaction/hash, forged goal in each encoding, non-goal input changes, unknown fields, duplicate IDs and malformed metadata all fail. Multiple live WORKs share the same comparator.

Actual read-only Source35→36 comparison passed under `vibe-live-catalog-actual-20261125a`: 2,327 tasks, 20 metadata-only changed rows; identity SHA256 `3f9f65867953dc7b327f4f624a737b8b8a5c6717dae0224ebd5251439e4954d7`. Exact input hashes and comparator hash retained at `/home/werg/vibe-cross-task-custody-ops/source35-source36-catalog-compatibility.json`. No live ingestion, registry registration or model inference was run by these tests; production collection continued.
