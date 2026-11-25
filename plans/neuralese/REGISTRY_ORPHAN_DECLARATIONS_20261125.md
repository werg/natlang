# Missing scratch registry declarations (2026-11-25)

Actual retirement warning, after useful ingestion had committed: strict path decoding refused an old absolute `/tmp/claude-1000/.../gdr/bench/work-A/step-chains` registration. Read-only DGX journal inventory found exactly two such dataset declarations, one ID, emitted by grpo.py at clean commit91b26036f78f44251f94865b19412d61bca54538 on November23 (PIDs4180062 and136214). The original artifact path is absent, no manifest exists, two file records remain, and no ID consumer was found. Source-hashed initial inventory: `/home/werg/vibe-registry-path-owner-ops/dgx-registry-path-inventory.json`. This is housekeeping metadata, not an ingestion or campaign failure.

## Inventory before editing

Shared path producers: data_registry.portable, snapshot, record_inventory, record_dataset, declare_dataset, _input_refs and consume, with _append as the final journal boundary. portable's ALLOW_UNREGISTERED_DATA opt-out previously admitted absolute scratch paths into the default machine journal. The GRPO automatic step-chain declaration exercised that path. Later strict absolute decoding correctly rejected it.

Consumers: Registry._paths/_lexical/role, sft registry loads, retained_state_archive's full active snapshot reference closure, promotion and recall tooling. They must distinguish immutable history from active registrations. Existing Git retired:true only filters role membership; supersedes is descriptive; the journal had only dataset/files/consume. No existing public tombstone or quarantine operation existed. Promotion consumes reg.datasets/reg.snapshots and therefore follows the common active projection. recall_to_nvme previously selected latest declarations directly from raw history; it now uses active snapshots so relocation cannot resurrect quarantined records.

## Shared correction

Absolute scratch producer/reader opt-out now requires both ALLOW_UNREGISTERED_DATA and an explicitly separate NATLANG_DATA_REGISTRY; setting it to the default machine journal does not isolate it. _append validates final dataset paths, including metadata overrides. Portable root-relative paths cannot escape their anchor with an absolute suffix or `..`.

One public append-only `quarantine_missing_dataset` operation (CLI `quarantine-missing`) binds exact canonical hashes of all active declarations. It refuses Git ownership, other-machine ownership, manifests, held kinds, protected/role entries, existing artifacts or symlinks, overlapping registrations, downstream inputs and consumers. A journal compare-and-append under the existing flock rejects concurrent changes and requires a fresh closure proof. Evidence binds the selected owner/ref, journal prefix hash, comparator source hash, missing paths and original producer identity. No original journal or file record is rewritten or deleted.

The common active projection excludes only those exact historical declaration hashes. Full journal provenance is retained. Any later valid declaration, even under the same ID, stays active. Native-state retirement continues checking all active snapshots, actual artifact references and consumer paths. There is no path-validation waiver or pack deletion in this metadata operation.

## Verification and operation

108 registry/retained-state/ingestion controls, including the held-kind negative, and quick gate passed. An additional relocation regression validates that an active sibling can move while a quarantined declaration stays inactive. Actual DGX application must use the ready isolated source and public API with independently observed declaration hashes, then validate the active registry/held identities. Canonical foreign WIP and frozen SDK sources must remain untouched. No model, teacher or paid probe is involved. Exact operation/result receipt is reported separately after execution.
