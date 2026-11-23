# Registry publication from pinned producers

A pinned producer imported `register_corpora.py` from its immutable checkout. Its module-level `REPO` named that checkout, so registration's Git publisher/sync-only advanced the detached pin to origin/main before publishing. The marker still named the original code commit, invalidating later provenance checks. Two publications succeeded before the next26 were blocked. Publication and immutable reads require separate roots.

Writer defaults now use `data_registry.registry_owner_repo()`; explicit writer destinations pass `assert_registry_writable()` before mutation. The shared API follows pin source/Git common ownership and rejects marker-containing ancestors, including resolved symlink destinations. `NATLANG_REPO`, `code_repo()`, default corpus/data registry reads, explicit snapshot reads and producer commit identity remain pinned.

## Writer mechanism inventory

| Surface | Action |
|---|---|
| scripts/neuralese_data/register_corpora.py (`register`, `attempt`, `link`, `registered_ids`) | Mutable owner default; guard before fetch, link creation, publisher/sync and path writes. All imported producer callers inherit this route. |
| scripts/data_registry.py (`promote`) | Promotion reads the mutable registry explicitly to preserve current curated metadata, invokes owner's publication module; ordinary read commands retain pinned selection. |
| scripts/neuralese_data/{render_parts,static_curriculum_campaign,clef_pools_daily,student_rerender,register_swe_bulk,target_set_watcher}.py | Call shared `register`/`link`; no separate Git publisher. Existing code/provenance reads remain local to pinned source. |
| scripts/neuralese_data/register_native_turn_render.py | Both direct and publisher routes covered; entry path expressed relative to owner; source-admission reads still use explicit pinned ROOT. |
| scripts/rl_tasks/register_{authored,source}_held.py | Packet, manifest and registry owner defaults; guard all output destinations before packet writes. |
| scripts/sync_training_corpora.py | Publish-only CLI default uses owner; explicit/read repo stays explicit. All atomic saves guard destinations. Verification's runtime receipts use canonical mutable checkout, while verified manifest/file identities remain the selected snapshot. |
| scripts/source_review.py | Manifest/decision/receipt writes guarded; writer CLI defaults owner. Verify/show and explicit read paths retain their selected checkout. |
| natlang_neuralese/artifacts.py and scripts/neuralese_artifacts.py | Artifact registration/manifest/store writes guarded, default output owner; readers retain prior snapshot root. Trainer commit defaults to executing code checkout, not mutable publication checkout. |
| common/corpus_registry.py (`write_entries`, `save`) | Shared lower-layer guards, owned by sibling strand. Default load/signature remain pinned. |
| scripts/pin_checkout.py | A new child pin preserves exact requested source commit while anchoring marker.source, shared data/env links and ignored dependency inputs directly to canonical primary checkout, rather than a retireable parent pin. |

## Read-layer exceptions retained deliberately

`common/data_registry.py:code_repo`, `load` and manifest selection; `common/corpus_registry.py:load`, `signature`, `load_ref`; render_native_turn_corpus.py's explicit ROOT admission reads; source_case_closure.py's explicit ROOT registry lookup; target_set_manifest.py's explicit origin/main read; student_rerender.py's pinned registry snapshot; assemble_neuralese_cohorts.py's manifest reads. These are consumers, not Git-layer writers. No default reader was globally switched to live owner. Current-state/dashboard explicit REPO reads are not publication paths; live-consumer selection changes are independently inventoried by the controller strand.

## Verification

Targeted regressions exercise isolated Git publication from a pinned producer without changing pin HEAD or its readable registry, explicit pin rejection before Git/link/data mutation, symlink/ancestor guards in the shared API, owner-relative render registration and real pin-of-pin creation followed by retirement of the first pin. Test results are recorded with the source commit when actually run; this document alone is not a pass claim.

Actual targeted run `registry-pin-writers-20261123a`:111passed in9.99s,385.5MiB peak, cap0.8GiB/reserve7.5GiB. Files: register_corpora, sync_training_corpora, source_review, registry_producer_owner, pin_checkout_shared, union_registration_protection, native_render_storage_binding. No production registry or immutable pin was used as a test write destination.
