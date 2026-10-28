# Legacy removal inventory (2026-10-28)

Owner rule: AGENTS.md "Lean codebase and economical testing". The core direction is the LFM3 joint SFT line, JOLT+
whole-graph RL, the tuning system, the metalearner, and the natlang runtime, harness and serving. Cross-model work
was struck on 2026-10-28. Git history is the archive.

Method: a token reference index over tracked files (code, tests, registries, docs; `plans/`, corpus manifests and
closure receipts counted separately, because they cite frozen custody copies rather than live paths), the recipe
loader's `HANDLERS`, imports, and live processes and systemd units checked read-only on 2026-10-28. Live services
import `scripts/coreweave/{private_export_guard,recovery_inputs,source_archive,cwctl}` and
`natlang_neuralese.common.hashing`. Nothing removed here is among those imports.

## Removed on branch `legacy-removal-20261028`

| Family | What | Why legacy | Callers handled |
|---|---|---|---|
| Old-paradigm recipes | 65 `neuralese-direct-*`, `neuralese-e2e-*`, `neuralese-mixed-*`, `diag-*` and `luna-direct-*` recipes, `recipes/HISTORY.md`, `scripts/prune_stale_recipes.py`, `tests/neuralese/legacy_recipes.py`, `eval/plain_weight_drift.py` | NeuraleseProjection, TokenChoiceProjection, input map and warm-up stages. `train/recipe.py` only knows native kinds, so these recipes could no longer load. | README rewritten; probe example made generic |
| Cross-model and old-paradigm train leaves | `shared_space.py`, `train/exposure_curriculum.py`, `train/exposure_data.py`, `train/trajectory_probe.py`, `train/foundation_task_eval_recipe.py` with their tests | Large/small shared space (MAPLE_QWEN_JOINT), greedy semantic-span exposure, held reader / write-donor selection, a one-off foundation recipe preparation | No importers besides their own tests |
| OpenCode collector | 16 scripts, `OPENCODE_LOOPBACK_BOOTSTRAP.md`, 10 node/python tests, 3 fixtures | Collected teacher trajectories from external providers; generation is retired | Duplicate-helper baseline shrunk |
| Generation orchestration | Zen/OpenRouter/Gemini/Luna starters, slots and successors; Pop replenish and campaign-resume systemd units; generation monitor and backlog; Qwen3.6 shared dispatcher; free-model watcher; remote-teacher sync; DGX v9 pool approval, handoff and relocation; one-off Luna audits; Sol semantic-stack generation priority queue | No teacher generation for LFM3 | Generation-authority test kept for its authority cases; ratchet baselines shrunk |
| Projected-value one-offs | Counterfactual reader proposal, value-visibility audit, projection references, eval reader ordering, gold-text packet and preview, v14/v15 authored successor world builders and tests, four preview materializers | Projected-value channel and warm-up line, and finished successor campaigns | None |
| Decision-pool and Pop one-offs | Visible-decision pilot, variants and toxicity pool; decision-children materializer with its `scripts/lib` modules and ad hoc tests; Step 5 packet validator; dev-sync partials and verifier; Pop container Python wrapper | Gemini decision pools, the OpenCode Step 5 collector, and the closed Pop machine | None |
| Cross-model role exchange (coordinator request) | `rl/native_role_release.py` (2,133 lines), `rl/local_cycle_semantic_return.py`, `scripts/coreweave/{native_role_exchange,native_role_exchange_io,native_shared_participant_export,prepare_native_scheduler_adoption,prepare_local_cycle_shared_semantic,compose_native_actor_runtime,verify_native_actor_runtime,prepare_native_actor_role,install_native_actor_layout}.py`, `native_deployment.prepare_role_scheduler`, `validate_joint_task_specs` and the `role-scheduler` CLI, and their tests | Source-host role ownership, release and exchange between the LFM and Mellum owners, feeding the shared VAE consumer | See the next section |

### Single-model behaviour kept in the role-exchange migration

Every role-exchange function short-circuits when a plan has no `role_exchange`. The callers keep exactly that path:

- `rl/native_raw_rounds.py`: the new `owned_models(plan)` returns every declared student. It rejects `role_exchange`,
  `owned_models`, `local_cycle_source`, `consumer_preparation_handoff` and `snapshot_preparation`.
  `validate_release` and `run` lose the exchange branches. The `on-shared-request` foundation policy now raises a
  configuration error, because it needed the exchange.
- `rl/local_cycles.py`: single-device RAW RL plus supervised continuations are unchanged. Plans that declare
  `local_cycle_role_plan` or `semantic_return` are rejected. The role participant binding, the
  foundation-request forwarding, the semantic return adoption and the handoff yield are removed.
- `train/native_readiness.py`: `ready_participants` keeps the per-participant and raw-release paths.
  `snapshot_successor` raises on a declaration. The preparation-cancellation and seal calls were no-ops without an
  exchange, so they are removed.
- `train/native_semantic_stage.py`: source-host role snapshots and snapshot adoption raise.
- `scripts/coreweave/provider_dependencies.py`: `snapshot_adoption` raises.

## Deferred: inside an active area (inventory only)

| Item | Callers | Decision |
|---|---|---|
| Cross-model semantic VAE family: `model/semantic_vae.py`, `semantic_channel.py`, `semantic_message.py`, `train/native_semantic*.py`, `eval/native_semantic_runtime.py`, `rl/native_semantic_{replay,route}.py`, the `train/native_readiness.py` shared-stage controller loop (it launches only when two or more participants are ready), recipe `HANDLERS` `native_shared_stage_controller` and `native_semantic_training`, recipe `native-state-semantic-curriculum-rest27-foundation-20261015-v1.json`, `scripts/neuralese_data/{qualify_native_joint_cuda,assemble_native_semantic_pair_union}.py`, about 15 `tests/neuralese/*semantic*` files. About 5.9k package lines. | `rl/native_rounds.py` refresh_job, `rl/production.py`, `serve/engine.py`, `serve/http.py`, `serve/native_guidance.py`, `train/recipe.py`, `train/native_capacity_continuation.py`, `train/native_state_sft.py` (`foundation_qualification_request`, `on-shared-request` policy), `rl/local_cycles.py` (`semantic_adapter`, `bind_state_semantic_route`, `publish_semantic_supervised`), and the converters' `semantic-pairs` sources | Remove after the RL, serving, LFM3-port and defaults-audit work lands. Next family to take. |
| `common/memetic_teacher_ferry.py` and `scripts/coreweave/memetic_teacher_ferry.py` (Qwen teacher relay for guidance authoring) | `rl/memetic_program.py`, `rl/memetic_task_runner.py` | Teacher generation is retired. Move guidance authoring to the LFM3 policy (JOLT+ privileged teacher), then remove. RL area. |
| `model/causal_feedback.py` (raw next-token embedding bootstrap from the old foundation) and its 2 tests | Tests only | Remove after the LFM3 port lands (model area) |
| `train/quantization.py` `STAGE_KINDS` (`core_text_warmup`, `raw_recurrence_training`, `neuralese_direct`) and `maple/family.py` `neuralese_direct` check | QAT component | Remove when QAT is wired to the LFM3 joint stage (stable-softmax and QAT area) |
| LFM2.5-only paths: `model/lfm2_port.py`, LFM2 branches in `isolated_sequence`, `native_decode_batch`, `serve/batch.py`, `data/render.py`, `lfm25` keys in scripts (see `plans/neuralese/LFM3_PORT_INVENTORY_20261028.md`) | Port in progress; the LFM2.5 step-41022 recovery checkpoint reader | Decide after the port. Keep the minimal reader only if a current consumer needs step 41022. |
| `scripts/build_neuralese_gold_text_delta.py`, `assemble_admitted_neuralese_cohort.py`, `audit_neuralese_recurrence.py`, `audit_observed_neuralese_outputs.py`, `assemble_neuralese_recurrence.py` | `scripts/neuralese_data/render_native_turn_corpus.py`, `build_native_turn_union.py` (live render units) | Data-conversion area. Fold the reader helpers into `neuralese_data`, then remove. |
| Whole-trajectory rendering (`--renderings ... whole-trajectory`, `whole_trajectory` holds in `build_native_turn_union`) | Live LFM3 render units | The objective is retired (own-output supervision). The rendering view needs a decision from the data-conversion owner. |
| `run_reviewed_luna_dispatcher.py`, `generation_authority.py`, `start_reviewed_generation_successor.py`, `luna_source_inventory.py` | `scripts/neuralese_data/{provider_campaign_source,snapshot_provider_campaign,verify_tune_pool_import}.py` import plan and ledger verifiers | Move the verifiers into the data readers, then remove the dispatcher |
| `replenish_luna_held_shapes.py` and `natlang-pop-luna-replenish.{service,timer}` | `scripts/neuralese_data/check_held_world_collisions.py` imports `world` | Move `world` into neuralese_data, then remove |
| `scripts/coreweave/teacher_*` (internal conversion, admission, catalog, helper continuation) | LFM3 data conversion, admitted-teacher export service | Readers stay. Remove generation-only parts with the data owner. |

## Ask the owner

### The Mellum line: retired (decided 2026-10-28, `plans/neuralese/LFM3_COMMITTED_DECISIONS_20261028.md` item 16)

Removed in this branch:

- the 3 Mellum recipes and the Mellum preference fixture;
- the LFM/Mellum line-parity verifier (`train/line_parity.py`, `line-parity.json`, its test, its check-main entry
  and the lineage table in `current_state.py`);
- `scripts/maple_*` (7) and `scripts/mellum_*` (2);
- the Maple MoE and Mellum QAT benches, `mellum_parity.py` and `bench_port_training_step.py`;
- the held-v3 Mellum CoreWeave job;
- `maple/{family,export_member,ternary_pack,routing}.py` with their tests.

The step-39257 export and all data stay.

Still to remove (deep Mellum, after the LFM3 port and QAT work land):

- Model dispatch: `model/native_model.py` and `native_backbone_restore.py` dispatch to `maple/student.load_student`
  and `maple/maple_port`.
- Maple-specific modules: `maple/{student,nested_train,maple_port,distill_data,generation_gate,qat_export}.py`,
  `train/joint.py`, `train/joint_kd.py` and the Mellum branch of `maple/qat_convert.py`. `train/quantization.py`
  imports `student`, `generation_gate` and `distill_data`, and `train/drift_monitor.py` imports
  `generation_gate` and `distill_data`.
- Mellum entries in `common/line_backbones.py` and the `mellum` keys in data scripts.
- `scripts/neuralese_data/{curate_mellum_recovery_proxy.py,text_tokenizer_twin_mellum4962.*}`. These are in the
  data-conversion area, and the `.natrecords` file is registered data.
- `tests/neuralese/{test_maple_model,test_mellum_port,test_mellum_convert_v3,test_maple_ternary}.py` and
  `maple_reference/`.

Shared code that LFM paths import stays wherever it currently lives: `maple/model.py` primitives,
`fused_moe`, `fp8_experts`, `expert_gemm`, `moe_glue`, the norm/rope/state-gather/precision kernels and `ternary`.
Moving it out of `maple/` into `train/` or `model/` is a rename to do with the port.

### Other questions

- The pre-Neuralese LoRA student pipeline: `scripts/train_lora.py`, `create_training_pipeline.py`,
  `run_training_pipeline.py`, `train_dpo.py`, `periodic_heldout.py`, `posttraining_phase.py`, the
  `student_improvement` scripts and `to_gguf.sh`, about 6k lines with tests. `ts-host` playground and browser-model
  training still invoke `train_and_publish_browser.sh` and `to_gguf.sh`. Is in-browser playground training still a
  product feature? If not, the whole pipeline goes.
- AGENTS.md "Neuralese training foundation" still states the old warm-up, identity and embedding-distillation
  rules and the `neuralese_direct` exception, which mention NeuraleseProjection and TokenChoiceProjection. These
  are owner rules, so this pass leaves them unedited. The proposal is to replace them with a pointer to the LFM3
  recipe.
- `training/neuralese/recipes/luna-neuralese-direct-lfm-base-20261011-v8.json` is kept because it is registered as
  held corpus `luna-neuralese-direct-lfm-base-v8-recipe-source-held-20261011-v1` (recipe identity, no admission).
  Should that registration, and the file with it, be retired?
- `plans/` holds many handover, decomposition and status documents of discarded work, such as MAPLE_QWEN_JOINT,
  NEURALESE_DIRECT_V1, E2E_*, LUNA_*, GEMINI_* and OPENCODE_GENERATION. A separate plans sweep should cut them,
  updating links that `scripts/check_spec_links.py` checks.
