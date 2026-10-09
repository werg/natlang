# Legacy code removal (2026-10-09)

Owner asked whether legacy code goes too, after the Maple-Preview BF16 checkpoint was deleted. Rule applied: remove only code with no importer, no recipe/test/systemd/package.json reference, no use by an active or resumable run, and no owner decision keeping its line open. Git history is the archive (`git log --diff-filter=D -- PATH`, restore with `git checkout <commit>^ -- PATH`).

## Removed

| Path | Why | Evidence | Commit |
|---|---|---|---|
| Defaults to `/home/werg/data/models/maple-preview-bf16` (`maple/student.py` DEFAULT_MODEL, `maple/foundation_heads.py --model`, `scripts/maple_mixed_data.py --model`, usage docstrings in `maple/routing.py`, `scripts/maple_gguf_verify.py`, `scripts/maple_llamacpp_parity.py`) | the checkpoint was deleted (owner); a default would silently pick a missing Maple path | `--model` now required; `student.MAPLE_PREVIEW_MODEL` kept only as the identity of Maple's converted-weights cache; 50 tests pass | 6b676ad7 |
| `training/neuralese/natlang_neuralese/eval/sketch_cutoff_probe.py` (+ its test) | probe on base Maple weights, stopped, not used for decisions (DECISIONS 2026-10-08); sketch replaced by the input map | no importer outside its own test; Pop confirmed standalone; manifest hits are immutable snapshot copies; test_text_warmup 68 pass | 06fd2ae1 |
| Worktree `/home/werg/.claude/jobs/9c849f00/tmp/wt-art2` | finished artifact-registry agent | clean, HEAD 6e62100b on origin/main | (worktree only) |

## Kept (in use)

| Path | Why kept |
|---|---|
| `train/sketch_handoff.py`, `train/sketch_defaults.py`, `eval/latent_sketch.py` | imported by trainer cores (text_warmup.py:975, trajectories.py:738, eval/text_warmup_runtime.py:4) and declared in TRAINING_RECIPE.md:346/457/461 (Pop, 21:34). Remove together with a refactor of those callers by their owners (Pop / architecture session C1–C3); serving sketch initialisation remains a consumer component (DECISIONS 2026-10-09). |
| Legacy marker/RMS projector profiles (model/heads, fork `legacy-rms-v1`) | needed to load existing checkpoints and the fork's pinned GGUF projector (llama-cpp-fork.json read_adapter_cpu_verification); frozen artifacts must stay loadable |
| `maple/` package | shared Maple-family code used by Mellum (load_maple, DenseExperts, ternary, maple_port, routing, nested_train, family, qat_convert/qat_export) |
| Diagnostic CLIs without importers (`eval/bench.py`, `cache_diagnostics.py`, `ffn_dispatch.py`, `mtp_draft_probe.py`, `recurrence_runtime.py`, `residual_transition.py`, `scheduler_throughput.py`, `stopping_diagnostics.py`, `maple/gate_diagnostic.py`, `train/delta_e2e.py`) | 0–6 days old research entry points (`python -m`), several edited this week; not superseded by a decision |
| `train/delta_projection.py` | M5 D projection, task in progress |

## Remove after a run or refactor

| Item | Condition |
|---|---|
| Sketch-era recipes `gold-text-warmup-lfm-cutoff8-rollout4-to8-v1`, `local-stage-sketch-credit-v1`, `gold-text-exposure-curriculum-v1` | historical per Pop (no current lineage); recipes are consolidated by the architecture session (C3): drop them there with the resolver tests |
| Sketch modules above | after their callers in text_warmup/trajectories/text_warmup_runtime are refactored |

## Ask owner

Scripts with no code, unit or recent-run reference and no change since 2026-10-02 (36 of 116 unreferenced scripts). Their lines are still listed as open choices in plans/HANDOVER.md (e.g. "Train Sharp-Spark on v5 (only when the user says so)"), so they are not removed without a decision:

| Script | Line / purpose | Last change |
|---|---|---|
| `scripts/acquire_directory_sources.py` | crisp teacher/data pipeline or one-off tool | 2026-09-29 |
| `scripts/acquire_workflowevals.py` | crisp teacher/data pipeline or one-off tool | 2026-09-29 |
| `scripts/analyze_generation_throughput.py` | crisp teacher/data pipeline or one-off tool | 2026-09-21 |
| `scripts/audit_recovered_sources.py` | crisp teacher/data pipeline or one-off tool | 2026-09-28 |
| `scripts/audit_static_data.py` | crisp teacher/data pipeline or one-off tool | 2026-09-28 |
| `scripts/build_teacher_coverage_selection.py` | crisp teacher/data pipeline or one-off tool | 2026-09-22 |
| `scripts/clone_runtime_tree.py` | runtime-tree copies for remote installs | 2026-10-02 |
| `scripts/combine_sft.py` | crisp teacher/data pipeline or one-off tool | 2026-09-29 |
| `scripts/create_model_swap_config.py` | student/Bonsai sequential serving | 2026-09-23 |
| `scripts/interleave_teacher_queue.py` | crisp teacher/data pipeline or one-off tool | 2026-09-29 |
| `scripts/inventory_bonsai_attempt_identities.mjs` | Bonsai teacher attempts | 2026-10-01 |
| `scripts/ling_grouped_experts.py` | Ling backbone candidate (Mellum chosen) | 2026-09-27 |
| `scripts/materialize_studio_teacher.py` | crisp teacher/data pipeline or one-off tool | 2026-09-21 |
| `scripts/merge_lora_overlay.py` | crisp teacher/data pipeline or one-off tool | 2026-09-27 |
| `scripts/prepare_sharp_training_model.py` | Sharp-MiniCPM5 student candidate | 2026-10-02 |
| `scripts/project_reviewed_leaf_pass.py` | crisp teacher/data pipeline or one-off tool | 2026-09-21 |
| `scripts/prune_reviewed_leaf_references.py` | crisp teacher/data pipeline or one-off tool | 2026-09-21 |
| `scripts/recover_native_evaluation.py` | crisp teacher/data pipeline or one-off tool | 2026-09-30 |
| `scripts/refine_lfm_teacher.sh` | LFM teacher polish (crisp line) | 2026-09-21 |
| `scripts/refine_lfm_teacher_action.sh` | LFM teacher polish (crisp line) | 2026-09-22 |
| `scripts/remote_teacher_bootstrap.py` | remote teacher hosts (natlang-remote era) | 2026-10-02 |
| `scripts/roll_teacher_runtime.py` | remote/queue teacher supervisors | 2026-09-30 |
| `scripts/run_bulk_classifier.py` | crisp teacher/data pipeline or one-off tool | 2026-09-19 |
| `scripts/run_managed_improvement.py` | student/Bonsai sequential serving | 2026-09-27 |
| `scripts/select_sft.py` | crisp teacher/data pipeline or one-off tool | 2026-09-26 |
| `scripts/select_teacher_polish.py` | crisp teacher/data pipeline or one-off tool | 2026-09-21 |
| `scripts/select_teacher_programs.py` | crisp teacher/data pipeline or one-off tool | 2026-09-20 |
| `scripts/self_improvement_data.py` | crisp teacher/data pipeline or one-off tool | 2026-10-01 |
| `scripts/serve_minicpm.sh` | Sharp-MiniCPM5 student candidate | 2026-09-29 |
| `scripts/spark_training_model.py` | Sharp-Spark-X2.5-4B student candidate | 2026-09-27 |
| `scripts/start_reviewed_remote_successor.py` | remote teacher hosts (natlang-remote era) | 2026-10-02 |
| `scripts/teacher_redo_loop.sh` | two-teacher redo loop (Bonsai era) | 2026-09-27 |
| `scripts/train_and_publish_browser_docker.sh` | crisp LFM browser publication | 2026-09-22 |
| `scripts/verify_completed_training_pipeline.py` | crisp teacher/data pipeline or one-off tool | 2026-10-02 |
| `scripts/verify_source_audit_identities.py` | crisp teacher/data pipeline or one-off tool | 2026-09-30 |
| `scripts/watch_bonsai.sh` | Bonsai teacher container watchdog | 2026-10-02 |

The other 80 unreferenced scripts changed after 2026-10-02 (Neuralese era) and are kept as manual tools.

Other owner items:

- `natlang-development-cache-hygiene.timer` still runs `/home/werg/natlang-remote/reclaim_file_cache.py` from the retired natlang-remote snapshot (AGENTS.md: no new jobs from it). The memory guard now releases cache itself (`memory_ledger.py release-cache`, own unit since 82b5dbcd). Recommendation: disable the timer, or port the script into the repo if its targeted roots are still wanted.
- `/home/werg/data/models/maple-preview-converted` (20 GB): Maple's converted-weights cache, only useful if Maple work resumes (re-creatable from a re-downloaded checkpoint).
- 39 worktrees under `.claude/worktrees/agent-*` (8 locked) belong to other sessions' worktree agents; their sessions should remove finished ones (`git worktree remove`). Not touched.
