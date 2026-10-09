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

## Removed by owner decision (2026-10-10)

Owner: "You have my authorization to make the deletions of storage and old candidate code." The 36 unreferenced scripts unchanged since 2026-10-02 were removed in three commits: student candidates (Sharp-MiniCPM5, Spark-X2.5, Ling, LoRA overlay merge), Bonsai teacher and remote teacher hosts, pre-Neuralese crisp teacher/SFT polish and one-off source tooling. Instructional docs carry a removal note; dated historical plans are left as history. Data and storage were not part of this change.

- `scripts/acquire_directory_sources.py`
- `scripts/acquire_workflowevals.py`
- `scripts/analyze_generation_throughput.py`
- `scripts/audit_recovered_sources.py`
- `scripts/audit_static_data.py`
- `scripts/build_teacher_coverage_selection.py`
- `scripts/clone_runtime_tree.py`
- `scripts/combine_sft.py`
- `scripts/create_model_swap_config.py`
- `scripts/interleave_teacher_queue.py`
- `scripts/inventory_bonsai_attempt_identities.mjs`
- `scripts/ling_grouped_experts.py`
- `scripts/materialize_studio_teacher.py`
- `scripts/merge_lora_overlay.py`
- `scripts/prepare_sharp_training_model.py`
- `scripts/project_reviewed_leaf_pass.py`
- `scripts/prune_reviewed_leaf_references.py`
- `scripts/recover_native_evaluation.py`
- `scripts/refine_lfm_teacher.sh`
- `scripts/refine_lfm_teacher_action.sh`
- `scripts/remote_teacher_bootstrap.py`
- `scripts/roll_teacher_runtime.py`
- `scripts/run_bulk_classifier.py`
- `scripts/run_managed_improvement.py`
- `scripts/select_sft.py`
- `scripts/select_teacher_polish.py`
- `scripts/select_teacher_programs.py`
- `scripts/self_improvement_data.py`
- `scripts/serve_minicpm.sh`
- `scripts/spark_training_model.py`
- `scripts/start_reviewed_remote_successor.py`
- `scripts/teacher_redo_loop.sh`
- `scripts/train_and_publish_browser_docker.sh`
- `scripts/verify_completed_training_pipeline.py`
- `scripts/verify_source_audit_identities.py`
- `scripts/watch_bonsai.sh`

## Ask owner (remaining)

The other 80 unreferenced scripts changed after 2026-10-02 (Neuralese era) and are kept as manual tools.

Other owner items:

- `natlang-development-cache-hygiene.timer` still runs `/home/werg/natlang-remote/reclaim_file_cache.py` from the retired natlang-remote snapshot (AGENTS.md: no new jobs from it). The memory guard now releases cache itself (`memory_ledger.py release-cache`, own unit since 82b5dbcd). Recommendation: disable the timer, or port the script into the repo if its targeted roots are still wanted.
- `/home/werg/data/models/maple-preview-converted` (20 GB): Maple's converted-weights cache, only useful if Maple work resumes (re-creatable from a re-downloaded checkpoint).
- 39 worktrees under `.claude/worktrees/agent-*` (8 locked) belong to other sessions' worktree agents; their sessions should remove finished ones (`git worktree remove`). Not touched.
