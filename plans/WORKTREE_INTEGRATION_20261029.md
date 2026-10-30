# Worktree integration, 2026-10-29

Owner: no work lost or abandoned in worktrees (AGENTS "Work lands on main"). A `git cherry` audit found 69 worktrees
whose commits or tracked changes were not on main. Most came from agents that are no longer running. A Claude
Sonnet reviewer integrated 33 commits onto main, from `ad4a781a0` to `624b9a67c`: legacy removal, harness
ergonomics, rerender reconciliation, service oracles, recovery tests and others. Its full report is summarized in
the 2026-10-29 Claude session. Worktrees named `*1029*` are today's Codex/Terra work, and their owners merge them.

## Decisions on the remaining items (Claude supervisor)

| Item | Decision | Reason |
| --- | --- | --- |
| `fork-rl-actor` 32d8c48c9: privileged insertion counts imported states' tokens | **Real bug on main, being fixed** | The engine computes the guidance `insert_at` over plan tokens. The RL capture records consume markers as own tokens, and the learner (`privileged_context` reconstruction, `student_form`) counts them. So guidance after an imported state is refused. The fix is on main via a Sonnet helper; the old-API commit is not cherry-picked. |
| `terra-lfm3-service-oracles`: 9 oracle-hardening commits bound to `rl/lfm3_pipeline.py` | Not integrated as code | They bind to the llama.cpp actor-fleet pipeline, which owner decision `248d069a6` replaced with the vLLM fork. The six engine-independent oracle checks are on main. Re-expressing the adversarial RL-admission proof on main's pipeline is open work for whoever owns RL task admission. AGENTS already requires meaningful wrong-answer oracle tests. |
| `agent-ac2db8bd04f632f22` 3ee870e82: model-interaction recorder choke point | Discarded | It conflicts with main's `invocation_role` / `x_natlang_privileged_context` capture path. Its blanket refusal of generations outside an invocation breaks the current `native-state-learning-scope` behaviour. Main's capture path already records the invocation identity. |
| `wt-lfm3-topology-backup` (uncommitted `diloco_install_recovery.py`) | Discarded | Written for the resident four-worker layout. Main's recovery staging is direct-mode. |
| `wt-lfm3-recovery-health` 15197d967, e1bc64d57 | Discarded | Resident-plan attestation was superseded by the direct-recovery rewrite of `diloco_periodic_recovery.py`. |
| `wt-native-failed-history-structured-spans` | Discarded | A witness schema with no producer on main. Rebuild from the capture side if failed-history spans are needed. |
| `wt-lfm3-diloco-identity-transition-20261028` | Code not integrated; **open bug** | Its fix for rewriting a checkpoint on a no-op resume fails its own assertion on main (the hunk was reverted in `1c4a216e2`). The underlying no-op-resume rewrite remains to be diagnosed by whoever owns DiLoCo continuation. |
| `natlang-native-state-scope` f8a274721 | Discarded | Test-only change that conflicts with the evolved test file and adds no coverage. |
| `agent-a6c693c5fbffb4329` dff24f2a4 | Discarded | Marked WIP / do-not-merge. |
| `wt-d32-render-pending` | Discarded | Its modifications delete protected decision holds. |

## 2026-10-30 round (Claude helper, DGX)

Audit: 97 worktrees; 39 held commits that `git cherry` reported missing from main (about 209 patches), and 30 more held only uncommitted or untracked changes. No process had a cwd, command line or environment in any of them. No coord message in the last 24 h named one, and no tracked file had changed since 2026-10-29 20:39. All were orphaned. Each commit was checked by simulated cherry-pick (`git merge-tree`), by an identical subject on main, and by containment (the share of its added lines, over 30 chars, present on main). Uncommitted patches got the same containment check.
Removed 66 worktrees and deleted 52 branches. Symlinks inside them were unlinked first; one pointed at the live `training/decision-holds`.

| Item | Decision | Reason |
| --- | --- | --- |
| `terra-prompt-variants-1029` 2abf698d3, 5905c73bb, 60e49a65f, 98456cc12 | **Integrated** | This code produced the full-prompt variants in the fresh-restart source (`prompt-variants-pilot-v1`). Before this, it existed only in the worktree. On 60e49a65f, `retain_prompt_tools` was re-expressed alongside main's `native_xml` rendering. Tested: prompt-variant, text-corpus and chain tests (20 passed). |
| `terra-lfm3-end-to-end` f21a16caf, 393f94ca5 (`prebuilt_build.sh`) | **Integrated** | The GGUF prebuilt still uses this script. These commits require an exact llama.cpp commit and archive identity. Main's `neuralese` branch label was kept. |
| `terra-lfm3-end-to-end` (other 34) | **Kept the worktree, not integrated** | The llama.cpp actor fleet and its oracle hardening were superseded by the SGLang actor (1dbddd06f). About 18.8k recorded RL task-packet receipts (`rl-qualified-dependency-union-*`, `rl-skill-required-read-groups-*`, `rl-effect-*`) bind this worktree's runtime inventory. The live RL config does not reference it. Remove it once those packets are retired or requalified. |
| `chain-frozen`, `claude-recovery-20261028/data/worktree` (rerender-reconcile, uncommitted) | **Kept** | `first-chain-e799-v3` receipts and its code relocation bind files at these paths. Their content is otherwise 98% and 100% on main. |
| `terra-lfm3-service-oracles` | Removed | Its 9 commits are a subset of `terra-lfm3-end-to-end`, which is kept. The 2026-10-29 decision stands. |
| Standalone/DiLoCo set: `wt-f59-rotation-rejoin`, `wt-standalone-*`, `wt-lfm3-mmap-surface`, `wt-sequential-withdrawals`, `wt-lfm3-transient-download`, `wt-lfm3-diloco-stale-reopen`, `wt-multi-source-provenance`, `wt-muon-rounding-custody-1029`, `terra-fresh-first-outer`, `terra-resident-declared/map` | Removed, already on main | These were rebased onto main with the same subjects (for example f73ddcee3, 08bcbd51e, cbd168d39, 0d026d13c, fdeaf6364), with 95 to 100% containment. |
| `wt-metalearner-recovery`, `wt-decision-binding-migration`, `wt-portable-stage-argv`, `wt-long-matmul-int64`, `wt-cw-classification-append-proof`, `natlang-sol-memetic-learning`, `natlang-native-recorded-objective-backend`, `wt-native-failed-history-deployment`, `wt-native-granted-source-adapters`, `agent-a233ac05`, `agent-a5e066fc`, `agent-a8f0a72a`, `natlang-ergo` | Removed, already on main | Each has the same subject on main or its patch is already on main. In `agent-a5e066fc`, the uncommitted eager-QAT fallback was superseded by 100140e0c and bae117676. |
| Oct 16 frozen-runtime compositions: `wt-cw-lfm-raw`, `wt-cw-mellum-raw` (an unfinished cherry-pick), `wt-cw-mellum-attention-index`, `wt-root-marker-b435` | Discarded | These composed main features onto retired cf7/Mellum cloud runtimes. 81 to 100% of their lines are on main, and the remainder is frozen-runtime glue. |
| `natlang-teacher-drain-export`, `natlang-teacher-zero-guided` (Oct 22) | Discarded | Operator patches for Qwen3.6 teacher generation, which LFM3 stopped (owner 2026-10-28: no additional teacher generation). Re-derive from git reflog if that generation resumes. |
| `fork-rl-actor` 32d8c48c9, e965cd70d | Discarded | The llama.cpp parity check was superseded by the SGLang parity checks. Main fixed the guidance-insertion bug differently (`_insert_privileged` `insertion_point`). |
| `wt-lfm3-recovery-health`, `agent-ac2db8bd`, `agent-a6c693c5`, `natlang-native-state-scope`, `wt-lfm3-topology-backup`, `wt-native-failed-history-structured-spans`, `wt-d32-render-pending`, `wt-lfm3-diloco-identity-transition` | Discarded | Per the 2026-10-29 decisions above. The identity-transition bug stays open. The d32 hold copies were byte-identical to main. |
| `agent-a97139d3` (uncommitted real NVFP4 GEMM for inline QAT at lambda=1) | **Preserved on branch `wip/nvfp4-qat-gemm-20261028` (pushed); needs a decision** | A potentially valuable throughput change on the default QAT path, but never tested. It needs Blackwell parity qualification before it can land on main by default. |
| Oct 27 fix worktrees (`wt-native-cpu-attention-backward`, `-cpu-graph-parity`, `-sft-packed-capacity`, `-new-source-priority`, `-release-compaction`, `-source-reuse`, `full-check/*`, `wt-tool-args-double-decode`, `wt-closure-receipt-force`), `agent-adcc9b9` (rl-reply-guard), `agent-a795699805` (guidance-writer-ts), `agent-ae5b93332` | Removed, already on main | 95 to 100% containment. `cpu-graph-parity`'s rowwise CPU path is on main in evolved form (`cpu_rowwise`). Its one-off diagnose script and the `.audit` snapshots for retired lines were dropped. |
| `wt-main-check-*`, `four-recovery-validation-*`, `check-fresh-restart-20261029` | Discarded | Validation checkouts carrying uncommitted vLLM-fork wheel files, superseded by the SGLang decision (1dbddd06f). |
| `natlang-native-graph-data`, `wt-cw-checkpoint-file-offload`, `wt-native-mellum-prefix-adjoint`, `lfm3-sft-launch`, `.claude/jobs/.../chk-wt` | Discarded | Oct 15–18 legacy docs and scripts, and the recipe of the rolled-back LFM3 lineage. `chk-wt` held only empty node_modules. |

The 30 clean worktrees with no unmerged commits and no changes were left in place. Some are frozen runtimes of live or recorded runs.
