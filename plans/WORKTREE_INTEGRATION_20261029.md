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
