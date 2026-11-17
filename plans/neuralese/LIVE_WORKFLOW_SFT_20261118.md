# Observed live workflow data

The source is the actual Natlang executor inside a test-owned Vibestudio workflow. Workspace and CDP effects are
verified in that world; they are not described as portable program replay. The previous isolated helper pilot is
retired. This adapter exports those observed calls through the existing SFT renderer, selection, masks and chain
format. A portable ProgramRecord may accompany tested code, but is not invented as a prerequisite for live data.

## Mechanism inventory

- Base `packages/natlang/src/driver.ts` snapshots each actual native request before dispatch and retains the normalized
  reply plus original Pi context/assistant, resolved model, gateway turn and attribution. `model_turns` on the original
  `natlang.calls/1` record includes generations and scores. The authoritative raw call summary supplies owner and eval
  run identity. Score results are evidence only, never supervised completions.
- `scripts/neuralese_data/live_teacher_sources.py` is a source adapter: verify immutable task, raw call, validator and
  effect bindings; preserve exact captured messages/tools and original assistant tool IDs/arguments. Its metadata task
  is explicitly not a `natlang.program/2` record. The outer workflow run and each actual EvalDO run remain distinct.
- `sft_sources.extract`, `verified_episodes`, `core_candidates` and `build` handle both sources. Observed-live admission
  reopens the bound source artifacts before build. The shared `held_excluded_ids`, caps, near-duplicate filter,
  `call_row`, `TemplateRenderer`, target census and `exp0_sft.chains_of` remain authoritative. Semantic rows retain
  world, source manifest, gateway evidence and observed-live verification mode. No probabilities or sampled IDs are
  invented; the learner template is a declared rendering of captured semantic contexts, not the teacher's tokenization.
- Portable collection/replay remains `task_rollouts.prepare/score` and `sft_sources.replay_episode`; the latter returns
  `exact:false` for observed-live episodes before launching any runtime. Live worlds need test-owned live reruns.
- `exp0_sft.py:482` consumes chains identically for either source. No second trainer, objective or weighting exists.
- Engine request/response protocols are unchanged: this is client-side durable recording and offline source admission.
  All engines continue through `TemplateRenderer.render_request` (`sft_sources.py:305`) and shared target rendering
  (`sft_sources.py:328`). The serving parity inventory remains `docs/ENGINE_PARITY.md`; no new engine feature is needed.
- The specialized Neuralese export commands still require their existing real state evidence and replay surfaces
  (`sft_sources.py:1320`); this text-only live route makes no native-state/whole-graph training claim.

## Closed input contract

`natlang.live-teacher-task/1` contains `id`, `family`, `split:"train"`, `source_namespace`, nonempty `source_ids`,
`source_groups`, `source_revisions`, clean `code:{git_commit,dirty:false}`, `learner_kinds:["natlang-executor"]`,
`world:{instance,run_id}`, `required_effect_kinds:["workspace","cdp"]` (only effects actually required), and bound
`calls:[{path,sha256}]` and `verification:{path,sha256}`. Paths resolve relative to this manifest. Each call file is the
unmodified raw detail `{callId,summary,record,...}` exported by natlang.trace. All model-executed calls in the selected
validator scope must be present; call records outside that scope are refused.

The bound `natlang.live-task-verification/1` receipt contains matching `task_id`, outer `run_id`, and the complete manifest `world`, `status:"passed"`,
nonempty `checks` whose values are true, a bound `validator:{path,sha256}` (scoped validator source/evidence),
`calls:[{call_id,owner_kind,owner_key,run_id}]` with actual authoritative EvalDO identities, and bound
`evidence:[{kind:"workspace"|"cdp",path,sha256}]`. Evidence must include every required effect kind: actual before/after
readbacks and typed test assertions belong here. Validation conversations/results are evidence, never executor targets.
Gateway attribution identifies transport callers; it does not encode the EvalDO run and must not be substituted for it.

These receipts are trusted test-runner attestations checked for integrity and attribution, not cryptographic proof
against a malicious producer. A passing model opinion or a self-reported effect alone is insufficient. Dataset
registration, staging and learner exposure remain separate operations under the existing corpus protocol.

## Use

From the pinned checkout, extract repeatable manifests, then use the standard build (no replay command is necessary):

```
python scripts/neuralese_data/sft_sources.py extract --work WORK --live-run CLOSED/manifest.json
python scripts/neuralese_data/sft_sources.py build --work WORK --out BUILD --tokenizer TOKENIZER \
  --held-manifest HELD/manifest.json --base-rates '' --no-default-teacher-pool
```

`--no-default-teacher-pool` only removes the historical default pool coverage requirement. Explicit `--teacher-pool`
values continue to be checked. Existing held-v3/v2 source/group exclusion is always applied; run authoritative
`held_eval verify` with the new live source inventory before relying on a new corpus. The result is the same
`BUILD/chains.pkl` input to `exp0_sft.py train --chains`, with `build.json` and `rows.jsonl.zst` proving actual admission,
rendered mass and provenance. No training launch follows merely from constructing these files.
