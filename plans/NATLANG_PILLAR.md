# NatLang pillar: weekly slot (recurring)

Owner decision 2026-10-12 (plans/REVIEW_20261012.md, recommendation 10): the NatLang pillar (language, runtime,
specializer, applications) had been idle since 2026-10-10 while training took every resource. It gets a fixed weekly
slot that a training run's critical path does not crowd out. The DGX agent owns the slot; Pop joins for items it owns.

## The slot

Every week, starting the week of 2026-10-12. Each slot does all four items; what is not finished carries over.

1. **Specializer on real applications.** One specializer pass on recorded calls of real apps (nldb and pi first,
   then the rebuilt apps as their call stores fill): `natlang specialize --program applications/<app>`, first
   `--dry-run`, under the memory ledger. Record accepted and declined groups in `plans/TRACE_SPECIALIZATION.md`.
2. **Tier 4 / tiered engine.** One concrete step of `plans/TIERED_ENGINE.md` (tier numbering, NL promotion policy in
   shadow), measured against the crisp policy.
3. **Measurement debt.** Measure unmeasured rows of `plans/MODEL_FACING_CHANGES.md` in the teacher window (below),
   48 samples per variant, oldest and most widely seen first. Update each row's Status and Measurement.
4. **Owner-review queue.** Review new entries of `plans/OWNER_REVIEW.md` against main (the owner delegated acceptance
   on 2026-10-12) and open follow-ups for what is broken.

## Teacher window

The live executor/teacher (`natlang-qwen36-teacher`, Qwen3.6-35B-A3B NVFP4, about 68 GB while serving) does not fit
beside Mellum direct training (about 93 GB peak). The window is therefore a planned pause, not a wait for idle memory:

- once per week, at a Mellum checkpoint (the run is resumable), the DGX training agent stops the training unit,
  `scripts/ensure_teacher.sh` brings the teacher up, and `runs/research-teacher-20261006/teacher-window.json` is
  written with `start`/`end` (3 hours);
- the slot's live work runs in priority order (below), at most 48 samples per variant per row;
- the teacher stops, the file is removed and training resumes from its checkpoint on the newest code.

A Mellum phase handoff (direct_sft to direct_trajectory) is also a window if it comes first.

First window: the first Mellum checkpoint after 2026-10-13 06:00 UTC (DGX training agent schedules it).

### Window order

0. **Mellum base task-success reference (first, before the teacher starts; it gates the Mellum task bar).** The
   untrained Mellum base (raw-token heads, native full rendering, GPU) on the 16-task cadence subset of
   `natlang-held-task-packet-v3-20261012`, with two runs (`noise_repeats` 2), under the in-run eval's budgets (8 workers, 7200 s and 128 requests per task;
   DECISIONS.md "Task-eval budgets"). Each run takes up to about 1.5 h (the slowest task). About 34 GB, so it runs while training is
   paused and before `scripts/ensure_teacher.sh`. Until it lands, Mellum's task gate fails with "base
   reference_successes unmeasured" (plans/neuralese/DECISIONS.md, "Held task packet v2"). Command:

   ```sh
   cd /home/werg/natlang && python3 scripts/memory_ledger.py run --unit heldv3-base-mellum --budget-gb 34 --reserve-gb 2 \
     --priority high --no-resumable --workdir /home/werg/natlang -- sh -c 'HF_HUB_OFFLINE=1 NATLANG_REPO=/home/werg/natlang \
     PYTHONPATH=/home/werg/natlang/training/neuralese .venv-neuralese/bin/python -m natlang_neuralese.eval.task_cadence \
     --heads runs/neuralese-direct-mellum-20261011/heads-raw-token.pt --device cuda --prompt-rendering full \
     --out runs/heldv3-base-mellum-full --step 0 --spec "{\"packet\":{\"path\":\"runs/natlang-held-task-packet-v3-20261012/cases.jsonl\",\"sha256\":\"535269866aeecc107e3cc8b165790a5b200cdb104884d4994958eee17eb6be95\"},\"cases\":[0,5,10,15,16,21,26,31,32,37,42,47,48,53,58,63],\"workers\":8,\"case_seconds\":7200,\"max_model_requests\":128,\"max_turns\":160}" \
     > runs/heldv3-base-mellum-full.log 2>&1'
   ```

   The result is `runs/heldv3-base-mellum-full/task-eval.jsonl`: `task_successes` is the first run, and
   `repeat_successes`/`repeat_noise` come from the second. Set `task_eval.reference_successes` to the lower count in
   the Mellum direct recipe (`neuralese-direct-mellum-v4.json`, overrides.stage_parameter_defaults.neuralese_direct).
   This is a declared backbone-inherent difference in line-parity.json. The new reference reaches the live run at
   its next handoff. Wall time is about 16 tasks x up to 600 s across 4 workers, so under 45 minutes for both runs.
1. Specializer dry run with the real executor on nldb and pi (`--profile dgx-qwen`), then without `--dry-run` if
   the dry run accepts groups. (The 2026-10-12 plumbing run used a small model, see "Log".)
2. Measurement batches with an existing harness: evidence (`ts-host/test/evidence-atlas.test.mjs` live), publisher
   (`ts-host/test/publisher.test.mjs` live), compilers (`bench --checked`, old driver at the parent commit against
   the new), fusion planner (`ts-host/scripts/fusion-bench.mjs`), tool-output page size (static families with long
   reads).
3. The remaining rows through the A/B replay runner (to build before the first window: old against new prompt
   bytes on recorded inputs from the call store, `traces export` as the input source, one executor, 48 samples per
   variant, result row appended to MODEL_FACING_CHANGES).

### Measurement batches (66 unmeasured rows on 2026-10-12)

| Batch | Rows | Harness |
| --- | --- | --- |
| Shared runtime text (errors, refinement feedback, shared prompt, `nl.with`, combinators, `view`, `read_code`, scope rendering, page size, choice prompt, undeclared-type feedback, file tools) | 12 | A/B replay on recorded S1/static calls |
| Rebuilt apps (logs, wiki, games, scheduling, workflow, build, migration: instructions, `Untrusted<T>`, `Is<>` types, mode vocabulary, order, `stop`) | 8 | A/B replay per stage on recorded inputs |
| Evidence and publisher | 2 | existing tests, live |
| Helpdesk, terminal, media, notebook, ide, `readNote` | 14 | A/B replay per stage |
| Compilers and nldb (pass manager, opt guards, front-end chain, `database.nl` step 1) | 4 | compilers bench, nldb scenarios |
| Program improver (split stages, feedback, policies, `editComponents`, counterexamples) | 11 | A/B replay on improver runs |
| Specializer, fusion planner, promotion | 6 | shadow against crisp, fusion-bench |
| Advisory programs (failure explainer, curriculum advisor, source review) | 3 | labelled samples (plans/FAILURE_EXPLANATION_PROGRAM.md §6, SOURCE_REVIEW_PROGRAM.md §7) |
| Training renderings (program header, port-record calls, LFM2 rows, item options, soft bodies, toxicity) | 6 | the student's task-success eval, not the teacher |

## Log

- 2026-10-12: slot created. Specializer dry run restarted under the ledger (unit `natlang-specializer-dryrun`,
  12 GB, `runs/specializer-dryrun-20261012/`): within 12 GB beside Mellum only a small executor fits
  (Qwen3.5-0.8B on :8084, profile `dgx-small`), so this run checks the plumbing (per-group writers, the
  `contextWindow` fix c5fc0302) and does not judge case quality; the real dry run is window item 1.
- 2026-10-12/13: plumbing dry run completed (attempt 10, `runs/specializer-dryrun-20261012/`, unit
  `natlang-specializer-dryrun`, about 16 GB with the executor). Earlier attempts failed in the run script (vLLM
  entrypoint, HF blob links into the archive disk, page cache, KV share, memory-profiling race; one preemption by a
  high-priority E2E unit). Findings:
  - nldb `translate` (named explicitly: its stored decline makes the target policy skip it silently): the per-group
    writer path ran end to end; 0 of 1 groups got a case because `chooseCondition` exhausted its action budget on the
    0.8B executor (expected for a model that small; not a plumbing fault).
  - pi `generation`: the writer request failed with HTTP 400, prompt at least 25,601 tokens plus 7,168 requested
    output tokens = 32,769 against the server's 32,768 window. The `contextWindow` path (c5fc0302) reaches the
    request, but the output-token sizing is one token over the window (prompt estimate or off-by-one). **Bug to fix
    before the teacher-window run**; ts-host model request sizing (`openai-compatible.ts` `serverContextWindow`
    consumer). The other pi definitions have no calls from a model that spent tokens.
  - Fixed (2026-10-13): it was the prompt estimate, not an off-by-one. The agent sizes a turn from a characters-per-
    token estimate (1/3.5 until the first reply calibrates it) against a budget of the window minus an eighth; the
    first writer request's real prompt (25,601) ran well above its estimate, and the overflow recovery only parsed
    llama.cpp's `exceed_context_size_error`, so vLLM's 400 was thrown. `contextOverflow()` in
    `ts-host/src/native/agent.ts` now reads llama.cpp and vLLM refusals (prompt size and window); the turn is retried
    with the calibrated estimate, and once a window is stated every request's `max_tokens` is at most the window
    minus the calibrated prompt estimate minus 64. Tests: `ts-host/test/compaction.test.mjs` (vLLM refusal then a
    fitting retry; parser cases).
- 2026-10-13 (DGX ops): window item 0, Mellum base reference (`heldv3-base-mellum`, greedy, packet v3 16-task subset, full rendering): run 1 = 11/16 (task_success 11, invalid_action 2, semantic_mismatch 3). Run 2 (repeat1) was stopped at 09:30Z with 7 solved, 2 invalid_action, 2 semantic_mismatch and 5 unresolved, so no `task-eval.jsonl` row was written; both journals are kept in `runs/heldv3-base-mellum-full/task-eval/step-0000000{,-repeat1}`. `reference_successes` = 11 in `neuralese-direct-mellum-v4.json` is PROVISIONAL: it is run 1 alone (the rule is the lower of two runs), and measured greedy while the in-run eval now samples at temperature 0.6 (repetition dragnet). Follow-up: measure the Mellum base at temperature 0.6, two runs, to replace it (CoreWeave if available, else the next GPU gap that does not slow v4, else ask Pop).
- 2026-10-13 window item 1 (DGX ops), specializer dry run, `--profile dgx-qwen`, teacher :8082 saturated by the other window batches (60 running requests); output `/mnt/external/natlang-window-20261013/specialize-dry/`. nldb `translate`: 0 of 1 groups would get a case (g1, answer-only, 12 calls: "budget: natlang task timed out"). nldb (all definitions): ran 09:09Z to 11:09Z-CEST+0 (2 h) and exited 0 with an empty log, so no result. pi: ran until the teacher stopped at 09:30Z; the log lists "no usable calls" for most definitions after the teacher went away and "summarize declined (16 recorded calls, 20 needed)". Verdict: no group was accepted; the dry run is inconclusive (executor saturation, then teacher stop), not a judgement of case quality. The non-dry run was not started. Rerun in the next window with an idle teacher.
