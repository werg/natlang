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
