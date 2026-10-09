# Natlang-native review, 2026-10-09

The owner asked for a review of where the codebase should be more natlang-native ("think in natlang-native terms", AGENTS.md).
Three read-only reviews covered ts-host/src, applications/, and scripts/ plus training/. This document ranks the findings
and orders the work. Line references are as of origin/main at the time of the review.

## The pattern behind the findings

Three things keep pushing work toward crisp code:

1. **No natural home for built-in natlang programs.** The runtime has no loader for `.nl` files that ship with natlang
   itself. Built-in judges live as `CallableDefinition` string literals in TypeScript: the progress judge in
   runtime/iterate.ts:127-166, the behavior judge in calls/judge.ts:45-60, the digest prompt, the game policy. Writing
   policy as TypeScript is simply the easiest path.
2. **Four mechanisms for "crisp or natural-language".** These are `pluggable()`, a service-level `implementation(point)`,
   `settings.x === 'crisp'` ternaries, and hand-written dispatchers. There are also three vocabularies for the same mode:
   `nl`, `natlang` and `natural-language`. Shadow comparison only exists where `pluggable()` is used.
3. **Review and judgment work happens ad hoc in agent sessions** instead of in programs. Examples are the hourly check-in,
   dataset-item review, failure explanation and review-packet verdicts. That work leaves no traces, so it can neither be
   specialized nor trained on.

## Enablers (no owner review needed; mechanism only, no model-facing change)

| # | Work | Why first |
| --- | --- | --- |
| N1 | A built-in `.nl` program loader in ts-host: one folder of natlang programs shipped with the runtime, loaded like callable folders | Gives every judge and policy a natlang home |
| N2 | Move embedded natural-language functions to `.nl` files byte-identically: progress judge, predicate prompts, behavior judge, digest, game policy, benchmark prompts. Prompt-piece ids and converter tests unchanged | Relocation without semantic change, so no live measurement |
| N3 | One `pluggable()` mechanism and one mode vocabulary (`crisp`, `nl`, `shadow`) across all apps and the runtime. The `implementation()` service feeds `pluggable()` | Gives every hot path shadow comparison, and removes four mechanisms in favour of one |
| N4 | Delete duplicates: build `mismatch()` vs `step/validity`, scheduling `order.ts` vs `Problem.order`, program-improver `parents.ts`/`selection.ts` vs optimization/strategies/gepa.ts | One implementation per concept |
| N5 | Fix DECOMPOSITION.md mismatches: compilers (5 planned files missing), nldb (4), and a stale wiki reference | The documents must match the code |

## Policy to move into natlang (each needs a DECOMPOSITION.md and an owner-review entry first)

Ranked by value.

| # | Where | Decision today | Natlang design |
| --- | --- | --- | --- |
| P1 | No code: the hourly check-in done by agents | Run health, cause, resume/replace/wait, next step | Withdrawn (owner, 2026-10-10): the hourly check-in is an agent semantically controlling the work. Decoupled from the agent it is useless unless it triggers one, so it stays with the agents. The implementation was removed. |
| P2 | teacher/source-review.ts:19-440 | About 400 lines of per-item review verdicts frozen as code | Verdicts move to a data file. `reviewSourceItem` recommends at intake; `resolved` stays an explicit decision |
| P3 | calls/tiers.ts:203-225 | Tier promote/demote by magic thresholds | `decideTierState.nl`, pluggable, crisp hard bound. In progress |
| P4 | optimization/authored-engine.ts, gepa.ts, applications/program-improver | Parent, component and composition choice (`iteration%4===3`), stop | `planExperiment.nl`, pluggable, with a journaled plan for resume (implemented 2026-10-09 as six pluggable policies and a journaled plan; see applications/program-improver/DECOMPOSITION.md) |
| P5 | scripts/audit_student_projection_failures.py:9-37, ts-host/scripts/admission-dispositions.mjs | Failure and rejection categories by substring; fallback "needs review" | `explainFailure` and `triageRejections` for the unclassified bucket only, advisory |
| P6 | teacher/curriculum.ts:24-28, curriculum-policy.ts:17-140 | Mix targets, per-family legacy rule ladders, extractive-answer equivalence | The rule ladders become a data registry. `judgeAnswerEquivalence` releases rows to human review. `nextCollectionBatch` is advisory |
| P7 | applications/compilers/index.ts:111, 212, 232-238 | The crisp driver re-implements the natlang pass manager's policy | `index.ts` becomes a verifier and service runner only |
| P8 | applications/logs/index.ts:16, 118-147; workflow/index.ts:25, 96-101; specializer/main.ts:31-55, 155, 245 | Status ladders, retention, timing defaults, "worth looking" | Stages return the status; crisp checks that it is allowed. `retire`, `worthLooking` and `summarizeDecline` are pluggable or NL |
| P9 | Mostly-crisp apps: helpdesk, terminal, media, notebook, evidence, publisher, ide | Triage, deadlines, escalation, ranking, recovery, user messages in TypeScript | Redesign per the sketches in the review, each with a DECOMPOSITION.md |
| P10 | Review-packet scripts (build_*_review_packet, bind_*_receipts) | The semantic review is done outside the repo | `reviewSourceRow`, with an optional second reviewer. The receipt records `reviewer: natlang@hash` plus the decision |
| P11 | eval gates (foundation.py, gate diagnostics) | Failed-gate explanation by hand | `explainGateFailure` after a gate fails; it never changes the gate |

## Model-facing cleanups (each must be measured live; log in MODEL_FACING_CHANGES.md)

- "Do not / never" guards in older apps. Convert each to a positive sentence plus a runtime check:
  - media/choose.nl, media/assess.nl;
  - terminal/explain.nl, terminal/interpret.nl;
  - ide/describe.nl;
  - notebook/choose_cell.nl, choose_goal.nl, explain.nl;
  - evidence/plan_search.nl;
  - publisher/compose.nl;
  - compilers/opt/simplify.nl, mem2reg.nl, inline.nl;
  - program-improver;
  - specializer/writeCase.nl.
- Split too-coarse functions:
  - program-improver/rewriteProgram.nl (a ~600-word bundle);
  - publisher/compose.nl;
  - media/choose.nl;
  - terminal/interpret.nl;
  - ide/interpret.nl (offsets become a crisp `locate`);
  - notebook/explain.nl.

## Correctly crisp (keep)

- **Model plumbing:** model/scheduler.ts (request batching, microsecond hot) and retry/backoff classification.
- **Exact checks:** compiler/policy.ts and source-policy.ts (exact syntactic checks), and adoption, rollback, provenance
  and digest code.
- **Gates and admission:** exact gates (the decision of a gate stays crisp; only its explanation becomes natlang),
  admission records, manifests, hashing, pins and training math.
- **Exact algorithms and parameters:** mining/induction algorithms and their parameters, and the call store.
- **Process control:** dispatcher identity, claim and recovery mechanics, and memory_ledger.py.

## Order of work

1. Enablers N1–N5 now: no owner review needed and no model-facing change.
2. DECOMPOSITION drafts for P1, P2, P4 and P9 (the 10 apps without one: helpdesk, terminal, media, notebook, evidence,
   publisher, ide, specializer, program-improver, refine-data), queued in OWNER_REVIEW.md. Nothing is restructured
   until the owner has reviewed.
3. P3 is already in progress. P5, P10 and P11 are advisory and read-only, so they can follow once their decompositions
   are reviewed.
4. Model-facing cleanups in the next teacher window, with live measurement.
