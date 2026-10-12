# Waiting for owner review

Decisions and documents that the owner rules say need the owner's review. Agents add entries. The owner, or an agent
acting on the owner's recorded answer, moves an entry to "Done" with the date and outcome.

## Open

| Since | Item | What to review | Status meanwhile |
| --- | --- | --- | --- |
| 2026-10-12 | applications/ops-retention/DECOMPOSITION.md (review rec. 7) | Storage-retention classification as a pluggable: crisp `crisp_verdict` (default) and `classifyRetention.nl` over the same `retention_facts`; shadow log of disagreements; exact bound refuses an nl delete of protected files | Implemented on main 65bf7108; review after the fact. One-week shadow run (`--classifier shadow`) per the DECOMPOSITION |

## Follow-ups from the 2026-10-12 acceptance

| # | Item | What needs doing |
| --- | --- | --- |
| F1 | Heartbeat leftovers | Delete the untracked, gitignored `applications/heartbeat/*.d.nl.ts` and `applications/dist/heartbeat` on DGX (and Pop if present); the program was removed in a0800c94. The reviewing agent's delete was refused by the permission classifier |
| F2 | refine-data `authoring.json` | Add the guard-sentence / `Is<T, "...">` pairs from the refinement tables of helpdesk, terminal, media, notebook, evidence, publisher, ide, specializer and program-improver (none registered yet; the 28 entries cover only the first seven apps) |
| F3 | FAILURE_EXPLANATION_PROGRAM questions 1-5 | Decide (agents may, per "keep moving forward"): explain tag 33 `answer_mismatch_needs_review`; judge redundant vs necessary delegations; `RULES` in `admission-dispositions.mjs` as a data table with manifest; confirm nothing tunes to the .99/.02/.05 thresholds; explanation files beside the reports or one `runs/explanations/` tree |
| F4 | Tiered engine and fused pipelines next steps | TIERED_ENGINE: tier 2 replay verification, persisted tier config, tier 1 confidence guard, tier 4. FUSED_PIPELINES: observed facts from live runs, wiki/migration chains and compilers `analyze -> declare`, consumer-side shadow, a real certificate with a qualified server and benchmark |
| F5 | Doc and consistency nits | `applications/migration/DECOMPOSITION.md` should name the `migrate/stop.ts` dispatcher; publisher keeps its own `read_note.nl` while terminal, media and notebook use the shared `readNote` built-in (unify or record why); `ts-host/src/improvement/authored-source.ts` must be regenerated (`bundle-improver.mjs`) after program-improver edits |

## Done

| Date | Item | Outcome |
| --- | --- | --- |
| 2026-10-12 | applications/games/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Every named stage is an .nl file; validate/settle/resolve/remember/narrate go through `pluggable(` with crisp rules as the counterpart; `Engine` is nl/crisp/shadow. Fixed two stale doc lines (`npc/respond`, shadow mode) |
| 2026-10-12 | applications/scheduling/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. All 10 stages exist; `enumerate` pluggable over `calendar.exact`; crisp is the verifier, revisioned commit and service binding only |
| 2026-10-12 | applications/workflow/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. `handle/choose/*`, `inform/*`, `when.nl` exist; `reviewAfter` and `handle` are pluggable, `allowedReview` bounds the NL answer; `service.ts` is single writer, receipts and transition verifier |
| 2026-10-12 | applications/wiki/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Merge and maintain stages exist; `changes.ts` and `staleness.ts` pluggable; `Policy` is crisp/nl/shadow with old values normalized by `pluggableMode` |
| 2026-10-12 | applications/logs/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. `investigate/*` and `retire.nl` exist; significance and eviction pluggable with crisp bounds; `statusAfter` accepts the stage's status only when receipts allow it |
| 2026-10-12 | applications/build/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Stages exist; ready/choose/validity pluggable with defaults crisp/nl/nl as documented; `BuildWorkspace` is the service, `step/commit.ts` the verifier. CLI help aligned with crisp/nl/shadow (also accepts `natlang`) |
| 2026-10-12 | applications/migration/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. `migrate.nl` iterates with a measure until the crisp `stop` callable, which selects `settled` via `pluggable()`; `site/edit/exact.ts` pluggable |
| 2026-10-12 | Crisp share of the rebuilt apps | accepted on owner's behalf 2026-10-12. Audit done in this review: in games, scheduling, workflow, build and logs every crisp file is a service, commit, verifier, CLI, plumbing or the crisp side of a `pluggable(` with an NL counterpart; no policy without one. Crisp line counts stay high because services and verifiers are large |
| 2026-10-12 | Pop's self-feedback gate thresholds (.99/.02/.05) | accepted on owner's behalf 2026-10-12 as report-only. `eval/self_feedback.py` emits only `proposed_gate_passed` and nothing reads it; Neuralese direct v1 qualification does not use them. TRAINING_RECIPE.md now says so |
| 2026-10-12 | plans/NATLANG_NATIVE_REVIEW.md | accepted on owner's behalf 2026-10-12. N1-N5, P2, P4-P11 verified on main (compilers `index.ts` is verifier and runner only; P8 logs, workflow, specializer pluggable). Added a status paragraph: P3 was built as `promote.nl` plus `promotionPolicy`, P1 withdrawn |
| 2026-10-12 | Scheduling task order (N4) | accepted on owner's behalf 2026-10-12. One order: `scheduler/order.ts` `topologicalOrder` (depth first in task order) is what `Problem.order()` calls; `domains.nl` keeps that order |
| 2026-10-12 | Mode vocabulary in model-facing text (N3) | accepted on owner's behalf 2026-10-12. wiki and logs `Policy` and games `Engine` are crisp/nl/shadow; `pluggableMode` accepts `natlang`/`natural-language` as deprecated `nl`; `migrate.nl` calls the crisp `stop` callable |
| 2026-10-12 | `stoppingCondition.nl` placeholder (N2) | accepted on owner's behalf 2026-10-12. `{progress}` is filled in `predicatePrompt` (ts-host/src/runtime/iterate.ts) with a comment explaining the byte-pinned system addendum |
| 2026-10-12 | applications/helpdesk/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Triage stages split; deadline, inbox order and escalation plan are `pluggable(` with `RESPONSE_TIMES`, `crispInboxOrder` (permutation check) and `crispEscalationPlan` as crisp sides; plan is data |
| 2026-10-12 | applications/terminal/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. `chooseRecipe.nl`, shared `readNote` built-in, `recipes.json` and `messages.json` as data, `cancel_requested` on the event |
| 2026-10-12 | applications/media/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Per-operation `plan*` stages; crisp `assemblePlan`/`checkPlan` and one retry; assess only when technical_ok; Plan has container and codec; tolerance is a setting |
| 2026-10-12 | applications/notebook/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Ready-cell choice pluggable (`lowestId` default); crisp evidence facts; `iterateOn` with a measure; 2 s timeout settable |
| 2026-10-12 | applications/evidence/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Split stages exist; host fills revisions and truncation; one repair; `select` returns `HitIds`. Fixed the stale type row (membership is a host check) |
| 2026-10-12 | applications/publisher/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Per-section compose in parallel, `unsourcedNumbers` scan, host-filled revisions, whole brief rejected on a twice-failed section |
| 2026-10-12 | applications/ide/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. Crisp `locate` with anchor feedback to `proposeEdit`; `describe` split into two view stages |
| 2026-10-12 | applications/specializer/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. `worthLooking`, `summarizeDecline` pluggable; `writeCase` split into `chooseCondition` and `writeBody`; rounds a named limit. Fixed stale `chooseCondition/*` names |
| 2026-10-12 | applications/program-improver/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12. diagnose/hypothesize/edit stages, search policies pluggable over .nl with crisp verifiers, one GEPA module (`ts-host/src/gepa`), reducers as the `transformations.ts` table |
| 2026-10-12 | applications/refine-data/DECOMPOSITION.md | accepted on owner's behalf 2026-10-12 (intent). The note records a decision for every part; status line updated. Not done: the P9 refinement pairs in `authoring.json` (follow-up F2) |
| 2026-10-12 | plans/HEARTBEAT_PROGRAM.md (P1) | accepted on owner's behalf 2026-10-12 as withdrawn: the owner removed it 2026-10-10 (a0800c94); plan and app sources are gone from main. Gitignored generated `applications/heartbeat/*.d.nl.ts` remain on DGX (follow-up F1) |
| 2026-10-12 | plans/SOURCE_REVIEW_PROGRAM.md (P2, P10) | accepted on owner's behalf 2026-10-12. Verdicts in `training/source-reviews/*` with manifest, fingerprints pinned by `source-review-migration.test.mjs`, `reviewSourceItem.nl`/`reviewSourceRow.nl` built-ins, receipt with `natlang@<hash>` and an explicit signed decision |
| 2026-10-12 | plans/FAILURE_EXPLANATION_PROGRAM.md (P5, P11) | accepted on owner's behalf 2026-10-12 (intent and implementation). The three .nl stages and entry points exist; `explain_gate.py` never changes a gate. Plan status line updated. Questions 1-5 still undecided (follow-up F3) |
| 2026-10-12 | P6 curriculum rule ladders | accepted on owner's behalf 2026-10-12. `curriculum-rules.ts` registry with differential test; `judgeAnswerEquivalence` gives only human_review/keep_held with admission unchanged; `nextCollectionBatch` has no caller beyond the advisory script |
| 2026-10-12 | plans/TIERED_ENGINE.md, plans/FUSED_PIPELINES.md | accepted on owner's behalf 2026-10-12. Tier classes match the numbering; promotion via `promote.nl` + `promotionPolicy`; fusion planner, observed facts, call-site keys and the explicit compilers chain exist. Remaining next steps listed in follow-up F4 |
| 2026-10-09 | Self-feedback (AR) qualification criterion | Channel faithfulness on the model's own history (plans/neuralese/DECISIONS.md) |
