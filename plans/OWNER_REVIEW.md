# Waiting for owner review

Decisions and documents that the owner rules say need the owner's review. Agents add entries. The owner, or an agent
acting on the owner's recorded answer, moves an entry to "Done" with the date and outcome.

## Open

| Since | Item | What to review | Status meanwhile |
| --- | --- | --- | --- |
| 2026-10-09 | applications/games/DECOMPOSITION.md | Per-part decisions (function / instruction / implicit / crisp), crisp verifiers, pluggable points | Rebuilt app is on main; the review was skipped before restructuring, contrary to the porting rule |
| 2026-10-09 | applications/scheduling/DECOMPOSITION.md | The same | The same |
| 2026-10-09 | applications/workflow/DECOMPOSITION.md | The same | The same |
| 2026-10-09 | applications/wiki/DECOMPOSITION.md | The same | The same |
| 2026-10-09 | applications/logs/DECOMPOSITION.md | The same | The same |
| 2026-10-09 | applications/build/DECOMPOSITION.md | The same | The same |
| 2026-10-09 | applications/migration/DECOMPOSITION.md | The same | The same |
| 2026-10-09 | Crisp share of the rebuilt apps | Crisp lines exceed natural-language lines in games, scheduling, workflow, build and logs. Audit pending: which crisp files are services, commit or verifiers (allowed) and which hold policy | Audit queued |
| 2026-10-09 | Pop's proposed self-feedback gate thresholds (.99 argmax, .02 KL, .05 own-output CE gap) | Thresholds before they bind | Reported only |
| 2026-10-09 | plans/NATLANG_NATIVE_REVIEW.md | Ranking and order of the natlang-native work (enablers N1–N5 proceed now; P1–P11 wait for their decompositions to be reviewed) | Enablers N1–N5 on main; P7 (compilers driver) and P8 for logs and workflow implemented on main 914ff66a, review after the fact (specializer part is another agent's) |
| 2026-10-09 | Scheduling task order (N4) | `scheduler/order.ts` (model-visible) and `Problem.order` (verifier) give different valid orders (tasks a after c, b, c: c,a,b against b,c,a). Unifying either way changes the candidate order models see. Which one stays? | Implemented on main 914ff66a; review after the fact: one order, the verifier's (depth first in task order), in `scheduler/order.ts` and `Problem.order`; model-facing row recorded |
| 2026-10-09 | Mode vocabulary in model-facing text (N3) | Stage signatures still say `'crisp' \| 'natural-language'` and the wiki, logs and games settings types default to `'natlang'`; `migration/migrate.nl:29` selects `settled` inside its instruction text. Moving these to `crisp\|nl\|shadow` changes prompt bytes and needs live measurement | Implemented on main 914ff66a; review after the fact: service declarations, wiki and logs settings types and defaults say crisp, nl, shadow (games already did); old values accepted and normalized at run time; `migrate.nl` calls the crisp `stop` callable, which selects through `pluggable()`; model-facing rows recorded |
| 2026-10-09 | `stoppingCondition.nl` placeholder (N2) | The stopping-predicate note is a built-in with a `{progress}` placeholder filled by `predicatePrompt`, rather than a typed argument | Implemented on main 914ff66a; review after the fact: the placeholder stays. The text is the system addendum of the user's stopping predicate, not a call, so a typed argument (rendered in the arguments block) would change its bytes; the reason is in a comment at `predicatePrompt` |
| 2026-10-09 | applications/helpdesk/DECOMPOSITION.md | Per-part decisions (triage split, pluggable deadline and inbox order, escalation plan as data), refinement candidates, model-facing changes | Implemented on main 7251c66f; review after the fact |
| 2026-10-09 | applications/terminal/DECOMPOSITION.md | `readNote` and `chooseRecipe` split, recipe and message tables as data, cancel fact on the event | Implemented on main 63cd5af5, fa93a208; review after the fact |
| 2026-10-09 | applications/media/DECOMPOSITION.md | `choose.nl` split per operation, crisp plan assembly and retry, `assess` skipped on technical failure; transcode has no codec field (question 1) | Implemented on main 71e11794; review after the fact. Codec and container added to Plan; tolerance is a named setting |
| 2026-10-09 | applications/notebook/DECOMPOSITION.md | Ready-cell order as pluggable (crisp default), crisp evidence facts for the explanation, structural loop measure | Implemented on main ac938023; review after the fact. The 2 s cell timeout is kept as a settable default; natural language does not write cells |
| 2026-10-09 | applications/evidence/DECOMPOSITION.md | Host-filled revisions and truncation, `compose` split, one repair attempt, `select` ids as a type | Implemented on main eb087c15; review after the fact. Context-dependent refinements (hit ids, quotes) are host checks with one repair, not `Is<>` types; owner questions 1-2 left open; `Passage`/`Claim` stay owned by evidence |
| 2026-10-09 | applications/publisher/DECOMPOSITION.md | Per-section outline and compose in parallel, number scan, host-filled revisions | Implemented on main eb087c15; review after the fact. Section numbers are checked against the section's passages and table; no partial publish; the loader follows type re-exports in `types.ts` (shared `Passage`/`Claim`) |
| 2026-10-09 | applications/ide/DECOMPOSITION.md | Anchor edits with crisp `locate` instead of model-counted offsets, `describe` split | Implemented on main de56689d; review after the fact |
| 2026-10-09 | applications/specializer/DECOMPOSITION.md | Pluggable `worthLooking` and decline summary, `writeCase` split into condition and body, rounds as a resource limit | Implemented on main ce9aac1e (also tier 2 guidance, P8 specializer part); review after the fact |
| 2026-10-09 | applications/program-improver/DECOMPOSITION.md | `rewriteProgram` split into diagnose, hypothesize, edit, finish; search policy as pluggable NL; one shared GEPA module; reducers as a data table | Draft only; app unchanged |
| 2026-10-09 | applications/refine-data/DECOMPOSITION.md | Short note: three stages stay functions; app decompositions feed `authoring.json` | Draft only |
| 2026-10-09 | plans/HEARTBEAT_PROGRAM.md (P1) | `diagnoseRun`, `planNext`, `triageInbox`; crisp read-only collectors; action allowlist and auto-apply rule; advisory first and comparison with agent decisions; systemd timer; traces to specializer and training | Implemented on main ecc8ec81; review after the fact. Shipped in advisory mode; timer not installed yet (scripts/install_heartbeat_timer.sh) |
| 2026-10-09 | plans/SOURCE_REVIEW_PROGRAM.md (P2, P10) | Exact migration of 184 verdicts to a data file (fingerprints pinned), `reviewSourceItem` and `reviewSourceRow`, receipt with reviewer hash and explicit decision | Implemented on main (commits "Source review ..." of 2026-10-09); review after the fact |
| 2026-10-09 | plans/FAILURE_EXPLANATION_PROGRAM.md (P5, P11) | `explainFailure`, `triageRejections`, `explainGateFailure`: advisory, unclassified bucket only; gates never change | Implemented on main 42ddb338; review after the fact. Entry points: `audit_student_projection_failures.py --explain-input`, `explain_gate.py`, `ts-host/scripts/explain-advisory.mjs`. Questions 1-5 of the plan were not decided: tag 33 and `delegation_review` are not explained, `RULES` stays code (question 3), files sit beside the reports |
| 2026-10-09 | P6 of plans/NATLANG_NATIVE_REVIEW.md (curriculum rule ladders) | `curriculum-policy.ts` ladders as a data registry (`curriculum-rules.ts`, differential test against the original ladders); `judgeAnswerEquivalence` releases held extractive rows to human review only; `nextCollectionBatch` advisory | Implemented on main 14228bec and fa5f12c9; review after the fact. `source-review.ts` untouched |
| 2026-10-09 | plans/TIERED_ENGINE.md, plans/FUSED_PIPELINES.md | Tier numbering, natural-language promotion policy, natural-language fusion planner; fused-pipelines follow-ups (observed readers, explicit chains, call-site marks for crisp TypeScript) | In progress; fused-pipelines follow-ups: implemented on main, review after the fact |

## Done

| Date | Item | Outcome |
| --- | --- | --- |
| 2026-10-09 | Self-feedback (AR) qualification criterion | Channel faithfulness on the model's own history (plans/neuralese/DECISIONS.md) |
