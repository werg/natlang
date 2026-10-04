# Student reliability, task skills and supported task scope

## Objective

Develop a useful task scope with at least 90% end-to-end success after runtime
adaptation, skills and post-training. Report coverage of the broader target mix
alongside reliability. Do not claim this from guided projection yield.

Projection admission asks whether teacher-initialized search produced a changed,
verified trajectory. It is not ordinary model accuracy. An unchanged correct
chain can produce no candidate; an accepted proposal can still be rare during
ordinary execution. Preserve these separate measurements.

## Failure evidence and reusable skills

`scripts/audit_student_projection_failures.py` streams saved proposal receipts,
pins input hashes, and groups overlapping observable symptoms. The v1 review is
`runs/student-posttraining-20261004/failure-clusters-v1.json`: 46 completed cases,
23 candidate episodes / 170 turns. Its labels require review:

| Pattern | Failed proposals tagged | Response to investigate |
| --- | ---: | --- |
| Answer mismatch | 167 | Split semantic judgment, extraction and exact computation through trace review |
| Repeated action | 127 | API inspection, state recovery, reuse of completed work |
| State recovery | 101 | Inspect the state after failure; effects may have survived |
| API or type contract | 73 | Read declarations; avoid invented properties and argument shapes |
| Incomplete execution | 41 | Distinguish loops, wasted exploration and genuine budget needs |
| Truncated generation | 20 | Review output format and per-case allowance; not a semantic negative |
| Finish protocol | 13 | Correct stop shape and verify typed completion |
| File/effect state | 11 | Verify actual writes independently of returned names |
| Delegation review | 6 | Check necessity and reuse; delegation itself is not a failure |
| Dishonest stop | 6 | Do not convert incomplete plans into success |
| Replay infrastructure | 1 | Live display identity fixed; keep strict replay equality |

Tags overlap. Prefix actions in suffix proposals can repeat and may be legitimate.
Counts are not distinct bugs or evidence that a particular skill improves results.
No failed output is converted to positive SFT or a DPO label by the audit.

Initial general library: `training/student-skills/task-workflows-v1/skills`:
calculate-from-data, move-and-verify-files, resume-after-partial-effects,
inspect-callable-apis, complete-the-call and judge-against-criteria. These
contain procedures and examples, no source-specific answers or identifiers.
Descriptions state when to use each skill. Use a small applicable topic pool,
not mandatory disclosure of every body. Metadata/descriptions are editable in
future skill optimization; this student's initial text library is crisp.

Direct Program IR execution now prepares its root through the same
`prepareDefinitionNode` binder as public calls. The collector previously used
raw `definitionNode` and skipped companion skill binding. The model sees names
and descriptions, and uses `read_code("skills.<name>")` for the body. The host
places selected files in `<entry-without-.nl>/skills/`. Program IR overlay hashes,
source-group lineage and library hashes must be pinned. Do not mutate historical
IR, trajectories, running runtime snapshots or teacher provenance to add skills.
The scripted collection test establishes discovery/disclosure, not model uptake.

## Experiment sequence

1. Finish broad projection v5 and recovery-guidance v6. Preserve all receipts.
2. Freeze a skill-enabled runtime and build versioned case overlays with shared
   source-group membership. Compare ordinary execution using a fixed student:
   no skills; skills available through discovery; explicit applicable-skill
   instruction. Keep tasks/oracles, deployment retries and budgets comparable;
   measure skill reading cost and usage. Use train/support cases for development.
3. If instructed skills help but discovery does not, collect successful
   skill-discovery/read/use trajectories from stronger teachers and verified
   student search. If both fail, improve the procedure or task support first.
   SFT only admitted complete trajectories; retain matched failure/repair evidence
   separately. DPO requires verified alternatives from the same visible context.
4. Run the audited resumable Muon post-training phase with rehearsal and protected
   held-out rows unchanged. Evaluate ordinary execution before/after. Keep skill
   discovery, execution correctness and unsupported-task routing separate.
5. Consider verifier-based RL only after reward/effect checks and deployment
   policy are stable. Do not reward malformed finishes, fake effects or merely
   matching answer strings when the task requires actions.

## Task difficulty and a defensible 90% subset

Use three provisional labels: supported; remediation candidate; unresolved at
this checkpoint/budget. A low guided-search yield alone cannot establish that a
task is too hard. Require repeated ordinary-execution evidence and unsuccessful
reasonable interventions before assigning unresolved. Keep model, skill library,
runtime and budget in that disposition; revisit it after training.

Routing features must be available before seeing the answer: declared return
shape, operation/effect requirements, visible input size and item count,
accessible APIs, need for research, and structural/nesting requirements. Avoid
case IDs, oracle answers and rules that simply select previous successes.
A fixed validation set can develop routing rules; qualification uses unseen,
source-disjoint groups after the rule is frozen. Cases already used for adaptive
prompt/skill decisions are validation, not qualification. Keep an exclusion and
contamination ledger. No held-out answers enter skills or training.

For each fixed routing rule report eligible coverage, first-attempt success,
bounded deployment success, abstention/unsupported rate, error severity and
resource cost. Count an accepted unsupported task that fails as a failure;
report rejected tasks in coverage, not as successes. Use independent source
units or group-aware uncertainty, not repeated trajectories as independent
samples. Require a confidence lower bound of at least 90% on a meaningful
qualification sample before describing the scope as 90% reliable. Retries and
fallbacks are allowed only as part of the declared deployment policy and count
in cost. No reliable scope has been certified yet.

## Remaining work

The library and shared root binding are implemented; actual student skill-use
ablations, query/qualification partitioning, router fitting, skill-use SFT corpus
admission and the post-training run remain. Check per-function child binding as
well as root binding before relying on topic skills inside delegated calls.
Do not schedule a competing GPU server while v5/v6 owns the local device.

## Discovery is part of the intervention

The umbrella term recovery is reserved for resuming after partial effects. The
library itself contains task procedures that can prevent problems during normal
execution. Each name describes a job; each description gives an observable use
condition and the help offered. Bodies remain on demand. Track helpful skill
selection, missed applicable skills, unnecessary reads, execution benefit and
reading cost. Reading a skill is not success by itself. Description/name tuning
uses support groups; freeze the catalog before query/qualification evaluation.
The initial unexecuted recovery-named library was superseded and archived in its
planning packet. No training data or running model state was migrated.
