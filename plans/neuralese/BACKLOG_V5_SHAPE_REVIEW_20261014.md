# Backlog v5 teacher action-shape review (2026-10-14)

`luna-held-shapes-closed-backlog-v5-native-turns-20261014-v1` had already
admitted exact-oracle successes. Exact final answers do not certify that the
teacher followed the natlang process requested by the source program. Pop
scanned every approved train decision with
`scripts/neuralese_data/audit_teacher_action_shape.py` and pinned the result
in `BACKLOG_V5_SHAPE_REVIEW_20261014-v2.json`. The source file SHA-256 is
`6ab5c37b559aa00c461ba253d2c241810950fa1e93dfb4fd0db27aaa94fa3cd2`.

Of 3,029 distinct parents, **1,012 are held** by exact action-shape checks:

| Source family | Held parents | Observed structural defect |
| --- | ---: | --- |
| composed process | 129 | Required `iterateOn` absent; many exact answers came from arbitrary capped loops. |
| edit stream | 101 | Required stepwise `iterateOn` and NL edit absent. |
| folder criteria reducer | 374 | Per-ticket NL criterion absent; many selected literal filenames. |
| workbench tickets | 12 | Required inline NL decision absent. |
| actor greenhouse | 140 | Required measured `iterateOn` controller absent; manually repeated actions. |
| iterate schedule repair | 256 | Required `iterateOn` repair process absent; many manually repeated calls. |

The first scan covered only four families and held 614 parents. A wider
family-by-family scan found the greenhouse and schedule omissions, so that
witness was replaced before the teacher delta build. The canonical witness
`training/source-quarantines/backlog-v5-teacher-shape-20261014-v2.jsonl`
(SHA-256 `29470612eecd21953f11282aa4e23dc0e9677df81123dde77807722f03bd0d68`)
contains 1,877 source groups. Use it as a `--quarantine-reference` in every
target union that includes backlog-v5 source renders. Group-level exclusion
also removes 28 otherwise unflagged parents that share a held group. A later
source-level recuration and rerender can recover sound siblings after semantic
review. Other backlog-v5 families remain candidates; this structural scan is
not their full quality certification.

The LFM and Mellum R8 teacher delta must include this witness in addition to
its existing protected held and source-quality references. Do not train from
the unfiltered backlog-v5 render. The currently running CoreWeave PH snapshot
does not include this proposed delta and needs no mutation.

## Source-design cause and new allocation

The same failure recurred in the live s52065/s52066 Luna batches: eight
accepted composed-process programs used no `iterateOn`; eight accepted edit
streams used no `iterateOn`; 14 of 15 accepted folder reducers made no inline
NL criterion call. In the source builders, `curriculum.minimum_sequence` and
`iterate: required` describe the intended training shape, but the executable
root instructions for these families do not clearly ask for it. The teacher
can satisfy the visible task and exact final oracle with ordinary loops,
string edits, or literal file names. This is a source-design and admission
defect, not evidence that the teacher cannot use the natlang operators.

The Luna replenisher stops *new* allocation for composed_process,
edit_stream, folder_criteria_reducer, actor_greenhouse, and
counterexample_revision, and iterate_schedule_repair while their task
surfaces and process checks are reworked. Existing campaign workers finish
and their rows remain available for review. The 100-case target stays supplied
by the other families, with more inline_review_each, recurrence_inline_graphs,
and knowledge_research cases.

For folder reducers, Pop prepared a paired 48-case-per-arm live-executor
comparison under
`/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1/folder-criteria-process-ab-20261014-v1`.
Arm A keeps the original root instruction; arm B asks for an inline NL
function applied to each ticket and the criteria. Both use the same 48
source cases and request seeds. The campaign is evaluation-only and has no
training admission. Measure exact task success, actual NL-per-ticket use,
requests, and latency before adopting B or changing the source builder.

The completed paired A/B/C live trial used 48 identical cases per arm. A
returned exact results on 48/48 but used an inline NL call on only 5/48.
B returned exact results on 45/48 and used NL on 48/48; median 52 model
requests, 3,350 completion tokens and 116.4 seconds per case. C asks for one
reusable inline Boolean ticket judge and returned exact results on 44/48,
with NL on 48/48; median 46 requests, 3,039 completion tokens and 104.6
seconds. A used median 6 requests, 433 completion tokens and 14.1 seconds.
The B/C failures were semantic ticket misclassifications, not transport or
runtime errors. In inspected successful C trajectories, the root awaited
the per-ticket judgments and moved files based on those returned Booleans.
The observed exact-success difference between B and C is one case; C has a
simpler reusable function shape and lower median cost, so `folder_criteria_reducer`
family version 2 adopts C's wording and marks inline use required. Its
reference now expresses the same NL judgment flow rather than hardcoded
filename moves. All generated results still need exact oracle and action-shape
review before training admission. A separate medium-reasoning Luna repair
targets the one case both B and C missed.

## Greenhouse process trial

The completed 48-case Luna greenhouse trial compared the original source with
two visible instruction variants on the same cases. V1 explicitly asked for
`iterateOn`; 37/48 outcomes were exact, but only **22/48** exact trajectories
actually called `iterateOn` in executed `eval` code. V2 also explained how to
stop a greenhouse with ineffective controls as `blocked`; 42/48 outcomes were
exact, including 7/12 broken-heater cases, but only **18/48** exact trajectories
actually called `iterateOn`. All seven accepted broken-heater V2 trajectories
used another control form. The earlier counts derived by searching complete
trajectory JSON were invalid because they matched the task's instruction text.
This review counts only `assistant.calls[].arguments.code` in action turns.

The two arms together provide 30 distinct exact cases with an actual
`iterateOn` call. Six source groups from seed 52041 overlap an existing accepted
parent corpus, so global source-group filtering is still required. V2 improves
honest stopping but does not improve process compliance; neither variant was
adopted as the generator source. `curriculum.iterate: required` currently produces only an
`iterate_missing` note in admission, so exact outcomes alone cannot be counted
as iteration examples.

The V3 instruction makes the state, control step and awaited
`iterateOn(...).until(...)` shape explicit. On 48 new, source-disjoint training
cases from seeds 52086 and 52087, it returned 38 exact outcomes and **all 38**
have recorded `iteration_step` events. The normal, sun and long variants were
36/36 exact; the broken-heater variant was 2/12 exact/blocked, with ten
failures to end as blocked. V3 is adopted for new greenhouse source generation
as family version 3, and the replenisher restores its quota of ten. Teacher
admission still requires both the exact oracle and recorded iteration steps.
Broken-heater failures are rejected; a later stopping-focused repair may
improve that slice.
