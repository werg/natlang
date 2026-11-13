# Difficulty log and contrastive training

Keep failures from both models, including failures of our interface and audit.
Do not erase an earlier failure when a prompt change passes a later instance.
Student evidence here is from the v8 350M model; teacher evidence is from
Ternary Bonsai 2 27B PTQ1_0. Aggregate rates belong to the recorded runs, not to
unmeasured individual lesson categories.

| Stable lesson ID | Observed difficulty and evidence | Required contrasts |
| --- | --- | --- |
| `quote_vs_semantics` | Teacher silently converted explicitly preserved Text to Num (`teacher-behavior-s881.json`). Runtime transport quoting is a separate issue. | Allowed conversion versus explicit type preservation; incidental quotes versus changing the requested meaning. |
| `missing_vs_empty` | Teacher fabricated empty text for absent input (`teacher-behavior-final-s882-audit-v3.json`, `teacher-simple-full-s884.json`). Ambiguous missing-read rendering contributed. | Missing evidence must block; supplied empty text must succeed; supplied nonempty text must be copied exactly. |
| `exact_destination` | Teacher redirected an impossible direct return into a record field (`teacher-simple-full-s886.json`). Earlier teacher runs also redirected to locals. | Impossible direct binding versus explicitly requested field binding; reject a plausible type-correct redirection; persist with the correct explicit field binding. |
| `ordered_effects` | Teacher attempted an error before the required earlier effect, with a truncated error payload (`teacher-simple-full-s886.json`). | Preserve required prior effects on later failure; no duplicate effects; distinguish intended failure from malformed transport. |
| `branch_fidelity` | Teacher produced a wrong value on a selected failing branch in the longer-prompt probe (`teacher-behavior-prompt-s881.json`). | Same program with opposite input flags; selected impossible branch fails, selected valid branch succeeds. |
| `honest_impossibility` | Student failure-generalization and frozen-review probes accepted wrong proposals despite type checks (see AGENT_SUPPORT.md and CAREFUL_MODE.md). | Contradictory bounds versus feasible bounds; correct error versus unsupported error on a feasible task. |
| `count_vs_element` | Student broader comparisons confused computed results and inputs; teacher used caller collection names instead of formal parameter names (`teacher-behavior-compact-s881-rescored.json`). | Count versus selected element, with values deliberately unlike counts; Text and Num inputs; exact formal argument binding. |
| `source_copy` | Teacher copied a scalar into a whole record; heterogeneous tool schemas produced malformed values. Student execution-state experiments did not solve this reliably. | Explicit field copy versus whole-record copy; computed value versus fabricated replacement. Keep transport fixes separate from semantic lessons. |
| `closure_is_not_execution` | Student action reviews frequently approved incorrect marks/proposals; see frozen 94-proposal comparisons in AGENT_SUPPORT.md. | Required call versus premature mark; correct completed work versus unjustified doubt; done/skipped differ but both close lines. |
| `map_fold_binding` | Teacher supplied the mapped item twice or a quoted numeric fold initializer (`teacher-simple-full-s884.json`). | Batched map/fold versus plain call; leave implicit parameters unbound; valid initializer quoting versus a genuine incompatible type. |
| `honest_check_and_persistence` | Student baseline approved 56/60 bad proposals; checklist still approved 29/60 while rejecting 19/34 good proposals (frozen s99105 comparison). | Wrong proposal: withdraw. Correct challenged proposal: approve with evidence. Impossible task: error. Missing information: blocker. A challenge alone is not evidence of error. |

Run paths above are under `runs/`. The teacher setup and trial sequence are in
[TEACHER_SETUP.md](TEACHER_SETUP.md). This log distinguishes observed mistakes
from inferred causes: prompt examples may encourage redirection, but that has
not been isolated as its cause.

Follow-up: `teacher-destination-effects-s886.json` passed all eight affected
failure/control cases across two groups after the short wording changes. Keep
the original failures in this log: a passing follow-up does not remove the
need for those training contrasts or establish general reliability.

## Training implementation

`scripts/generate_agent_support.py --reviews ...` creates paired proposals at
the same pre-action state. Execution references are run through the runtime;
wrong proposals are deliberately constructed contrasts. Each review records:

- `contrast_group` and `program_id`: keep every related example in one split.
- `lesson_ids`: link the example to this log.
- `proposal_justified`: whether the exact proposal should be released.
- `task_feasible`: true, false, or unknown because evidence is missing.
- `label_source` and `confidence`: oracle provenance; confidence is null unless
  actually measured. Correct failure reports are themselves justified actions.

Both correct and incorrect proposals receive the same distribution of skeptical
check-up wording. Correct targets explain the specific requested destination,
input, or value. Incorrect targets distinguish withdrawing an action from
declaring the whole task impossible. Additional contrasts include unsupported
failure claims on feasible tasks, correct error/blocker reports under challenge,
and the teacher's observed scalar-to-record-field redirection.

Current generated coverage includes all listed lessons except dedicated
map/fold review pairs and a separate supplied-empty review pair; those remain
covered by teacher/runtime probes, not yet the contrastive generator.

## Calibration work

The labels above support measurement; they are not invented probability targets.
For future student evaluation, capture confidence **before** executing the
proposal, join it to independently audited `proposal_justified`, and report
reliability bins, Brier score, wrong-proposal acceptance, and correct-proposal
withdrawal. Break these down by lesson and by whether the model already made a
commitment that a correct continuation cannot honor. Calibrate thresholds only
on a development split, then evaluate on fresh instances and held-out templates.

Tool-token likelihood and an explicit self-reported probability are different
signals. Neither is automatically a probability of semantic correctness. An
explicit confidence head/tool remains an experiment; it is not added to the
default execution interface. Train and evaluate persistence alongside honesty
so blanket doubt cannot masquerade as calibration.

## Proposed help/escalation experiment

User direction: investigate whether training an explicit confusion/help action
improves honesty more broadly. Keep this distinct from known impossibility
(`report_error`) and missing required information (`report_blocker`). A proposed
`request_help` action would mean that the agent cannot reliably determine the
intended operation; it would return the current state and a specific explanation
to the caller, without automatically repairing instructions or retrying.
This tool is not implemented or present in the current training batch.

Use matched pairs: uninterpretable text versus unusual meaningful wording;
instructions and argument roles that do not match versus unfamiliar but
explicitly defined names; genuinely ambiguous instructions versus difficult
unambiguous tasks. Include nonsense strings as legitimate data so their mere
presence does not become an escalation cue. Do not teach a model to claim a
known contradiction merely because it is uncertain.

Targets should identify the unresolved interpretation and the smallest useful
clarification, not merely say "I'm confused." Evaluate both appropriate help
requests and unnecessary escalation on solvable controls, along with execution
accuracy and downstream caller resolution. Any transfer to general honesty or
confidence calibration is a hypothesis, not an established benefit. Initially
keep this as an isolated tool/training ablation so it does not complicate the
default runtime or hide regressions behind more frequent abstention.
