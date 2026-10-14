# QAD matched48 recovery dragnet (Pop, 2026-10-14)

## Observed result

The protected 48-case packet was run with the same full prompt, temperature
0.6, one worker, per-case seeds, native executor and request limits against
clean QAD base and the direct-stage step-15000 weights. Each passed 1/48. The
base had 41 semantic mismatches, three invalid actions, two request-budget
exits and one incomplete result; no repetition stop. Step-15000 had 21
repetition stops, ten invalid actions, nine request-budget exits and seven
semantic mismatches. Both arms completed without infrastructure failures.
The source packet and trace artifacts are under
`/srv/storage/natlang-artifacts/qad-base-vs-15000-matched48-w1-20261014-v1`.
It is held evaluation material and must not enter training or repair queues.

The student now attempts useful tool work more often than the base, but its
own continuation frequently fails to recover from feedback. Among the 21
repetition endings, the final repeated action was `read_code` in 11, `eval` in
nine, and `return_result` in one. The last observation in seven carries a
specific parser/runtime correction: two invalid service-binding shadowings,
two undeclared fields, one TypeScript syntax error, one `ReferenceError`, and
one unbounded iteration. The other 14 include repeated help or service reads,
world-state feedback and an incomplete result. The final repeated action is
only a triage label, not necessarily the first mistake in a trajectory.

## Diagnostic work before an approach verdict

1. For each repeated `read_code`, compare consecutive request payloads,
   tool-result text, invocation scope and prompt history. Confirm that the
   model actually sees the returned source or help in the next call and that
   the result is not truncated or attached to a different child. Separate
   legitimate new-scope reads from no-progress same-scope reads.
2. For repeated `eval` failures, identify the first failed action, the exact
   one-sentence feedback, and whether the next action copied, mutated or
   corrected it. Check binding availability, executor rendering, error text,
   tokenization and any context-window loss. The repeated `const cancelTickets`
   declaration and `read_code(nl)` pattern are examples to trace, not prompts
   to teach the model.
3. Compare the step-15000 and clean-base distributions on the same early
   decisions. Check reasoning-to-action transition, stopping, context length,
   action decoding, loss masks and train/full-prompt proportions. The exact
   train-only action audit in `QAD_ACTION_TARGET_DRAGNET_20261014.md` found
   correct action suffixes fully supervised and prompt-token parity for two
   recovery decisions, but two decisions do not cover this packet.
4. Collect teacher recoveries from **disjoint train sources** for the common
   failure families, using real failing action and feedback handoffs. Admit
   only exact-oracle successes with correct continuation and held-group
   exclusion. Include them in the next declared broad union; then remeasure
   matched task behavior, not merely teacher-forced action NLL.

The present evidence does not distinguish insufficient recovery training from
a prompting, serving, state or template bug. Preserve that uncertainty while
running the checks above. The new Luna closed backlog contributes 23,651
approved train targets, including many nested semantic actions, and its LFM
render is registered and synced to DGX. Its next broad-union inclusion is
pending; it was not part of step-15000.

## Train-only closed-backlog action audit (2026-10-14)

Pop scored six source-disjoint, admitted closed-backlog decisions from six task
families using the step-15000 QAD checkpoint, both `program-header` and `full`
LFM renderings, and the exact serving prompt renderer. The report is
`runs/qad-action-target-mask-audit-20261014-v1/closed-backlog-report.json`, SHA-256
`d432c8fa5df59c674197d6c111cb9f7284e68de3d9435114d38b5d991c726c70`;
selection and source row hashes are in the adjacent
`closed-backlog-selection.json`. All
twelve training prefixes equal their live serving token IDs exactly (lengths
1,699–2,223 header; 5,949–6,632 full). Each target occupies a train window
with positive assistant loss weight, zero excluded target positions, and no
truncation in this selection. Gold tool names have perfect argmax agreement
across all twelve observations (two to three tokens per case); action delimiter
tokens also agree. Most disagreement is in reasoning/reply tokens (per-case
agreement 0.415–0.625), with some argument differences. This rules out a gross
mask or render mismatch for these twelve teacher-forced *new* decisions; it
does not explain held task repetition or establish executable success. The
first argmax disagreement is at target token zero for each case, which is
sensitive to free-form reasoning choice. Next inspect actual repeated held
continuations and train-only recoveries at their first failed tool action; do
not use held traces as training targets.

Four independent temperature-0.6 samples on each of those six full-mode
train decisions, with a 192-token action window, are recorded at
`runs/qad-action-target-mask-audit-20261014-v1/closed-backlog-sample-report.json`
(SHA-256 `1f4e93d112d2acd7a38ce72b5e9a38764cf12b9ca77822b686ca207ffa98b847`).
All 24 emitted a tool call; 23 reached an ordinary stop and one hit the local
sampling length cap. Thirteen used the gold tool name, while the rest often
read a service first. This checks immediate action formation only; it does
not execute actions or grade answers. Some emitted code shadows a service,
calls `iterateOn` without a structural stopping measure, or includes stray
reserved-token text. Inspect raw tokenization and training provenance before
classifying the latter as a template or tokenizer bug.

In one held workbench repetition, the exact prior `read_code(nl)` result is
present in later provider requests, their message count grows from four to
sixty, and request hashes and sampling seeds differ. The model still repeats
the read. This rules out missing tool feedback or a cached identical request
for that child only. Across 20 retained partial traces of the 21 repetition
endings, the first observed difficulties include unbound names, service-name
shadowing, nonexistent fields, malformed TypeScript and an unbounded
iteration; the remaining repetition has a terminal result rather than a
partial. These are failure-family observations, not held training inputs.

## Whole-student train rollouts

The first attempted 12-family, two-seed rollout mistakenly used S7's default
authored-root packet. Its apparent one-turn successes and failures replayed the
reference root: `action_provenance=authored_reference_root_eval`, zero generated
tokens and empty raw model calls. That is useful for a *child-policy* experiment
but cannot diagnose the student's root choices. The unit was stopped and its
131 disposable output files were listed and deleted. No score or training
admission came from it.

`task_rollouts prepare --root-mode student` now explicitly removes authored-root
replay while preserving the train row and exact oracle, records source and packet
hashes in `preparation.json`, and makes scoring reject any replayed authored
root. The original `authored` mode remains explicit for child-policy S7. The
corrected source-disjoint train packet has 12 rows, two seeds each, source SHA-256
`63d3ce7ffdd1af6dd46ad282b40d0485111c04f715e3707bcac34f9694a8aa44`,
packet SHA-256 `52dbe9ae7878bc53f559f8abb89e71f096395a87c0fe111568a169fc34044f9d`,
and lives under `/home/werg/natlang-artifacts/qad-student-train-rollouts-20261014-v1/round1-student-root`.
The selected programs were checked against the actual protected 48 packet
(SHA-256 `63fc865bfdb666d3dbfe43b229708ed6469741eb532eafb1d9cd49894313d96e`):
zero shared program IDs, source IDs or source groups (19 train groups versus
44 held groups). The local `selection.json` records those counts and hashes.
The 24 train-only student-root rollouts are now terminal and scored: two exact
successes, ten semantic mismatches, nine invalid actions and three budget
exhaustions. Two cases hit the declared model-request budget with no result;
both have exact terminal journal evidence. There are no missing finishes or
transport errors. Two of twelve groups have success/failure spread (schedule
repair and composed process); the other ten require teacher seeding or
rejection sampling. The score is in `round1-student-root/score.json`; no
training admission has yet been made. The collector wrapper originally treated
the two declared budget exits as a fatal shard and skipped scoring; Pop merged
the complete journals and scored them, then changed the wrapper to accept only
exactly covered, declared model-request-budget exits when a collector returns
nonzero. Unexpected or missing finishes still fail closed. The wrapper also
stops its named Docker server before waiting for the Docker CLI, avoiding a
GPU-claim hang seen at the end of this run.

The actual failures include wrong answers after real tool work, syntax repair,
wrong field recovery, wrong schedule results and invalid world receipt calls.
Next review first failed actions and produce replay-checked teacher handoffs
from these train sources only. The held48 remains entirely outside the train
packet and all recovery queues.

Separate Luna generation triage: `s52048/worker-06/case-009` is an eight-message
dataset workbench program. Its first run exhausted the model-request budget;
the retry reached 824 model replies, about 95,000 completion tokens and 433
invocations without an exact output. Some child calls answered `blocked` when
no routing rule applied, although the root specifies the `editorial` fallback.
The reference has only 24 child judgments and all eight expected routes are
`editorial`. This is an extreme model/source interaction, not a hung provider:
request hashes were unique, responses and durable checkpoints advanced, and
the collection budget ended it. It stays out of training and automatic retries.
The failed attempt's redundant partial/evidence files have no repair or DPO
consumer; keep the compact review and queue status, prune those large files,
and review this task shape before producing another variant.
