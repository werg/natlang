# QAD action-target mask and rendering dragnet (Pop, 2026-10-14)

The step-15000 QAD LFM2.5 student scored 4/48 on the protected task packet.
Ordinary-token/cache parity on captured prompts and sampled control-token
checks narrowed serving explanations. This diagnostic asks whether a correctly
admitted train-only action would be supervised and rendered as the executor
will see it, and how the exact step-15000 weights score it. The selected
`s52054` recovery rows were generated after the r3 QAD training input froze;
they were **not** part of that checkpoint's training set. This is a pipeline
compatibility and generalization probe, not a claim of learned examples. It makes no
training-data admission or checkpoint decision.

## Fixed input and mechanisms

Use the registered `luna-service-result-recovery-s52054-v1` native turns,
source split `train`, exact runtime/oracle admitted. Select one parent with an
approved service call and approved final return; preserve its row IDs and
source hashes. Render only those decisions with the existing
`render_native_turn_corpus.py`/`gold_text_rows` path in full and program-header
modes, plus the whole-trajectory view. The renderer and tokenizer are crisp
mechanisms and their provenance/omission receipts decide whether this row may
be scored. Do not manually invent a target string or mask.

The existing `read_text_corpus`, windowing and `text_supervision` functions
own the direct-stage token weights and masks. Use them to report target-bearing
positions, actual assistant-action and tool/context weights, and whether the
entire correct action suffix survives the direct-stage window. Separately
derive the live prompt through the shared serving engine's `_prompt_plan` on
the same native `messages`/`tools`. Compare the prompt token IDs to the
training document prefix at the assistant boundary, identifying the first
difference and its neighboring decoded tokens. The model cannot be blamed for
a missing or differently presented action target.

Restore the exact step-15000 checkpoint and clean heads through the shared
`restore_engine`; use its existing backbone/readout to teacher-force the gold
suffix (no new inference stack). Score mean and per-position negative log
likelihood, argmax agreement, first disagreement, and breakdown by assistant
tool name, arguments and final answer versus context/tool positions. Pin model,
heads, native rows, renderer output, tokenizer and code hashes. Read-only
diagnostic artifacts stay outside the training registry. No optimizer update,
new corpus union, new training recipe or protected held trace enters this run.

## Policy

The judgment of whether the student lacks action supervision, sees a different
template, or struggles despite a correct mask is a natural-language review of
the exact measurements and source trace. Scripts may only render, hash,
teacher-force, and calculate exact token/mask statistics. A bad result is a
bug hypothesis until mask, boundary, weights, scoring and known-good reference
paths have been checked.

## First exact result (2026-10-14)

`scripts/neuralese_data/qad_action_target_audit.py` used the two admitted,
train-only `service_result_recovery` decisions from parent
`teacher-program:13911248983c7601fb54`, exact native source/row hashes from
`runs/qad-action-target-mask-audit-20261014-v1/selection.json`, the shared
renderer outputs, the resolved direct-SFT stage parameters, step-15000 weights
and the clean QAD heads. Raw machine-readable receipts are
`runs/qad-action-target-mask-audit-20261014-v1/report.json` (local
diagnostic artifacts, not corpus entries).

The correctly typed `eval` action has 100 gold suffix tokens, zero masked;
the final `return_result` has 65, zero masked. Each fits in one direct-SFT
window. The suffix receives respectively 389/663 and 474/873 of the window's
effective loss weight; system prompt weight is zero. The assistant still has
the full prompt as input. Thus the sampled action is neither absent nor
accidentally excluded by the failed-action mask. The two rendered full-mode
prefixes exactly equal live executor `_prompt_plan` token IDs at the assistant
boundary (5935 and 6180 tokens).

The program-header train prefixes have 1644 and 1889 tokens. They exactly
match the live executor when the natlang runtime omits its own generic tool
schemas in that mode, as `apply_prompt_rendering` and the runtime contract
require. The first diagnostic mistakenly passed the native generic tools as
if they were supplied by an outside client; the executor correctly preserved
those supplied schemas, creating an artificial 1053-token difference. That
diagnostic artifact was discarded and the source-level tool argument fixed.
Both supported rendering modes now show exact token-boundary parity on these
two rows. The shorter program header is a declared curriculum choice, while
the current task evaluation uses full rendering.

Teacher-forcing the exact suffix under the step-15000 student gives mean NLL
1.646 / 1.336 and argmax agreement 62.0% / 69.2% in full mode; the shorter
program-header view gives 1.614 / 1.297 and 61.0% / 69.2%. These two actions
alone do not estimate task success. They show the trained model can assign
substantial probability to correct actions under the actual full prompt even
though its sampled task rollouts often choose a different action. Continue the
bug dragnet at action choice, length/repetition, and transfer from the declared
program-header/full mix; do not treat this narrow audit as an approach verdict.

The same exact logits segmented by the already-rendered action show where this
case's difficulty lies. In full mode, the tool names and delimiters are
near-certain (tool-name NLL 0.01/0.03; argmax both 100%). Arguments are also
strong under gold history (NLL 0.34/0.01; argmax 93%/100%). The earlier
reasoning/reply spans are much less certain (NLL 2.38/2.41; argmax 45%/44%).
This is teacher forcing: a good score on later action tokens after gold
reasoning does not show the model reaches that action under its own reasoning.
Inspect the transition from self-generated reasoning to action in actual
rollouts and measure whether reasoned continuation, repeated tool use or the
decision to stop is the first divergence. Region boundaries are diagnostic
labels, not a modified training target or mask.

An eight-continuation read-only probe on these two train-only cases used the
live full prompt, exact step-15000 weights, temperature 0.6 and seeds
820100–820103 (`runs/qad-action-target-mask-audit-20261014-v1/sampled-report.json`).
All eight reached an ordinary stop with a tool call; this is neither runtime
execution nor a task-success evaluation. On decision 0002, all four emitted
the correct `return_result(status='success', value={status:'pending',
inspector:'zal'})`. On decision 0001, one called `eval` with `return permits;`
instead of reading the specific site; three wrote
`const permits = permits.current({"site":"skait quay"}); return permits;`.
The latter shadows the service binding in JavaScript and refers to it before
local initialization. The native prompt already included the runtime's
`invalid-binding` feedback for that exact failure seed, including a distinct
`permitsResult` name as the repair example; the student repeated the bad
binding in three sampled continuations anyway. Thus finding the right tool
name is insufficient here: code-argument composition and recovery from an
observed binding error are concrete targets. This recent recovery cohort was
not in step-15000 training, so add it to a declared future corpus and remeasure
before inferring that trained recovery instruction failed. There is no evidence
here that the error text itself needs changing. Four samples per decision do
not establish prevalence across task families.
