# QAD action-target mask and rendering dragnet (Pop, 2026-10-14)

The step-15000 QAD LFM2.5 student scored 4/48 on the protected task packet.
Ordinary-token/cache parity on captured prompts and sampled control-token
checks narrowed serving explanations. This diagnostic asks whether a correctly
admitted train-only action is actually supervised, rendered as the executor
will see it, and learnable under the exact step-15000 weights. It makes no
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
`runs/qad-action-target-mask-audit-20261014-v1/{cpu,gpu}-report.json` (local
diagnostic artifacts, not corpus entries).

The correctly typed `eval` action has 100 gold suffix tokens, zero masked;
the final `return_result` has 65, zero masked. Each fits in one direct-SFT
window. The suffix receives respectively 389/663 and 474/873 of the window's
effective loss weight; system prompt weight is zero. The assistant still has
the full prompt as input. Thus the sampled action is neither absent nor
accidentally excluded by the failed-action mask. The two rendered full-mode
prefixes exactly equal live executor `_prompt_plan` token IDs at the assistant
boundary (5935 and 6180 tokens).

The program-header train prefixes have 1644 and 1889 tokens; setting the same
executor to program-header mode produces 2697 and 2942 live prompt tokens.
Their first difference is token 1079, where the live prompt continues with a
tool-list description while the training system message ends and the user turn
starts. This is an exact template difference, not whitespace-only. Determine
whether the data curriculum intentionally uses a shorter program header while
task evaluation uses full mode; if so, quantify the transfer cost before any
recipe change. The full-mode match means the executor path itself is not
universally mismatched.

Teacher-forcing the exact suffix under the step-15000 student gives mean NLL
1.646 / 1.336 and argmax agreement 62.0% / 69.2% in full mode; the shorter
program-header view gives 1.614 / 1.297 and 61.0% / 69.2%. These two actions
alone do not estimate task success. They show the trained model can assign
substantial probability to correct actions under the actual full prompt even
though its sampled task rollouts often choose a different action. Continue the
bug dragnet at action choice, length/repetition, and the declaration of the
program-header/full mix; do not treat this narrow audit as an approach verdict.
