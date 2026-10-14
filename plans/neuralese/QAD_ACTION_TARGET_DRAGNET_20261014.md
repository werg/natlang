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
