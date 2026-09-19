# Pending writes and confidence-triggered review

Implemented and tested on the v8 350M interpreter, 2026-09-19.

## Execution behavior

- Literal write proposals reach runtime validation by default. Native grammar
  still constrains call syntax and available paths; `write_constraints="typed"`
  remains available for comparisons.
- Extra `{value: X}` object unwrapping was removed. Real records with a `value`
  field remain valid when the destination expects them.
- Quote tolerance remains: a single JSON-text layer is parsed when the original
  string does not fit, and an unquoted scalar can be rendered as JSON text for a
  Text destination. Objects are never unwrapped or stringified to make them fit.
  Text that already fits is preserved, including intentional quote characters.
- Validation failures return to the caller by default. Prior effects remain;
  no automatic retry or rollback is added.

## Confidence and the review fork

Native decoding retains raw selected-token logprobs and token bytes. AST literal
spans identify `write.value` and `edit.new`. Token byte spans are aligned to the
actual generated text, including UTF-8 and multi-token literals. A token crossing
a value boundary is counted whole. Scores include geometric-mean and minimum
raw token probability. No exact alignment or missing logprobs means unknown,
not zero confidence. Source-copy writes have no generated value score.

`careful_threshold` is opt-in. If a value's geometric-mean score is below it:

1. Keep the entire proposed batch pending. No operation in it has executed.
2. Fork the original messages from immediately before the proposal, appending
   the proposal as quoted JSON in an "Are you sure?" check against instructions
   and available evidence. The numerical score is not shown to the reviewer.
3. Constrain the response to exactly one `review_write` call with a free-text
   reason and a decision from `approve`, `error`, `blocker`. Reason comes first
   by default; `--review-order decision_first` reproduces the alternative.
4. Approval releases the **original, unchanged** batch to normal runtime
   validation. The review cannot substitute another value. An error/blocker
   quiesces the lambda and returns its diagnostic to the caller. Malformed
   responses and exhausted budgets leave the proposal unapplied.
5. Resume the original agent conversation after successful execution. Review
   messages are saved separately and never inserted into that conversation.

This is a fork of conversational context using the same model, not a second
model or rollback of already-executed actions. Reviews consume the same turn,
token, and wall-clock budgets as the episode. They run at most once per flagged
proposal call. All review decisions precede all effects in that proposed batch.

## Measurements

The natural validation probe has three valid controls and six impossible tasks.
All runs below use caller feedback and runtime write typing.

| Configuration | Valid controls correct | Impossible tasks accepted | Review calls |
| --- | ---: | ---: | ---: |
| No careful check | 3/3 | 4/6 | 0 |
| Threshold 0.5, decision first | 3/3 | 4/6 | 3 |
| Threshold 0.5, reason first | 3/3 | 1/6 | 3 |
| Threshold 0.994690001059068, reason first | 3/3 | 0/6 | 5 |

Decision-first checks approved all three low-confidence wrong proposals. Two
then gave reasons describing a conflict. Reversing the generation order allowed
those reasons to precede and inform the verdict. The wrong conditional value
had high raw confidence (~0.978), so threshold 0.5 did not trigger on it.
Reasons are sometimes imprecise, and missing information may be called an error
rather than a blocker. These counts do not certify diagnostic explanations.

### Threshold calibration on separate instances

`scripts/confidence_probe.py` ran 120 programs: ten fresh groups (seed 992) of
the twelve existing matched templates. Those templates are familiar from
training, so this checks new instances rather than template-level transfer.
Only complete direct `return` writes have automatic semantic labels. A shadow
runtime validates and quote-normalizes each proposed value before comparing it
with the fixture oracle. Type/shape errors remain a separate class; partial
writes, locals, edits, and source copies are unscored. "Semantic mismatch" is an
observable output label, not an inference about deceptive intent.

Baseline: 106/120 correct executions. Of 70 scored values, 56 were correct and
14 were semantic mismatches; 11 of those mismatches had scores above 0.95.
The raw-score >0.95 bin averaged 0.996 but was only 83.6% correct. Raw likelihood
is therefore not a calibrated correctness probability here.

The script fits a **review-trigger threshold**, maximizing balanced accuracy,
on even-numbered groups and evaluates it on odd-numbered groups. Ties prefer
fewer reviews. It does not fit a universal probability calibrator.

| Partition | Wrong values flagged | Correct values flagged |
| --- | ---: | ---: |
| Calibration, even groups | 6/6 | 2/29 |
| Held-out instances, odd groups | 7/8 | 1/27 |

Selected threshold: **0.994690001059068**. Running careful mode with that
threshold on the 60 odd-group programs produced 53/60 correct versus baseline
52/60, with eleven reviews. Four incorrect conditional executions were rejected
correctly, but one correct bounded-number answer was mistakenly rejected. Two
binding-error outcomes also regressed in runs with no review, illustrating
batch/cache/numerical output variability; they cannot be attributed to review
verdicts. This is mixed evidence for overall task success, despite the strong
result on the smaller natural-language diagnostic set.

Keep careful mode experimental. New free-form text, models, prompts, or task
families may need different thresholds. High confidence can accompany fabricated
values, and self-review can confidently reject a valid result. Measure both
misses and false rejections, along with the added work.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/confidence_probe.py --groups 10 --seed 992 --out runs/confidence.json
.venv/bin/python scripts/confidence_probe.py --groups 10 --seed 992 --partition test --careful-threshold 0.994690001059068 --out runs/confidence-careful.json
.venv/bin/python scripts/validation_probe.py --policies caller --careful-threshold 0.994690001059068 --out runs/validation-careful.json
```

Observed runs: `runs/v8-confidence-familiar.json`,
`runs/v8-confidence-heldout-careful.json`,
`runs/student-v8-careful-{baseline,half,reason-first,calibrated}.json`.
The older `half` artifact predates the review-order metadata field and used
decision-first. Proposed values and review conversations are recorded separately.

Tests cover quote coercion, wrapper rejection, real `{value: Num}` records,
UTF-8 probability alignment, unknown scores, review budgets, guided choices,
unchanged-value approval, conversation isolation, and preventing the complete
batch's effects on refusal. The threshold test verifies that held-out labels
cannot influence the selected threshold.
