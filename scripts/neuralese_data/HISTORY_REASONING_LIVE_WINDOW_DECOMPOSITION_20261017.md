# History-reasoning live window

This runner compares the pinned checkpoint's current history renderer with the
explicit KEEP renderer on the same source-bound, train-only task contexts. The
runner does not update weights or turn its outputs into training data.

| Part | Representation | Reason |
| --- | --- | --- |
| Case eligibility and source binding | Crisp verifier | Exact split, identity, profile-presence and file-hash checks must fail closed. |
| Profile selection | CLI setting plus checkpoint default | The baseline is the checkpoint's actual saved or backbone-default policy; KEEP is the declared candidate. |
| Prompt construction and generation | Shared `Engine` and `bind_history_reasoning` | This is the production renderer and decoder, with only the template profile changed. |
| Target likelihood and coverage | Shared native forward plus crisp token accounting | Exact token IDs, target scores and earlier-reasoning presence are measured without changing model parameters. |
| Durable outputs | Atomic report plus append-only JSONL | Preserve checkpoint, tokenizer, case, profile and per-sample evidence across interruption. |

The input case packet binds each row to a source file and row ID, labels every
row as train-owned, and contains a complete prior conversation plus its actual
next assistant target. The same 48 cases run in both variants. Each case must
contain at least one earlier assistant reasoning field; otherwise it cannot
measure this profile change. The runner records generated responses, target
log-likelihood, rendered prompt IDs and whether earlier reasoning was present.

The runner is prepared for the next safe executor window. Adoption still
requires the actual 48 paired samples and independent review; checkpoint and
recipe state remain unchanged during the comparison.
