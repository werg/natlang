---
description: Own one evidence assessment and append one reconciled effect before shrinking the frontier.
args:
  state: LedgerState
returns: LedgerState
---
1. If pending is empty, return state. Otherwise select its first EffectCandidate and remove it from a copy of
   pending. The measure decreases by exactly one independently of whether the evidence resolves the effect.
2. Call assessEvidence(candidate, state.index, state.correlation). Preserve the returned observation classes,
   possibleStatuses and conflicts. They are the evidentiary premise for this ledger entry.
3. Recover the candidate's business fields from its linked attempts and explicit receipt assertions. A receipt
   cannot silently change order, amount or kind; disagreements become conflicts with both witnesses.
4. If a candidate has an established committed effect, retain that effectId. If only an uncertain attempt exists,
   leave effectId null and retain ledgerId as the attempt reference. A possible future receipt name is metadata.
5. Construct Effect from the agreed fields and the assessment's possibleStatuses. If the only supported status
   is absent and the attempt was explicitly rejected, put its IDs in discardedAttemptIds instead of effects.
6. Append the effect or rejection to a copy of state. Combine its source witnesses without multiplying quantities.
   Previously appended effects are unchanged; this function owns only the selected candidate's ledger delta.
7. Before returning, ensure the new effect does not repeat an existing committed effect ID. If it does, merge the
   observation witnesses into that existing effect and report a candidate-construction conflict for audit.
8. Return the new state with the shorter pending list, effects, discardedAttemptIds and accumulated conflicts.
