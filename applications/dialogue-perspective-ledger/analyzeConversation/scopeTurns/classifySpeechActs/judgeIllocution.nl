---
description: Judge one locally framed illocution with source-defined adoption and condition rules.
args:
  unit: SpeechUnit
  node: QuoteNode
  turn: TurnRecord
  references: ReferenceBindings
  source: DialogueSource
returns: SpeechActMeaning
---
1. Capture the actual unit.focus, its original supporting quotes, actual owned substring, node speaker/time/mode, resolved references, explicit source
   speech rules and any authenticated tone notes. Exclude global expected query answers.
2. Construct and invoke a scoped NL SpeechActMeaning literal over this actual supplied unit. Its numbered local policy
   first identifies literal content, then speech force, then each explicitly asserted proposition with its own value.
3. Return performed true only for an act performed in its actual original event. A depicted quoted
   original has performed false while its kind may still be promise; missing force scope stays null.
   Separate actual promise from quotation of a promise. A reporting-only sentence can identify the
   original target and meaning without creating a new commitment by the reporter or intermediate archive.
4. A conditional promise names issuer, beneficiary, actual consequent predicates and each independent
   activation prerequisite. Return conditions with role activation and consequences with role fulfillment.
   Preserve strict/inclusive temporal scope, personally performed acknowledgment and exact brief version.
5. An original day-only collection promise has a source-bounded collection consequent, not merely a
   true statement about its due date. Preserve due hour null; the explicit day rule supplies timely scope.
6. A private wish to cancel is an intention act, not a communicated retraction. Return the declared
   intention proposition under its actual private speaker, without superseding the promise act.
   A communicated withdrawal of a named promise is retraction, separately from a factual correction.
   A correction or private note may explicitly endorse a proposition and target a previous act. Read
   the stated target rather than defeating every similar belief in every participant's ledger.
7. Signed irony framing can defeat literal-praise endorsement. Missing tone cannot certify sincerity
   or irony, and unknown pronoun attribution cannot be repaired by selecting the most likely actor.
8. Return per-proposition claims rather than one aggregate polarity: acknowledgment of one version
   and rejection of another, or a correction denying one time and affirming another, retain opposite
   values under their exact source IDs. Mentioning a future promised event does not assert it happened.
9. Return the actual meaning with nullable claim values/issuer, condition predicates, alternatives and
   nonempty exact quotes. Classification does not itself grant the recipient adoption or verified knowledge.
