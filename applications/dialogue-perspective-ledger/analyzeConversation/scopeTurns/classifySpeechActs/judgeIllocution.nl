---
description: Judge one locally framed illocution with source-defined adoption and condition rules.
args:
  node: QuoteNode
  turn: TurnRecord
  references: ReferenceBindings
  source: DialogueSource
returns: SpeechActMeaning
---
1. Capture the actual owned substring, node speaker/time/mode, resolved references, explicit source
   speech rules and any authenticated tone notes. Exclude global expected query answers.
2. Construct and invoke a scoped NL SpeechActMeaning literal over this unit. Its numbered local policy
   first identifies literal content, then speech force, then any explicitly endorsed proposition.
3. Return performed true only for an act performed in its actual original event. A depicted quoted
   original has performed false while its kind may still be promise; missing force scope stays null.
   Separate actual promise from quotation of a promise. A reporting-only sentence can identify the
   original target and meaning without creating a new commitment by the reporter or intermediate archive.
4. A conditional promise names issuer, beneficiary, consequent and each independent prerequisite.
   Preserve strict/inclusive temporal scope, personally performed acknowledgment and exact brief version.
5. A correction or private note may explicitly endorse a proposition and target a previous act. Read
   the stated target rather than defeating every similar belief in every participant's ledger.
6. Signed irony framing can defeat literal-praise endorsement. Missing tone cannot certify sincerity
   or irony, and unknown pronoun attribution cannot be repaired by selecting the most likely actor.
7. Return the actual meaning with nullable polarity/issuer, condition predicates, alternatives and
   nonempty exact quotes. Classification does not itself grant the recipient adoption or verified knowledge.
