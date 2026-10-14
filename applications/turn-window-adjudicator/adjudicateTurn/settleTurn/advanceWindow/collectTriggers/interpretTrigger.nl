---
description: Join one trigger condition with resolved quantities and its captured or last-known controller basis.
args:
  rule: RuleReading
  bundle: EventBundle
  replacements: ReplacementDecision[]
  batch: BatchDelta
  beforeBoard: BoardSnapshot
  afterBoard: BoardSnapshot
  announcements: Announcement[]
returns: TriggerDecision
---
1. Gather this rule's actual source meaning, capacity witnesses, event scope and the resolved evidence
   relevant to its condition. Preserve original announcement ReferenceBindings separately from current
   board controllers and batch Departure last-known controllers.
2. Capture that context and the rule. In eval declare a local TriggerDecision shape. Construct a scoped
   NL literal over the actual bundle, replacement history and batch outcome; ask it to decide eligibility,
   qualifying part IDs, precise controller basis, recipient and source-authorized consequence parts.
3. Its semantic task distinguishes positive actual damage from announced or converted quantities,
   chooses a source-defined earliest qualifying printed part when required, and tests a conversion
   threshold against the quantity actually converted at that replacement step. It also reads token state.
4. Directly await the literal. Consume its actual eligible, recipient, consequenceParts and tokenIds.
   A captured recipient comes from the selected part's original reference binding; a last-known one
   comes from its actual departure; a current one comes from the rule's specified snapshot owner.
5. Validate nonempty exact witnesses against the supplied original evidence and require referenced
   qualifying parts/departures to belong to this batch. For a new-instance effect retain the source
   specified card/instance identities and initial fields; the semantic result must not refresh turn tokens.
6. An ineligible judgment returns no consequence parts or consumed token IDs and states its specific
   source reason. Missing decisive evidence returns null rather than a guessed recipient or event amount.
7. Bind capacityTokenIds from the actual owning rule.tokenIds, including ineligible decisions; these
   identify capacity separately from tokens consumed now. Return the checked TriggerDecision with
   source priority. The parent queues this value once;
   subsequent board changes cannot cause a second interpretation of its already captured recipient.
