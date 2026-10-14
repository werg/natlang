---
description: Use a local NL judgment to bind one expression to its attributed original frame.
args:
  expression: string
  node: QuoteNode
  turn: TurnRecord
  source: DialogueSource
returns: DeicticMeaning
---
1. Capture expression, actual node attribution/time, its exact original substring, surrounding source
   context and source clock/reference rules. The outside report frame is available only as a distinct fact.
2. Construct a scoped NL literal returning DeicticMeaning and directly invoke it on the expression.
   It first classifies person/time/place, then identifies the owning frame and eligible source referents.
3. For tomorrow in an authenticated quotation use that original utterance day. For a diary that openly
   endorses a mistaken forwarding-day reading, describe that actual claimed reference without treating
   the diary belief as the source-correct temporal interpretation of the original promise.
4. Resolve a person only when context establishes one referent. Where source says two are eligible
   and no disambiguating intention is recorded, return both IDs, ambiguous status and the supporting quote.
5. Separate an original day anchor from a specified hour. Return a nullable original hour in resolvedAt
   when the source provides only a day; do not manufacture a deadline at an arbitrary clock boundary.
6. Copy the actual local judgment, including alternatives/reason and exact original quotes. The caller
   binds witness identity and arithmetic, while later belief stages determine who endorsed the meaning.
