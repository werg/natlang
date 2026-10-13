---
description: Judge and consume one replacement application against current parts and the fixed before-batch board.
args:
  state: ReplacementState
returns: ReplacementState
---
1. Take the first remaining application and locate its actual RuleReading and current parts with the
   named original lineage. Retain the immutable beforeBoard and the reservedTokenIds from prior steps.
   Remove this one application from the copied frontier regardless of applicability.
2. Gather source clauses for this rule, its condition's identity/controller basis, finite capacity and
   governing replacement timing. Capture those clauses, beforeBoard and reservedTokenIds. The varying
   literal input is the application and its actual current parts, including prior applied-rule history.
3. In eval declare a local ReplacementDecision shape matching the interface. Form a scoped NL literal
   that judges applicability from the rule's semantic scope, current part kind/quantity and correct
   snapshot, then returns rewritten parts and consumed token IDs only for an established application.
4. Directly await that literal. Its returned true, false or null is the actual decision. For true,
   use the returned afterParts as the next lineage contents and add its consumed shared tokens. For
   false, retain current parts and record the source-specific failed condition. For null, retain the
   missing decisive premise and stop further rewriting with an unresolved result.
5. Require nonempty exact original witnesses for every decided contribution. Bind provenance against
   supplied source clauses and check beforeParts equal the actual current lineage. Each rewritten
   part retains its original lineage and records this rule in appliedRuleIds; a converted effect kind
   governs later applicability. Invalid provenance or lineage integrity requires contribution repair.
6. Replace only this lineage's parts, preserving its printed order and every sibling part. Append the
   actual decision once; carry reserved tokens forward. Keep the common beforeBoard unchanged even
   when a hypothetical eventual damage outcome would destroy another creature.
7. Return the copied frontier with its smaller remainingApplications list, actual bundle, decisions,
   reservations and unresolved state. The parent consumes these fields; it must not regenerate a final
   rewrite history from the desired final counters or from original announced amounts.
