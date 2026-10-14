---
description: Bind a rule to its governing event version and source-defined precedence.
args:
  rule: RuleReading
  rules: RuleIndex
  clauses: SourceClause[]
returns: TimingBinding
---
1. Read this rule's actual returned kind, condition, consequence and timing together with its source
   timing/identity/capacity contexts from rules.contexts. Select relevant context by its returned role
   and actual source links, retaining its meaning and witnesses. A damage noun alone does not decide whether the condition observes proposed,
   replaced or resolved damage; use the complete source relation.
2. Identify the phase in which this rule can act. An input announcement captures state; a submitted
   response changes the resumed board; a replacement rewrites a pending part; an after-resolution
   trigger reads actual effects. Keep a context-only definition outside event scheduling.
3. Bind readsSnapshot to the precise source basis. A current-controller reference during a simultaneous
   batch reads the common resumed before-state; a departing-instance reference reads last-known state.
   Neither selects the captured controller unless the rule expressly requests announcement-time capture.
4. Resolve explicit named before/after relationships against the actual RuleIndex. Preserve their
   direction. A replacement that removes a damage event can prevent another damage-only rule from
   applying, so an explicit order may change the result and cannot be treated as cosmetic sorting.
5. Read a printed trigger priority only when the source defines its scheduling meaning. Leave absent
   priority null; independent rules need not acquire a fabricated total priority to make an array sortable.
6. Return ruleId, phase, readsSnapshot, beforeRuleIds, afterRuleIds, priority, exact witnesses and
   unresolved. A missing snapshot basis is an unresolved source role, not permission to read whichever
   state makes the intended final answer convenient.
