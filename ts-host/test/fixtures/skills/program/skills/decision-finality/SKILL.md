---
name: decision-finality
description: Decides whether a note states a current, final decision, respecting withdrawals, negation and superseded earlier states. Use when selecting records by the status their notes finally give.
natlang:
  scope:
    finalityRubric:
      type: "{ cue: string, meaning: 'final' | 'superseded' | 'withdrawn' | 'proposed' }[]"
      file: knowledge/rubric.json
      description: cue phrases and what they mean for finality
  tests:
    - name: withdrawal overrides earlier confirmation
      input: "My attendance confirmation is withdrawn; I will not attend."
      expected: false
    - name: later confirmation supersedes refusal
      input: "The earlier decline is superseded: I now confirm attendance."
      expected: true
  provenance:
    author: fixture
    support_groups: ["fixture:decision-finality:support"]
    families: [active-urgency, explicit-consent, completed-delivery, active-license, final-cancellation,
      overall-recommendation, authorized-access, resolved-support, confirmed-attendance, final-renewal]
---

Read the whole note before deciding. The last explicit decision wins:

1. A later statement that supersedes, withdraws or revokes an earlier one replaces it.
2. Proposals, requests, predictions and suggestions are not decisions.
3. A negated decision ("not granted", "declined") is a decision against.

See `examples/` for worked cases, and use `finalityRubric` for cue phrases.
