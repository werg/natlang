# Semantic follow-up probe: sample problems

These are 16 representative cases from the [96-case IR corpus](../data/teacher/semantic-followup-probe.ir.jsonl). The agent sees the instructions, typed inputs, and imported function signatures. It obtains the report or investigation note only by calling the imported lookup function (or reading its source). `expected` and `generation` metadata are oracle data and are not shown to the agent.

## Pilot outcome

On 2026-09-23, Bonsai accepted all 10 sampled report cases and all 6 sampled composite cases after `eval` gained console observations and local helper support. The composite cases used 3–6 `eval` calls: computation, note retrieval, and a subsequent semantic decision. This is a targeted pilot, not a result for all 96 cases. The first composite attempt failed 0/6 because the evaluator rejected the model's `console.log` calls.

## semantic-followup:incident-00

**Family:** `semantic_followup_incident`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Route a confirmed exposure or unauthorized access to security; resource exhaustion to capacity; and a failure caused by another service to dependency. A ruled-out possibility is not the cause. The categories are security, capacity, dependency.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "incident-00"
}
```

**Observation returned by the lookup helper**

> The checkout API is responsive. Audit logs show a token belonging to another tenant read private invoices; the CPU alarm was a stale dashboard.

**Expected typed result**

```json
{
  "case_id": "incident-00",
  "category": "security"
}
```

## semantic-followup:incident-05

**Family:** `semantic_followup_incident`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Route a confirmed exposure or unauthorized access to security; resource exhaustion to capacity; and a failure caused by another service to dependency. A ruled-out possibility is not the cause. The categories are security, capacity, dependency.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "incident-05"
}
```

**Observation returned by the lookup helper**

> The status page says degraded capacity, yet local workers are idle and the external payment gateway returns 503 for each charge.

**Expected typed result**

```json
{
  "case_id": "incident-05",
  "category": "dependency"
}
```

## semantic-followup:customer-00

**Family:** `semantic_followup_customer`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Choose refund when the customer clearly asks to cancel or get money back; replacement when they want the purchased item fulfilled again; clarify when intent or eligibility is genuinely unresolved. Do not treat a quoted or rejected option as their request. The categories are refund, replacement, clarify.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "customer-00"
}
```

**Observation returned by the lookup helper**

> The box arrived damaged. I do not want another unit; please return the payment to my card.

**Expected typed result**

```json
{
  "case_id": "customer-00",
  "category": "refund"
}
```

## semantic-followup:customer-05

**Family:** `semantic_followup_customer`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Choose refund when the customer clearly asks to cancel or get money back; replacement when they want the purchased item fulfilled again; clarify when intent or eligibility is genuinely unresolved. Do not treat a quoted or rejected option as their request. The categories are refund, replacement, clarify.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "customer-05"
}
```

**Observation returned by the lookup helper**

> I wrote 'refund' in the subject because that was the menu choice. I am still deciding whether to keep the item.

**Expected typed result**

```json
{
  "case_id": "customer-05",
  "category": "clarify"
}
```

## semantic-followup:evidence-00

**Family:** `semantic_followup_evidence`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Judge whether the observation supports, contradicts, or leaves the claim unresolved. Match the population and time period; a related metric or an unrepresentative sample is not proof. The categories are supports, contradicts, inconclusive.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "evidence-00"
}
```

**Observation returned by the lookup helper**

> Claim: the new cache reduces median lookup latency. A controlled comparison on the same workload measured 82 ms before and 51 ms after, with unchanged hardware.

**Expected typed result**

```json
{
  "case_id": "evidence-00",
  "category": "supports"
}
```

## semantic-followup:evidence-05

**Family:** `semantic_followup_evidence`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Judge whether the observation supports, contradicts, or leaves the claim unresolved. Match the population and time period; a related metric or an unrepresentative sample is not proof. The categories are supports, contradicts, inconclusive.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "evidence-05"
}
```

**Observation returned by the lookup helper**

> Claim: the endpoint is faster for mobile users. Desktop timings improved, but no mobile requests were measured.

**Expected typed result**

```json
{
  "case_id": "evidence-05",
  "category": "inconclusive"
}
```

## semantic-followup:code_diagnosis-00

**Family:** `semantic_followup_code_diagnosis`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Classify the most direct cause: implementation logic is wrong, caller and callee disagree on an interface contract, or the test expects behavior contrary to the stated specification. Do not infer from an error name alone. The categories are implementation, contract, test_expectation.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "code_diagnosis-00"
}
```

**Observation returned by the lookup helper**

> Spec: keep first occurrence order. The function constructs a Set, then sorts its values alphabetically; the failing test expects original order.

**Expected typed result**

```json
{
  "case_id": "code_diagnosis-00",
  "category": "implementation"
}
```

## semantic-followup:code_diagnosis-05

**Family:** `semantic_followup_code_diagnosis`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Classify the most direct cause: implementation logic is wrong, caller and callee disagree on an interface contract, or the test expects behavior contrary to the stated specification. Do not infer from an error name alone. The categories are implementation, contract, test_expectation.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "code_diagnosis-05"
}
```

**Observation returned by the lookup helper**

> Spec requires stable sorting for equal scores. The implementation preserves input order; the test expects equal-score ids alphabetized.

**Expected typed result**

```json
{
  "case_id": "code_diagnosis-05",
  "category": "test_expectation"
}
```

## semantic-followup:deployment-00

**Family:** `semantic_followup_deployment`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Proceed when evidence supports continuing a safe rollout, hold when evidence is incomplete or a pre-launch blocker remains, and roll back when the active rollout causes a material regression. Distinguish a current regression from an unrelated historical alert. The categories are proceed, hold, rollback.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "deployment-00"
}
```

**Observation returned by the lookup helper**

> The canary has served representative traffic for two hours. Error rate and latency match baseline, and the required migration check passed.

**Expected typed result**

```json
{
  "case_id": "deployment-00",
  "category": "proceed"
}
```

## semantic-followup:deployment-05

**Family:** `semantic_followup_deployment`

**Agent instructions**

```text
Get the case report with lookup_report(case_id) and inspect what it says.
Interpret the whole report using this policy: Proceed when evidence supports continuing a safe rollout, hold when evidence is incomplete or a pre-launch blocker remains, and roll back when the active rollout causes a material regression. Distinguish a current regression from an unrelated historical alert. The categories are proceed, hold, rollback.
Record the category you chose with record_decision and return its Decision.
```

**Typed inputs**

```json
{
  "case_id": "deployment-05"
}
```

**Observation returned by the lookup helper**

> The rollout changed the authorization layer. New canary requests can read another tenant's records; controls cannot.

**Expected typed result**

```json
{
  "case_id": "deployment-05",
  "category": "rollback"
}
```

## semantic-followup:rollout-00

**Family:** `semantic_followup_rollout`

**Agent instructions**

```text
Compute the control before/after averages and the focus before/after averages.
Compare their changes to infer the adjusted trend; treat an adjusted difference with magnitude at most 2 as stable or neutral.
Get the investigation note with lookup_context(case_id) and interpret its meaning.
Combine the computed trend and the note: Compare the change in canary error rate with the change in control. A material canary-only regression linked to this release calls for rollback; incomplete attribution or an outstanding gate calls for hold; stable or improving evidence with gates complete supports proceeding. Use these exact labels: trend = improved, regressed, stable; context = blocker, linked, uncertain, unrelated; conclusion = hold, proceed, rollback.
Record the trend, context finding, and conclusion with record_assessment; return its Assessment.
```

**Typed inputs**

```json
{
  "case_id": "rollout-00",
  "control_before": [
    1,
    1,
    2
  ],
  "control_after": [
    1,
    2,
    1
  ],
  "focus_before": [
    1,
    2,
    1
  ],
  "focus_after": [
    9,
    10,
    8
  ]
}
```

**Observation returned by the lookup helper**

> The new parser is the only canary change. Replaying the old request through it reproduces the errors; reverting it stops them.

**Expected typed result**

```json
{
  "case_id": "rollout-00",
  "trend": "regressed",
  "context": "linked",
  "conclusion": "rollback"
}
```

## semantic-followup:rollout-03

**Family:** `semantic_followup_rollout`

**Agent instructions**

```text
Compute the control before/after averages and the focus before/after averages.
Compare their changes to infer the adjusted trend; treat an adjusted difference with magnitude at most 2 as stable or neutral.
Get the investigation note with lookup_context(case_id) and interpret its meaning.
Combine the computed trend and the note: Compare the change in canary error rate with the change in control. A material canary-only regression linked to this release calls for rollback; incomplete attribution or an outstanding gate calls for hold; stable or improving evidence with gates complete supports proceeding. Use these exact labels: trend = improved, regressed, stable; context = blocker, linked, uncertain, unrelated; conclusion = hold, proceed, rollback.
Record the trend, context finding, and conclusion with record_assessment; return its Assessment.
```

**Typed inputs**

```json
{
  "case_id": "rollout-03",
  "control_before": [
    1,
    2,
    1
  ],
  "control_after": [
    1,
    1,
    2
  ],
  "focus_before": [
    2,
    1,
    1
  ],
  "focus_after": [
    7,
    8,
    7
  ]
}
```

**Observation returned by the lookup helper**

> A new failure is visible, but the trace IDs were lost and no one can yet attribute it to this release or the upstream.

**Expected typed result**

```json
{
  "case_id": "rollout-03",
  "trend": "regressed",
  "context": "uncertain",
  "conclusion": "hold"
}
```

## semantic-followup:study-00

**Family:** `semantic_followup_study`

**Agent instructions**

```text
Compute the control before/after averages and the focus before/after averages.
Compare their changes to infer the adjusted trend; treat an adjusted difference with magnitude at most 2 as stable or neutral.
Get the investigation note with lookup_context(case_id) and interpret its meaning.
Combine the computed trend and the note: Compare treatment change with control change, then judge whether the study note makes that comparison credible. A credible favorable effect supports the claim, a credible adverse effect contradicts it, and a confounded or unmeasured effect is inconclusive. Use these exact labels: trend = negative, neutral, positive; context = confounded, credible, unknown; conclusion = contradicts, inconclusive, supports.
Record the trend, context finding, and conclusion with record_assessment; return its Assessment.
```

**Typed inputs**

```json
{
  "case_id": "study-00",
  "control_before": [
    40,
    41,
    39
  ],
  "control_after": [
    41,
    40,
    41
  ],
  "focus_before": [
    40,
    39,
    41
  ],
  "focus_after": [
    50,
    51,
    49
  ]
}
```

**Observation returned by the lookup helper**

> Assignment was randomized, groups used the same measurement window, and the outcome was specified before enrollment.

**Expected typed result**

```json
{
  "case_id": "study-00",
  "trend": "positive",
  "context": "credible",
  "conclusion": "supports"
}
```

## semantic-followup:study-03

**Family:** `semantic_followup_study`

**Agent instructions**

```text
Compute the control before/after averages and the focus before/after averages.
Compare their changes to infer the adjusted trend; treat an adjusted difference with magnitude at most 2 as stable or neutral.
Get the investigation note with lookup_context(case_id) and interpret its meaning.
Combine the computed trend and the note: Compare treatment change with control change, then judge whether the study note makes that comparison credible. A credible favorable effect supports the claim, a credible adverse effect contradicts it, and a confounded or unmeasured effect is inconclusive. Use these exact labels: trend = negative, neutral, positive; context = confounded, credible, unknown; conclusion = contradicts, inconclusive, supports.
Record the trend, context finding, and conclusion with record_assessment; return its Assessment.
```

**Typed inputs**

```json
{
  "case_id": "study-03",
  "control_before": [
    40,
    39,
    41
  ],
  "control_after": [
    44,
    45,
    43
  ],
  "focus_before": [
    40,
    41,
    39
  ],
  "focus_after": [
    44,
    43,
    45
  ]
}
```

**Observation returned by the lookup helper**

> Both groups faced the same seasonal demand shift. The outcome was recorded for every assigned user.

**Expected typed result**

```json
{
  "case_id": "study-03",
  "trend": "neutral",
  "context": "credible",
  "conclusion": "inconclusive"
}
```

## semantic-followup:fault_locality-00

**Family:** `semantic_followup_fault_locality`

**Agent instructions**

```text
Compute the before/after averages separately for upstream failures and local failures.
Determine whether each average increased by more than 2; record the combined pattern as upstream_only, local_only, both, or neither.
Get the investigation note with lookup_context(case_id) and interpret its meaning.
Combine the computed trend and the note: Compute how local and upstream failure counts changed, then use trace evidence to identify the fault. A simultaneous increase alone does not prove which system caused it. Choose local, upstream, or unresolved. Use these exact labels: trend = both, local_only, neither, upstream_only; context = ambiguous, points_local, points_upstream; conclusion = local, unresolved, upstream.
Record the trend, context finding, and conclusion with record_assessment; return its Assessment.
```

**Typed inputs**

```json
{
  "case_id": "fault_locality-00",
  "upstream_before": [
    1,
    2,
    1
  ],
  "upstream_after": [
    1,
    1,
    2
  ],
  "local_before": [
    1,
    1,
    2
  ],
  "local_after": [
    9,
    8,
    10
  ]
}
```

**Observation returned by the lookup helper**

> Local stack traces point to the new decoder before any upstream request is sent.

**Expected typed result**

```json
{
  "case_id": "fault_locality-00",
  "trend": "local_only",
  "context": "points_local",
  "conclusion": "local"
}
```

## semantic-followup:fault_locality-05

**Family:** `semantic_followup_fault_locality`

**Agent instructions**

```text
Compute the before/after averages separately for upstream failures and local failures.
Determine whether each average increased by more than 2; record the combined pattern as upstream_only, local_only, both, or neither.
Get the investigation note with lookup_context(case_id) and interpret its meaning.
Combine the computed trend and the note: Compute how local and upstream failure counts changed, then use trace evidence to identify the fault. A simultaneous increase alone does not prove which system caused it. Choose local, upstream, or unresolved. Use these exact labels: trend = both, local_only, neither, upstream_only; context = ambiguous, points_local, points_upstream; conclusion = local, unresolved, upstream.
Record the trend, context finding, and conclusion with record_assessment; return its Assessment.
```

**Typed inputs**

```json
{
  "case_id": "fault_locality-05",
  "upstream_before": [
    1,
    1,
    2
  ],
  "upstream_after": [
    2,
    1,
    1
  ],
  "local_before": [
    1,
    2,
    1
  ],
  "local_after": [
    2,
    1,
    1
  ]
}
```

**Observation returned by the lookup helper**

> A customer supplied a screenshot from last month; there is no current reproduced failure.

**Expected typed result**

```json
{
  "case_id": "fault_locality-05",
  "trend": "neither",
  "context": "ambiguous",
  "conclusion": "unresolved"
}
```
