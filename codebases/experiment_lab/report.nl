---
description: Interpret an exact experiment summary without hiding failures or pending semantic review.
args:
  question: Text
  plan: Plan
  candidates: Candidate[]
  trials: Trial[]
returns: Report
types:
  Plan: '{ selected: Text[], reason: Text }'
  Candidate: '{ id: Text, source_revision: Text, model_id: Text, model_seed: Num }'
  Trial: '{ candidate: Text, case_id: Text, attempt: Num, status: "done" | "quiesced" | "exception" | "missing", label: Text, provenance: Bool, quality: "pass" | "fail" | "pending", value_digest: Text, trace_id: Text }'
  Metric: '{ candidate: Text, planned: Num, done: Num, failed: Num, missing: Num, provenance_ok: Num, reviewed_pass: Num, reviewed_fail: Num, review_pending: Num, repeats_compared: Num, repeats_agree: Num }'
  Analysis: '{ interpretation: Text, unknowns: Text[], followups: Text[] }'
  Report: '{ plan: Plan, metrics: Metric[], trials: Trial[], interpretation: Text, unknowns: Text[], followups: Text[] }'
---
function report(question, plan, candidates, trials) -> Report
  metrics = summarize(candidates, trials)
  analysis = interpret(question, plan, metrics, trials)
  return assemble(plan, metrics, trials, analysis)
