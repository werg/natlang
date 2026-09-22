import { assemble } from "./report/assemble";
import { interpret } from "./report/interpret";
import { summarize } from "./report/summarize";
---
description: Interpret an exact experiment summary without hiding failures or pending semantic review.
args:
  question: string
  plan: Plan
  candidates: Candidate[]
  trials: Trial[]
returns: Report
types:
  Plan: '{ selected: string[], reason: string }'
  Candidate: '{ id: string, source_revision: string, model_id: string, model_seed: number }'
  Trial: '{ candidate: string, case_id: string, attempt: number, status: "done" | "quiesced" | "exception" | "missing", label: string, provenance: boolean, quality: "pass" | "fail" | "pending", value_digest: string, trace_id: string }'
  Metric: '{ candidate: string, planned: number, done: number, failed: number, missing: number, provenance_ok: number, reviewed_pass: number, reviewed_fail: number, review_pending: number, repeats_compared: number, repeats_agree: number }'
  Analysis: '{ interpretation: string, unknowns: string[], followups: string[] }'
  Report: '{ plan: Plan, metrics: Metric[], trials: Trial[], interpretation: string, unknowns: string[], followups: string[] }'
---
function report(question, plan, candidates, trials) -> Report
  metrics = summarize(candidates, trials)
  analysis = interpret(question, plan, metrics, trials)
  return assemble(plan, metrics, trials, analysis)
