import { check_candidate } from "./infer/check_candidate";
import { finalize } from "./infer/finalize";
import { propose } from "./infer/propose";
---
description: Propose a TS-style natlang signature from bounded source and caller evidence. Read files for source context when the bounded snapshot leaves a question open.
args:
  target: Target
  context: Context
returns: Assessment
effects: [types.check]
types:
  Target: '{ name: string, body: string, parameters: string[], revision: string }'
  Obligation: '{ kind: "argument" | "return", parameter: string, type: string, source: string }'
  Signature: '{ name: string, args: Record<string, string>, returns: string }'
  Witness: '{ id: string, callee: string, arg_types: Record<string, string>, source: string }'
  Context: '{ named_types: Record<string, string>, signatures: Signature[], obligations: Obligation[], required_effects: string[], witnesses: Witness[] }'
  Candidate: '{ args: Record<string, string>, returns: string, effects: string[], reason: string, alternatives: string[] }'
  Diagnostic: '{ level: "exact" | "hypothesis" | "unknown", source: string, message: string }'
  FitReport: '{ parseable: boolean, obligations_ok: boolean, checked: number, diagnostics: Diagnostic[] }'
  Assessment: '{ candidate: Candidate, fit: FitReport, status: "consistent" | "uncertain" | "invalid" }'
---
function infer(target, context) -> Assessment
  candidate = propose(target, context)
  fit = check_candidate(target, context, candidate)
  return finalize(candidate, fit)
