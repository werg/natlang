import { assemble } from "./check/assemble";
import { check_calls } from "./check/check_calls";
import { identify_calls } from "./check/identify_calls";
---
description: Explore possible calls in a natlang body and separate witnessed mismatches from hypotheses. Read files for source context when the bounded snapshot leaves a question open.
args:
  target: Target
  context: Context
returns: CheckReport
effects: [types.calls]
types:
  Target: '{ name: string, body: string, parameters: string[], revision: string }'
  Obligation: '{ kind: "argument" | "return", parameter: string, type: string, source: string }'
  Signature: '{ name: string, args: Record<string, string>, returns: string }'
  Witness: '{ id: string, callee: string, arg_types: Record<string, string>, source: string }'
  Context: '{ named_types: Record<string, string>, signatures: Signature[], obligations: Obligation[], required_effects: string[], witnesses: Witness[] }'
  CallClaim: '{ callee: string, arg_types: Record<string, string>, evidence_id: string, rationale: string }'
  Diagnostic: '{ level: "exact" | "hypothesis" | "unknown", source: string, message: string }'
  CheckReport: '{ claims: CallClaim[], diagnostics: Diagnostic[] }'
---
function check(target, context) -> CheckReport
  claims = identify_calls(target, context)
  diagnostics = check_calls(context, claims)
  return assemble(claims, diagnostics)
