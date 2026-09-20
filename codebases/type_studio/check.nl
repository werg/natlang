---
description: Explore possible calls in a natlang body and separate witnessed mismatches from hypotheses.
args:
  target: Target
  context: Context
returns: CheckReport
effects: [types.calls]
types:
  Target: '{ name: Text, body: Text, parameters: Text[], revision: Text }'
  Obligation: '{ kind: "argument" | "return", parameter: Text, type: Text, source: Text }'
  Signature: '{ name: Text, args: Dict<Text>, returns: Text }'
  Witness: '{ id: Text, callee: Text, arg_types: Dict<Text>, source: Text }'
  Context: '{ named_types: Dict<Text>, signatures: Signature[], obligations: Obligation[], required_effects: Text[], witnesses: Witness[] }'
  CallClaim: '{ callee: Text, arg_types: Dict<Text>, evidence_id: Text, rationale: Text }'
  Diagnostic: '{ level: "exact" | "hypothesis" | "unknown", source: Text, message: Text }'
  CheckReport: '{ claims: CallClaim[], diagnostics: Diagnostic[] }'
---
function check(target, context) -> CheckReport
  claims = identify_calls(target, context)
  diagnostics = check_calls(context, claims)
  return assemble(claims, diagnostics)
