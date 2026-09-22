---
description: Propose a TS-style natlang signature from bounded source and caller evidence. Read files for source context when the bounded snapshot leaves a question open.
args:
  target: Target
  context: Context
  files?: Dict<File>
returns: Assessment
effects: [types.check]
types:
  File: '{ kind: Text, text?: Text, bytes: Num }'
  Target: '{ name: Text, body: Text, parameters: Text[], revision: Text }'
  Obligation: '{ kind: "argument" | "return", parameter: Text, type: Text, source: Text }'
  Signature: '{ name: Text, args: Dict<Text>, returns: Text }'
  Witness: '{ id: Text, callee: Text, arg_types: Dict<Text>, source: Text }'
  Context: '{ named_types: Dict<Text>, signatures: Signature[], obligations: Obligation[], required_effects: Text[], witnesses: Witness[] }'
  Candidate: '{ args: Dict<Text>, returns: Text, effects: Text[], reason: Text, alternatives: Text[] }'
  Diagnostic: '{ level: "exact" | "hypothesis" | "unknown", source: Text, message: Text }'
  FitReport: '{ parseable: Bool, obligations_ok: Bool, checked: Num, diagnostics: Diagnostic[] }'
  Assessment: '{ candidate: Candidate, fit: FitReport, status: "consistent" | "uncertain" | "invalid" }'
---
function infer(target, context) -> Assessment
  candidate = propose(target, context)
  fit = check_candidate(target, context, candidate)
  return finalize(candidate, fit)
