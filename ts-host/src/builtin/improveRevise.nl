---
description: Rewrite a program function's instructions so that its troubled recorded calls come out right (natlang improve DIR).
args:
  target: FunctionUnderReview
  evidence: RunEvidence
  diagnosis: string
returns: string
types:
  FunctionUnderReview: "{ path: string, declaration: string, instructions: string }"
  RunEvidence: "{ calls: number, failures: { inputs: unknown, outcome: string, reason: string }[], escalations: { inputs: unknown, local: unknown, fallback: unknown, agreed: boolean, reason: string }[], repairs: { inputs: unknown, problems: string[], outcome: string }[], feedback: { inputs: unknown, output: unknown, verdict: string, expected?: unknown }[], wrong: { inputs: unknown, expected: unknown, received?: unknown, error?: string }[] }"
---
target is one natural-language function of a program, evidence is what its recorded calls show, and diagnosis says which part of its instructions led those calls astray.

Return target's new instructions: the complete text that replaces target.instructions, without the declaration. Make the change the diagnosis calls for, so that the stronger model's answers, the testers' expected results and the examples' expected results would come out, and keep every instruction the evidence does not contradict. State a new rule by what the inputs mean, in general terms that cover the troubled calls and inputs like them; the calls in evidence are examples of the rule, not a list to copy into the instructions. Refer to the arguments by the names the declaration gives them. Keep the length of the instructions close to what they need.
