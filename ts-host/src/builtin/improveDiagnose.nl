---
description: Say what a program function's recorded runs show its instructions get wrong (natlang improve DIR).
args:
  target: FunctionUnderReview
  evidence: RunEvidence
returns: string
types:
  FunctionUnderReview: "{ path: string, declaration: string, instructions: string }"
  RunEvidence: "{ calls: number, failures: { inputs: unknown, outcome: string, reason: string }[], escalations: { inputs: unknown, local: unknown, fallback: unknown, agreed: boolean, reason: string }[], repairs: { inputs: unknown, problems: string[], outcome: string }[], feedback: { inputs: unknown, output: unknown, verdict: string, expected?: unknown }[], wrong: { inputs: unknown, expected: unknown, received?: unknown, error?: string }[] }"
---
target is one natural-language function of a program: its file, its declaration (front matter) and its instructions. evidence is what the function's own recorded calls show: calls that failed with their reason; decisions that escalated to a stronger model, with the local answer and the stronger model's answer; results that broke a postcondition and had to be revised; calls a tester marked wrong, with the expected result when given; and saved examples the current instructions answer wrongly.

Find what the troubled calls have in common: a kind of input, a case the instructions do not mention, a rule they state ambiguously, or a step that leads the function astray. Treat the stronger model's answer, a tester's expected result and an example's expected result as right.

Answer in two to four sentences: the shared feature of those inputs, and which instruction misleads or what the instructions leave unsaid. Answer "nothing to fix" when the troubled calls share no cause in the instructions.
