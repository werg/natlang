---
args:
  state: State
  event: UiEvent
returns: Decision
---
Build a behavior collection and seek counterexamples. save stores TypeScript source text exporting main(input) and contract secondary, invalidating results. add supplies case ID target, JSON input text, JSON expected secondary. run_case evaluates the target case using actual child execution. Natlang should propose boundary cases from the contract, but exact host execution determines results. Remove redundant examples only with a justified request; do not label generated cases as reviewed training data.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
