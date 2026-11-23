---
description: What one step of a natural-language function's instructions asks the model to do.
args:
  step: string
  signature: string
  file: string
returns: "judgment" | "mechanism" | "guard-for-type" | "redundant"
readout: decision
---
step is one numbered step or paragraph of the instructions of file, a function with signature. Say what the step
asks the model to do:

- "judgment" when it weighs evidence, meaning, authority or intent and decides something a fixed rule cannot.
- "mechanism" when a program would do it the same way every time: picking or copying fields, set unions and
  membership by id, counting, sorting, arithmetic, assembling a record, or calling a fixed sequence of functions and
  binding their results.
- "guard-for-type" when it only restates what signature already guarantees, such as the allowed values of a result.
- "redundant" when it repeats another step or asks again for something already decided.
