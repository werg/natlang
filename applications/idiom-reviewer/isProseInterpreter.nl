---
description: Decide whether a natural-language function as a whole is a prose interpreter of plumbing.
args:
  file: string
  steps: KindedStep[]
  hasHelpers: boolean
returns: boolean
readout: decision
---
steps are the instructions of file, one entry per numbered step or paragraph, each with what it asks the model to
do. hasHelpers says whether file already has TypeScript helpers in its callable folder.

Return true when the function as a whole is an algorithm written in English: the model is asked to carry out
mechanical steps one after another, and the few judgments in it would stand on their own as small typed functions
called from TypeScript. Return false when the function is mainly one judgment, even if a step or two of mechanism
surrounds it.
