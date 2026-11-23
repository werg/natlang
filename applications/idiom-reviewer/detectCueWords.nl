---
description: Judge whether a .nl instruction relies on brittle keyword or cue-word shortcuts instead of meaning.
args:
  body: string
returns: boolean
readout: decision
---
Decide whether body specifies a brittle keyword or cue-word lookup rule (such as mapping fixed trigger words or exact phrases directly to labels) rather than genuine semantic criteria. Return true only when a trivial keyword matcher would implement the instruction.
