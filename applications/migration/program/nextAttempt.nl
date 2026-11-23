---
description: After a migrated program failed its equivalence comparison, whether to write another candidate.
args:
  comparison: Comparison
  attemptsLeft: number
  history: string[]
returns: Next
readout: decision
---
comparison is how the migrated program did against the original on the same cases and seeds; its problems name the
cases it lost. history holds the problems of the earlier attempts, oldest first. Answer "retry" when another rewrite
told these problems could plausibly win the lost cases back: the losses are few, or differ from the earlier attempts'.
Answer "give-up" when attemptsLeft is 0, or when the same cases are lost again and again.
