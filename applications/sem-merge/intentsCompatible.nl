---
description: Judge whether two overlapping edits to the same base hunk have compatible intents that can both be preserved.
args:
  hunk: ConflictHunk
returns: boolean
readout: decision
---
Compare hunk.left and hunk.right against hunk.base for hunk.path.
Return true when the two edits pursue compatible intents that can be combined into one coherent block preserving both changes.
Return false when the two edits directly contradict each other, choose mutually exclusive behaviors or values for the same setting, or cannot both hold at once without silently dropping one side.
