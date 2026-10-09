---
description: Merge verification for one intent. Decide whether a merged text carries what one change meant.
readout: decision
args:
  change: Change
  base: string
  merged: string
returns: Honor
---
Decide whether merged, the merged text of a block, carries change, one author's update to base, the block's text before
any update.

- honored: every statement in change.adds is in merged, in meaning, and every statement in change.removes is gone from
  merged or has been replaced by what another author's update puts there.
- partly: some of change.adds or change.removes holds in merged and some does not.
- lost: merged does not show the intent of change at all.

Judge by meaning and not by wording. change.intent says what the author wanted.
