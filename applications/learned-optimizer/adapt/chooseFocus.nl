---
description: Choose the bindings whose adaptation should change now.
args:
  view: EvidenceView
  history: AdaptationState["history"]
returns: string[]
---
view.facts.failuresByBinding counts failed attempts by the binding where each first failed; view.items are the
attempts, demonstrations and earlier steps. history lists the revisions already tried in this episode.

1. Start from the bindings whose behavior must change for the program to pass its evidence: bindings with failures,
   most failures first, then bindings whose demonstrations show behavior the program does not have yet.
2. Leave out a binding whose last two revisions in history were rejected, unless it is the only binding with failures.
3. Keep at most three.
4. Return their names in alphabetical order. An empty list is a valid answer when nothing needs to change.
