---
description: Read the exact failure facts and decide where the program should change, which evidence shows each problem, and what behavior is missing.
args:
  view: EvidenceView
  history: AdaptationState["history"]
returns: InterpretedEvidence
---
view.facts.failuresByBinding counts failed attempts by the binding where each first failed; view.items are the
attempts, demonstrations and earlier steps, each with an id. history lists the revisions already tried in this episode
with their support loss.

1. focus: the bindings to revise now. Start from the bindings with failures, most failures first. Leave out a binding
   whose last two revisions in history were rejected, unless it is the only binding with failures. Keep at most three.
2. keyEvidence: for each focused binding, the ids of its failed attempts first, then the ids of demonstrations at
   that binding. Keep at most six ids per binding.
3. digests: for each focused binding, one short native span that describes the behavior the binding is missing,
   written from its failed attempts and demonstrations: what it should do that it does not do now.
4. Return { focus, keyEvidence, digests }. An empty focus is a valid answer when nothing failed.
