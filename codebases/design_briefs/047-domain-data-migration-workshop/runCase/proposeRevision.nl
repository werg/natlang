---
args: { failure: string, target: string }
returns: RevisionProposal
---
Propose the smallest staged change that addresses the concrete failure. Name affected contracts, benefit, regression risk, and tests; do not alter authority.
Return the typed result, preserving missing facts as missing and citing evidence IDs where a decision depends on evidence.
