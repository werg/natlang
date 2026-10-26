---
args: { input: CaseInput }
returns: PolicySelection
---
Compare policy candidates by authority, effective period, scope, and explicit supersession. Never use upload order to resolve precedence; retain unresolved precedence.
Return the typed result, preserving missing facts as missing and citing evidence IDs where a decision depends on evidence.
