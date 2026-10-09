---
description: One repair round of a migration. Triage the failing candidate, repair it, apply the patches (a rejection is data for the next round), run the checks, and return the next state.
args:
  state: RepairState
  intent: Intent
uses: [loop/triage, loop/repair]
returns: RepairState
---
Run one round on state and return the next state. Work in eval. The repository service applies and validates.

1. findings = await triage(intent, state.snapshot, state.validation, state.rejected). repairable = the findings
   whose repairable is true.
2. When repairable is empty, return { ...state, findings, remaining: state.remaining - 1 }.
3. patches = await repair(intent, state.snapshot, repairable). When patches is empty, return { ...state, findings,
   rejected: "the repair wrote no patch for the findings", remaining: state.remaining - 1 }.
4. Try candidate = repository.apply(state.snapshot.revision, patches), then validation = await
   repository.validate(candidate.revision), and return { snapshot: candidate, validation, rejected: "", findings,
   remaining: state.remaining - 1, revisions: [...state.revisions, candidate.revision] }.
5. When apply rejects, return { ...state, findings, rejected: the error's message, remaining: state.remaining - 1 }.
