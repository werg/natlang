---
description: Judge one dialogue conjunction under captured information access.
args:
  event: EventView
  actor: string
  factIds: string[]
  prior: BeliefState | null
  acquisitions: InformationEvent[]
  world: WorldIndex
returns: DialogueDecision
---
1. Capture the actor, requested fact conjunction, prior known/believed content, reception evidence,
   and temporal prerequisites. Include explicit missing premises in this compact local context.
2. In eval declare local LicenseJudgment with status: 'licensed' | 'unsupported' | 'unresolved',
   informationPath: string[], reason: string, witnesses: {sourceId: string; path: string; quote: string}[]. Construct nl.with<LicenseJudgment>
   over the changing input {assertedMeanings: event.assertedMeanings, reportedMeanings: event.reportedMeanings}.
3. Its instructions are: judge whether this speaker can responsibly assert this conjunction at this
   point; distinguish proposition knowledge, acquisition provenance, marked conjecture and report;
   require a supported origin and directed reception path; preserve unresolved prerequisite uncertainty.
4. Directly await the literal. Its actual result is the license premise; record its informationPath
   in the returned decision and carry its exact witnesses to downstream obligation audits.
5. A correct factual destination is insufficient if the line falsely claims a particular reading event.
   Conversely an oral route can license destination knowledge without granting physical letter custody.
6. If prior is null, the literal can license only visibly given initial knowledge or a marked uncertainty.
   Do not assign a fabricated state based on what an audience rather than this actor has seen.
7. Return sceneId, actor, assertedFactIds and the consumed judgment's status, path, reason and witnesses.
   Keep explicit uncertainty distinct from an evidenced lack of access.
   Include localDecision with this exact site path, current assertion input IDs, the returned judgment
   premise and its witnesses; create the ledger entry only after the awaited literal is consumed.
8. The literal returns a structured object through direct invocation. Do not treat the object as a
   finite-choice decision token or substitute a string-inclusion test for semantic conjunction judgment.
