---
description: Replay one private candidate through the real semantic branches.
args:
  analysis: Analysis
  impact: ImpactResult
  component: AffectedComponent
  candidate: Candidate
  selectedEdits: ProposedEdit[]
uses: [readRecords, world, epistemics, props, motifs, audit]
returns: CandidateReview
---
1. Combine already selected edits and this candidate's edits. Call storyEvidence.checkEdits on the
   original snapshot and allowed paths. Exact overlap or span failure makes replay non-executable.
2. If exact checks fail, retain the original analysis in Replay, with claimJudgment containing no covered IDs and all claimed IDs unsupported,
   and return an Audit with violated
   source-execution findings for the component obligations. Do not pretend the replacement happened.
3. Otherwise call storyEvidence.applySnapshot to create a new immutable in-memory snapshot.
   Source files and original analysis remain unchanged; this counterfactual has its own revision.
4. Call readRecords(candidateSnapshot), retaining the original meaning/rules in a new IntakeResult.
   Director restrictions apply to both old and edited story even if a proposal tries to rewrite them.
5. Call world(replayIntake), then epistemics(replayIntake, replayWorld), props(replayIntake, replayWorld)
   and motifs(replayIntake, replayWorld, replayKnowledge). Consume their changed typed contributions.
6. Capture candidate.claimedCoverage with original required meanings and the replayed branch views.
   In eval declare local ReplayClaimJudgment with coveredIds: string[], unsupportedIds: string[],
   explanation: string, witnesses: {sourceId: string; path: string; quote: string}[]. Construct nl.with<ReplayClaimJudgment> over the changing
   input candidate.edits; directly await it to judge whether replay actually realizes claimed changes.
7. Its instructions distinguish an edit's promised meaning from what its resulting scene entails.
   Knowledge provenance, custody and callback meaning are checked jointly; retain source witnesses.
8. Put the actual judgment in Replay.claimJudgment. Populate Replay.consumedLocalDecisions from
   replayWorld.frames.localDecisions, replayKnowledge.localDecisions, impact.localDecisions, and the
   actual replay-claim literal decision ID, retaining each distinct ID once. Do not promote claimed
   coverage automatically; an unsupported claim is carried as a finding in the following audit.
9. Assemble Replay from the candidate, new snapshot, reconstructed Analysis and consumed decisions.
   Call audit(originalAnalysis, impact, component, replay) and return {replay, audit: returnedAudit}.
10. Keep private application mechanics exact. All authority, information, physical and dramatic
    changes are judgments of the called NL branches, not replacement-string classifiers.
