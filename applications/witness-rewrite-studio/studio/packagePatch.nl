---
description: Assemble an exact patch proposal with complete source-grounded coverage.
args:
  request: RewriteRequest
  analysis: Analysis
  impact: ImpactResult
  state: RepairState
returns: PatchPacket
---
1. Collect chosen candidate IDs, selected edits, rejected proposals and consumed local decisions
   from state.decisions, original frame and knowledge localDecisions, and impact.localDecisions.
   Keep each actual decision ID once. Do not invent a final repair absent from a settled component choice.
2. For every impacted obligation, collect the chosen candidate's actual audit findings. Mark coverage
   satisfied only when every required owned finding is satisfied and exact checks are valid.
3. Otherwise mark that obligation unresolved and carry its source IDs and concrete reason. A packet
   can contain valid independent edits alongside unresolved work without claiming full completion.
4. Check all accumulated edits together with storyEvidence.checkEdits against the original snapshot.
   Quote/path failures remain sourceChecks and prevent an executable complete packet.
5. Verify original-source witnesses against analysis.intake.snapshot. Edited-scene witnesses were
   checked against each replay snapshot; retain their audit checks rather than reciting them as original.
6. Determine unchangedSceneIds from selected scene identities with no selected edit. Determine
   preservedUncertaintyIds from protected uncertainties whose chosen ambiguity findings are satisfied.
7. Include state unresolved, original interpretation unknowns, graph defects and cross-component
   compatibility findings. Keep concrete IDs and meanings; do not replace them with a success slogan.
8. Return baseRevision, requestId, selectedProposalIds, edits, unchangedSceneIds, preservedUncertaintyIds,
   obligationCoverage, rejectedProposals, unresolved, sourceChecks and consumedLocalDecisionIds.
9. This packet is pure effect data. An exact host can apply it after base/span validation, but this
   stage neither publishes a story nor interprets a private decision folder as a completed commit.
10. Every selected replacement must be backed by an audited proposal and a connected obligation.
    Exact services verify bytes and identity; NL children own all meaning preservation judgments.
