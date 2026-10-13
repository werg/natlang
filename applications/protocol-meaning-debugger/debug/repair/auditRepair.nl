---
description: Rebind repaired definitions and rerun every discriminating trial through the same NL materialization.
args:
  context: InvestigationContext
  state: InvestigationState
  diagnosis: Diagnosis
  proposal: RepairProposal
uses: [readEvidence, lineage, plans, probes/predict]
returns: RepairReplay
---
1. Call measurementLab.checkRepairs on original documents, proposal edits and allowed paths.
   Exact span, hash, identity or overlap failure yields verified:false and explicit violated findings.
2. Otherwise call measurementLab.applyDefinitions to create private edited documents. Keep the
   original input data and target question authority unchanged.
3. Call readEvidence(editedDocuments), then lineage(editedIndex, context.request). Construct edited
   Intake with the original input/question and newly interpreted index; use the rebound definition links.
4. Call plans(editedIntake, editedLineage, baselineTrial, request). Consume all three repaired baseline
   semantic views and exact numerical receipts rather than reusing the old plan fields.
5. For every discriminating or control outcome already consumed, call predict with the same raw seed
   under edited context. Preserve materialized row membership, anchors and numerator interpretations.
6. For every claimed repaired contrast compare target and repaired right meanings and choices across
   baseline and probe trials. Return a source-grounded satisfied/violated/unresolved finding per contrast.
7. Verify the left/reference implementation and raw probe facts are unchanged. A repair cannot improve
   agreement by deleting inconvenient test rows, rewriting the target or selecting an archived protocol.
8. Set verified true only if exact checks pass, every required cause has satisfied coverage and every
   required trial preserves the target proposition. Numerical equality is necessary but not sufficient.
9. Return proposal, repaired baseline, probe replays, findings and exactChecks. A missing repair
   coverage remains explicit and cannot be filled by a success sentence in report.
10. This child performs actual NL reinterpretation and exact numeric replay. Its source-valid witness
    remains distinct from independent semantic verification or a live executor trajectory.
