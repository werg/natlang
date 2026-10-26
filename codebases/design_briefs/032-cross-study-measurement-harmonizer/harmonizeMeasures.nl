---
description: Align cross-study measures only when construct meaning, missingness semantics, exact transforms, and row-level provenance support comparison.
args: { request: HarmonizationRequest, corpus: MeasurementCorpus }
returns: HarmonizationResult
uses: [measureHarmonizer/selectHarmonizationPolicy, measureHarmonizer/frameHarmonization, measureHarmonizer/compareConstructDefinitions, measureHarmonizer/classifyMissingness, measureHarmonizer/verifyCategoryMapping, measureHarmonizer/bindRowTransforms, measureHarmonizer/traceParticipantOverlap, measureHarmonizer/buildValidatedSlice, measureHarmonizer/probeMappingSensitivity, measureHarmonizer/invalidateRevisedMapping]
---
1. Bind applicablePolicy = selectHarmonizationPolicy(request, corpus) and frame = frameHarmonization(request, corpus.harmonizations, applicablePolicy). Hold target construct, analysis, owner, protocol and dataset version fixed. If owner or dataset authority remains unresolved, preserve it as a blocker on every derived slice and transformation.
2. Bind bridge = compareConstructDefinitions(request, corpus.measureCatalog, corpus.evidencePages). Do not align measures sharing a label when their instruments, populations, timepoints or construct differ.
3. Bind missingness = classifyMissingness(request, corpus.missingnessEvidence). Keep valid zero, nonresponse, structural absence, censoring and unknown distinct before any mapping.
4. Bind categoryCheck = verifyCategoryMapping(bridge, corpus.categoryDefinitions). A combined category may not be split into concepts that collection evidence does not record.
5. Bind transforms = bindRowTransforms(request, corpus.rowCollectionVersions, corpus.exactTransformReceipts). A unit change inside a dataset requires row-version-specific receipts; no exact receipt means no conversion.
6. Bind overlap = traceParticipantOverlap(request, corpus.participantLinkageEvidence). Keep uncertain cross-study overlap explicit and do not treat linked observations as independent.
7. Only when bridge comparability, missingness, category mapping and all required row transforms are evidenced, bind slice = buildValidatedSlice(request, bridge, missingness, transforms, overlap, corpus.rowCollectionVersions). Otherwise slice remains null and affected rows stay excluded.
8. Only for a validated slice, bind sensitivity = probeMappingSensitivity(slice, corpus.evidencePages). Compare evidence-supported alternatives and state which conclusions change; this is not a new observed dataset.
9. Bind revisionEffect = invalidateRevisedMapping(request, corpus.mappingRevisionEvents, slice when present otherwise []). Revisions invalidate only slices consuming the changed rule.
10. Return { caseId: request.caseId, applicablePolicy, measureDefinitions: corpus.measureCatalog, bridge, missingness, transforms, slice when validated otherwise null, sensitivity when slice exists otherwise [], revisionEffect, unresolved: applicablePolicy.precedenceQuestions + bridge.unresolved + missingness.unresolvedSemantics + categoryCheck, commitAllowed: false } as HarmonizationResult. No numeric conversion is executed in this semantic program; exact outputs come only from bound receipts.
