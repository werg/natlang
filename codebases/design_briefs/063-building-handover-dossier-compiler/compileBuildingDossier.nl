---
args: { handover: HandoverCompilationInput }
returns: DossierPackage
---
1. Bind manuals = selectInstalledVariantManuals(handover.installedAssets, handover.manuals). A manual covering several variants must mark the installed variant and only the applicable sections.
2. Bind certificates = matchCertificatesByExactAsset(handover.installedAssets, handover.certificates). A familiar model number cannot overcome a certificate naming the wrong asset ID.
3. Bind archive = assessPortalDeliverablePreservation(handover.portalDeliverables). An expiring portal link requires a permitted durable copy and supplied hash verification before it can count as archived.
4. Bind exports = planAccessibleHandoverExports(handover.exportProfiles, handover.portalDeliverables). If facilities cannot open the source format, require an approved accessible export and preserve the source artifact.
5. Bind obligations = transferTemporaryArrangementTasks(handover.temporaryArrangements). Carry each arrangement expiry, accountable owner, and follow-up obligation into the handover package.
6. Bind missing = reconcileHandoverRequirements(handover.requirements, manuals, certificates, archive, exports, obligations). Mark ready only if each required asset artifact is correctly matched, permitted and accessible, and temporary obligations have owners.
7. Return { caseId: handover.caseId, manuals, certificates, archiveActions: archive, accessibleExports: exports, transferredObligations: obligations, missingRequirements: missing, readyForAcceptance: missing.length == 0, commitAllowed: false }.
