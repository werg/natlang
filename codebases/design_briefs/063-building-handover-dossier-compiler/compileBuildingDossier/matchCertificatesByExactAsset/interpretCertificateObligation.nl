---
args: { asset: InstalledAssetVariant, certificate: AssetCertificate, requirement: HandoverRequirement }
returns: CertificateObligationFinding
---
Interpret requirement.assetClass, requiredArtifact and completionCondition against this installed asset and certificate. Decide applicability from the supplied model/variant/location and named asset evidence; an unstated class mapping is unknown. If the requirement concerns another artifact class, return applicability "no" without treating that as certificate acceptance.

For an applicable obligation, require the namedAssetId to match, the evidence reference to be nonempty and versioned, and the certificate status and revision to satisfy the actual completionCondition. An expired, failed, superseded or unverified status cannot satisfy a condition merely because the equipment model is familiar. Ambiguous status vocabulary or a missing required revision is a question, not permission to choose a favorable interpretation.

Return { requirementId: requirement.id, applicability: "yes", "no" or "unknown" as established above, acceptable: true only for an applicable and evidenced condition, reasons: the actual identity/status/revision/coverage findings, evidenceRefs: [certificate.evidenceRef, asset.evidenceRef] plus requirement.sourceRefs actually used } as CertificateObligationFinding. Do not invent a regulator, expiration rule or acceptance standard absent from this requirement.
