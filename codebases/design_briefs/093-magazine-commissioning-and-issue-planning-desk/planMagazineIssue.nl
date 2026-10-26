---
description: Evaluate pitches, secure exact contributor terms, and reconcile an issue portfolio for freeze.
args:{ sourceRecord: MagazineIssue; sourceEvidence: EvidenceItem[];brief:IssueBrief;issue:IssueState;checkpoint:EditorialCheckpoint}
returns:IssueReadinessPacket
---
Bind boundIssue = `bindMagazineSource(sourceRecord, sourceEvidence, brief, issue)`. This binding uses `issueTheme`, `audience`, `sectionIds`, `editorId` and `publicationDate` from the typed MagazineIssue record itself. Replace the generic placeholder theme, audience and date in both brief and issue state with those exact values; keep all pitch, commission and production facts only where their source evidence supports them. Carry sectionIds as editorial section constraints and preserve the source editor as accountable editor. If the original evidence IDs are absent from sourceEvidence, retain a blocker.

Read `issueTheme`, `audience`, `sectionIds`, `editorId` and `publicationDate` from the top level of the MagazineIssue record. These values govern pitch fit, section allocation and deadline state; a generic brief or placeholder date cannot override the source record. Carry every source fact and evidence ref into the analysis branch that depends on it; when any transformed proposal conflicts with this anchor, hold that branch for review instead of accepting the placeholder.

Bind assessments = `scorePitchPortfolio(boundIssue.brief, boundIssue.issue.pitches, boundIssue.issue.commissions)` to evaluate feasibility, distinctive value, reporting dependencies, and overlap as a portfolio. Bind coverage = `mapIssueCoverage(boundIssue.brief, assessments, boundIssue.issue.contingencies)`; identify genuine audience and format gaps and redundancy without manufacturing false balance.

Bind commissions = `secureExactCommissions(boundIssue.brief, assessments, boundIssue.issue.commissions)`. A pitch is not agreement: only a contributor's acceptance bound to the exact terms revision moves the commission to accepted. Unavailable access reopens its reporting plan; do not represent the original plan as completed. Bind findings = `reconcileEditorialProduction(boundIssue.issue, commissions, checkpoint)` to check each piece's reporting, fact-check, rights, layout assets, headline and promotional claims.

When an accepted story withdraws, select only a prequalified contingency, recompute coverage and dependencies, and leave the gap visible if none fits. Return `qualifyIssueFreeze(boundIssue.issue, findings, coverage, checkpoint)`; credit changes invalidate print, web and promotional derivatives. Freeze only one exact issue revision after all gates clear.
