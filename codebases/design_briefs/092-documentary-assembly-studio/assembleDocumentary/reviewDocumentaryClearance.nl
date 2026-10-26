---
description: Route rights, correction and unresolved fact issues to bounded production tasks.
args:{assets:MediaAsset[];assembly:AssemblyState;indexed:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string;}[];corrections:EvidenceItem[];findings:CutFinding[];project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:{clearanceTasks:string[];factCheckTasks:string[];blockedAssetIds:string[];correctedClaimIds:string[];}
---
Use project.archiveAssetIds and interviewIds to identify the actual rights and speaker sources for each cut. Bind correction evidence to the exact source refs; an interview correction or missing archive permission remains open. Preserve the project editor as responsible reviewer.

Bind consentReview = `reconcileConsentAndCorrections(assets, assembly, indexed, corrections, findings, project, sourceEvidence)`. Use its cut-specific blockedAssetIds and correctedClaimIds as controlling inputs; add consentConflicts to clearanceTasks with their actual source refs. Preserve renewal and fact-check tasks from the original review only when they identify a selected cut. Return the complete clearance object with unioned tasks, blocked asset IDs and corrected claim IDs; never let an unbounded release or unrelated correction close a cut-level conflict.
