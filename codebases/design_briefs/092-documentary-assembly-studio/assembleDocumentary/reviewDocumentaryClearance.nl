---
description: Route rights, correction and unresolved fact issues to bounded production tasks.
args:{assets:MediaAsset[];assembly:AssemblyState;indexed:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string;}[];corrections:EvidenceItem[];findings:CutFinding[];project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:{clearanceTasks:string[];factCheckTasks:string[];blockedAssetIds:string[];correctedClaimIds:string[];}
---
Use project.archiveAssetIds and interviewIds to identify the actual rights and speaker sources for each cut. Bind correction evidence to the exact source refs; an interview correction or missing archive permission remains open. Preserve the project editor as responsible reviewer.

Check each selected range's permission against recorded channel, expiry and territory scope in its source rights. Missing or expired authority blocks that asset and requests the responsible renewal. Match speaker corrections to the exact claim and affected cuts; open a focused fact-check, preserve prior wording, and mark affected captions and narration for update. Do not infer rights or factual resolution from silence.
