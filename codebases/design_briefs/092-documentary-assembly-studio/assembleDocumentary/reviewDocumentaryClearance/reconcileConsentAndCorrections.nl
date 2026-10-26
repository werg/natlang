---
description: Reconcile speaker consent, channel scope and corrections against the exact documentary cut ranges.
args:{assets:MediaAsset[];assembly:AssemblyState;indexed:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string;}[];corrections:EvidenceItem[];findings:CutFinding[];project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:{clearanceTasks:string[];factCheckTasks:string[];blockedAssetIds:string[];correctedClaimIds:string[];consentConflicts:string[]}
---
For each selected cut, bind the asset's source revision and speaker identity to its permission IDs. Check recorded consent for this channel, territory, term and the material use created by its neighboring transition. A participant's correction is attached only to the exact claim and source range it names; preserve the earlier version and open review when evidence conflicts. Return exact affected asset IDs, unresolved consent conflicts, renewal tasks and corrected claim IDs. A general release or a correction to one quote does not resolve a different cut's scope.
