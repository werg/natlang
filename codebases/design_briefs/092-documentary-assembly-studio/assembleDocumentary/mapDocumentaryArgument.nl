---
description: Map sequence beats to evidence and distinguish theme from causation.
args:{assembly:AssemblyState;indexed:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string;}[];project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:{beats:string[];claimIds:string[];causalTransitions:string[];unsupportedLinks:string[];}
---
Bind each cut to project.interviewIds or project.archiveAssetIds by exact source ID. Keep the project editorialBrief and targetDurationSeconds as the scope for claim adjacency and timeline review. Missing evidence links block attribution.

Trace each beat to approved claim identifiers. Record every adjacency and narration as an argument choice. Similar subject matter alone establishes no causal link. When the transition implies causation, require a supplied evidence claim that covers that causal proposition; otherwise preserve an unsupported-link finding.
