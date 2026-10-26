---
description: Review quoted meaning and sequence-level implication under exact runtime.
args:{assembly:AssemblyState;indexed:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string;}[];argument:{beats:string[];claimIds:string[];causalTransitions:string[];unsupportedLinks:string[];};project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:CutFinding[]
---
Require every causal or chronological claim to fit the project editorialBrief and cite sourceEvidence actually linked by the project. Keep targetDurationSeconds as the intended runtime; if the supplied assembly frame target disagrees, record a duration blocker instead of changing the brief.

Bind sequenceReview = `reconcileSequenceImplications(assembly, indexed, argument, project, sourceEvidence)`. Use each returned transition's actual source IDs and finding when evaluating chronology or causal adjacency; do not collapse transitions that reuse an asset. Check if each selected quote retains qualifiers that change its meaning. Any unsupported causal transition is blocked. Sum exact frame durations using frame rate; runtime mismatch blocks the cut. Return the complete union of quote, runtime and sequenceReview findings, preserving their exact affected asset IDs and repair actions.
