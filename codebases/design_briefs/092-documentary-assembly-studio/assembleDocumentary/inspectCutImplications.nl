---
description: Review quoted meaning and sequence-level implication under exact runtime.
args:{assembly:AssemblyState;indexed:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string;}[];argument:{beats:string[];claimIds:string[];causalTransitions:string[];unsupportedLinks:string[];};project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:CutFinding[]
---
Require every causal or chronological claim to fit the project editorialBrief and cite sourceEvidence actually linked by the project. Keep targetDurationSeconds as the intended runtime; if the supplied assembly frame target disagrees, record a duration blocker instead of changing the brief.

Check if each selected quote retains qualifiers that change its meaning. Evaluate the apparent proposition created by the adjacent clip and narration, not only each authentic clip in isolation. Any unsupported causal transition is blocked. Sum exact frame durations using frame rate; runtime mismatch blocks the cut. Emit findings only for identified assets/transitions and state specific restoration or transition revisions.
