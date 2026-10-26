---
description: Trace each adjacent documentary sequence transition for claim, chronology and causal overreach.
args:{assembly:AssemblyState;indexed:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string;}[];argument:{beats:string[];claimIds:string[];causalTransitions:string[];unsupportedLinks:string[];};project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:{transition:CutTransition;fromSourceId:string;toSourceId:string;finding:CutFinding}
---
Walk the ordered cuts and bind each transition to the actual preceding and following asset IDs and the corresponding recorded narration. Compare the joint proposition with the project brief and cited evidence, checking whether the edit implies shared time, causation, agreement or contrast beyond the source ranges. A source's qualifier or surrounding context that changes the apparent claim must remain attached to that cut. Return one typed finding for each material transition, with exact source identities and the repair needed; do not treat authentic source clips as proof of the relationship created by editing.
