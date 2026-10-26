---
description: Test each pitch against actual issue sections, accepted commission workload and publication deadlines.
args:{brief:IssueBrief;pitches:StoryPitch[];commissions:Commission[];sectionIds:string[];publicationDate:string}
returns:{pitchId:string;eligibleSectionIds:string[];deadlineConflict:boolean;workloadConflicts:string[];formatConflict:boolean;dependencies:string[];unresolvedConstraints:string[]}[]
---
For each pitch, match its format and angle to the supplied section identifiers and brief formats. A candidate is not placeable when no section can carry it, its deadline falls after publicationDate, or an already accepted commission consumes the contributor capacity or reporting access it requires. Preserve conditional access as a dependency rather than a rejection. Return one placement record for every pitch, naming eligible sections, exact conflicts, shared dependencies and unknown capacity or section rules. Do not create a section or capacity value that the issue contract did not supply.
