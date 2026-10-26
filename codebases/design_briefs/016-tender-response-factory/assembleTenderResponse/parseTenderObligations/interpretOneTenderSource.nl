---
args:{goal:GoalContract;policy:PolicySelection;response:TenderResponse;evidence:EvidencePage[];addenda:EvidencePage[]}
returns:TenderRequirement[]
---
From this tender response and its source-linked schedules, forms and ordered addenda, extract each distinct eligibility rule, scored criterion, conditional duty, response slot, naming requirement and attachment. Bind every duty to its exact source edition and answer location. Treat a changed deadline or superseding clause as scoped to the obligations it actually changes; preserve unaffected requirements. Unsupported claims remain unanswered rather than invented.
