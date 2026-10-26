---
description: Separate fictional puzzle content from actual venue, prop, access and exit constraints.
args:{brief:EscapeBrief;props:PropState[];change:string[];sourceRecord:EscapeRoomProject;sourceEvidence:EvidenceItem[]}
returns:{prohibitedMechanisms:string[];exitRequirements:string[];accessibilityNeeds:string[];unverifiedPropIds:string[];}
---
Take creativeBrief, venueId, propIds, designerId and targetDurationMinutes from the top level of EscapeRoomProject. Match each propId and the seated clue route to the submitted room state; missing props or access paths remain unavailable. Carry source evidence IDs into the safety and fairness constraints. The story premise cannot replace an operational venue invariant.

Apply the supplied safety rules to physical props and participant movement. Preserve unrestricted emergency exit and staff override. Mark unknown prop capabilities as unavailable until verified. Carry participant access requirements without exposing unnecessary personal details.
