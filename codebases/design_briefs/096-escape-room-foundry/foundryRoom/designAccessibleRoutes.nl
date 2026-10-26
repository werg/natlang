---
description: Provide equivalent clue paths for supplied sensory or access constraints.
args:{room:EscapeRoomState;constraints:{prohibitedMechanisms:string[];exitRequirements:string[];accessibilityNeeds:string[];unverifiedPropIds:string[];};clueReview:{ambiguousClueIds:string[];lateClueDependencies:string[];misreadingCandidates:string[];repairTargets:string[];}}
returns:AccessibilityRoute[]
---
For each declined sensory element, map an equivalent clue to the same logical information and terminal progression. Do not remove a required decision or add a prerequisite available only on the default route. Keep the emergency path independent of puzzle completion.
