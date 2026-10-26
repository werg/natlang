---
args: {"promises": "PromiseEpisode[]", "progress": "PromiseProgress[]", "prerequisites": "ContinuityPrerequisite[]", "slices": "OwnershipSlice[]", "introductions": "IntroductionIntent[]"}
returns: "{ transferComplete:boolean; workComplete:boolean; openWork:string[]; unacceptedTransfers:string[]; missingAccess:string[]; cycles:string[][]; questions:string[] }"
---
Check every active immediate obligation against the accepted ownership partition and confirmed next checkpoint. Preserve uncovered or overlapping scope as a transfer defect. Detect finite dependency cycles and explain the blocked next action. Distinguish ability to continue from completion of the promised deliverable. Handover transferComplete can be true while workComplete is false; neither status follows merely from departure, a sent dossier or proposed introductions. Return the specific unanswered acknowledgment, access or cadence confirmation needed next.
