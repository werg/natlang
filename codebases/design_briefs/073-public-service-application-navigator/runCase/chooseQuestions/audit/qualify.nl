---
description: Finite independent gate for chooseQuestions in Public Service Application Navigator.
args: { evidence: EvidenceSlice; draft: DraftNavigationQuestion }
returns: '"pass" | "fail" | "needs-review"'
readout: decision
---
Return needs-review whenever draft.missingFields is nonempty or required values are absent. Return pass only when draft.missingFields is empty and draft.values satisfies this exact domain gate with attributable, current supporting evidence: No unnecessary collection or repeated request for already available evidence. Return fail when current admissible evidence affirmatively contradicts a required claim or a hard invariant is violated. Return needs-review when decisive support is absent, sources materially conflict, authority/freshness is unclear, or the searched scope is incomplete. Evidence status unknown is never a pass. Check the supplied slice and draft; do not infer hidden facts.
