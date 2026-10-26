---
description: Rehearse each witnessed graph state against prop loss, out-of-order clues, repeated actions and access variants.
args: { room: EscapeRoomState; proof: PuzzleVerification; variants: AccessibilityRoute[]; sourceEvidence: EvidenceItem[] }
returns: PlaytestFinding[]
---
1. Bind cases = `enumerateFailureCases(room, proof, variants)` from the exact explored proof states. Keep each state, affected prop and route variant attached to its case; do not invent a state absent from the proof.
2. For each case call `judgeFailureBranch(room, proof, variants, sourceEvidence, case)`. Preserve its distinct recovery finding. A viable story explanation is not a staff procedure, and a suggested hint is not a receipt that the world state was restored.
3. Return every finding, including the branch that blocks a required goal, has no safe staffed recovery, or loses equivalent access. Do not collapse different graph states because they mention the same prop.
