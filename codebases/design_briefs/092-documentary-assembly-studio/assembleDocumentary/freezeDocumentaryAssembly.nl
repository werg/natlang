---
description: Freeze deliverables only when one exact version clears every release gate.
args:{assembly:AssemblyState;findings:CutFinding[];clearance:{clearanceTasks:string[];factCheckTasks:string[];blockedAssetIds:string[];correctedClaimIds:string[];};checkpoint:DocumentaryCheckpoint;project:DocumentaryProject;sourceEvidence:EvidenceItem[]}
returns:AssemblyRelease
---
Freeze only for the exact DocumentaryProject ID and version, after editorId review. Bind the sourceEvidence IDs, targetDurationSeconds and current editorial brief into the release digest; a corrected clip invalidates only outputs downstream of that clip.

When any blocked finding or unresolved task remains, return review state and do not freeze. Otherwise bind edit decision list, script, caption file, description and asset manifest to the exact assembly revision digest and source asset versions. A change to sequence, source revision or correction invalidates downstream deliverables. If checkpoint receipt is unknown, compare remote readback digest; retry only when absence of the prior effect is proven.
