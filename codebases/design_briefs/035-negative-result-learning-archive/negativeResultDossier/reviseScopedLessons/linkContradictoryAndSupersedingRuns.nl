---
args: { scope: LessonRevision[], attempts: AttemptTimeline[], inferences: InferentialStatus[], rivals: RivalMechanism[] }
returns: LessonRevision[]
---
Link every later valid attempt that contradicts or narrows a lesson under matching conditions. A success under changed conditions narrows the earlier applicability; it does not erase the old valid negative result.
