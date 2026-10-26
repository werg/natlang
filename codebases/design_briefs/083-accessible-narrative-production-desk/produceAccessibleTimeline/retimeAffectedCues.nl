---
args: {"edits": "SceneEditDependency[]", "prior": "AccessibleCue[]"}
returns: "AccessibleCue[]"
---
For each affected cue call shiftCueInterval(startMs, endMs, oldSceneStartMs, newSceneStartMs) when all four values are known and the edit is a pure translation. A trim, changed duration or changed reveal order requires fresh channel allocation rather than blind offset. Preserve cue identity and source dependency while marking its old player review stale. Unknown bounds remain unresolved instead of defaulting to zero.
