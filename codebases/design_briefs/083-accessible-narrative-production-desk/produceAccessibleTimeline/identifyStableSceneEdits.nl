---
args: {"changes": "EvidenceItem[]", "scenes": "SceneInformation[]", "prior": "AccessibleMediaState"}
returns: "SceneEditDependency[]"
---
Match timeline changes through stable scene IDs and actual prior/new source versions. Identify insertions, trims, moves and changed reveal order separately. Derive exact millisecond shifts only when the timeline provides exact millisecond or exactly convertible time values; unresolved frame/timebase conversion remains a blocker. List only cues depending on changed scene/time/reveal data and preserve unrelated cues and their reviews.
