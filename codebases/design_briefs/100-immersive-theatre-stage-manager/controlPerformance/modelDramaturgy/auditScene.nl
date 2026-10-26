---
description: Audit one theatre-scene draft against its scoped evidence slice and reveal graph.
args: { candidate: SceneCandidate; draft: DraftTheatreScene }
returns: StageAudit
---
1. Bind verdict = qualify(candidate.evidence, draft). This child returns the finite string pass, fail or needs-review. Preserve its failed or unresolved result; it is not a StageAudit object.
2. Examine each graph edge entering candidate.sceneId against the draft's prerequisites and named recipients. A supported scene requires a permitted reveal before its use on that same route. Check knowledge by character and by audience group separately. Do not infer a recipient's knowledge from narrator knowledge or from a future scene.
3. If an evidenced hard contradiction or prohibited reveal remains, the final verdict is fail. If an essential source, ordering relation, recipient identity or required draft field is absent, the final verdict cannot exceed needs-review. A graph check can downgrade pass, but cannot upgrade the child's fail or needs-review.
4. Return StageAudit { verdict: the final verdict, gate: "reveal-order-and-knowledge", findings: the exact scoped contradictions and evidence gaps, repairTargets: the affected scene and reveal-edge identities }. A pass has no unresolved required findings. Keep optional uncertainties distinguishable from prerequisites.
