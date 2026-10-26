---
description: Choose only a rehearsed route that preserves dramatic beat, consent, knowledge and current resource limits.
args: { incident: ShowIncident; dramaturgy: DramaturgyMap; resources: ResourceTimeline; showEvidence: EvidenceItem[] }
returns: VariantSet
---
1. Bind candidates = compareRehearsedRoutes(incident, dramaturgy, resources, showEvidence). Work separately for every affected scene, preserving the candidate's approval revision, intended beat, consent path and named resource demands.
2. For each alternative bind check = verifyBeatAndKnowledge(candidate, dramaturgy, resources). Keep only rehearsed-approved alternatives whose exact check.eligible is true. Record every rejected candidate and its actual blockers; do not let an approved label override a changed cast, prop, consent or knowledge condition.
3. Among eligible alternatives, compare the current approved production priorities: preserved dramatic beat, materially equivalent nonparticipatory experience, least change to already witnessed knowledge, then current resource feasibility and authorized recovery timing. Explain the source-backed tradeoff. Array position is not a selection rule. An unresolved material tie is a stage-manager question, and no route is selected for that scene until it is resolved.
4. Build one RouteComparison per scene containing all alternatives, the chosen ID or an empty ID, the comparative reasons and unresolved blockers. Collect selected candidates and scene-specific blockers in VariantSet { comparisons, selected, blockers }. If the request affects no scene, explain that absence instead of inventing a scene.
5. A refusal cannot trigger another invitation. A failed prop excludes that prop and unapproved physical substitutes. Keep an otherwise promising unapproved rewrite separate for rehearsal review; it cannot enter selected.
