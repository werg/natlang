---
description: Audit asserted dialogue against actor-local prior information.
args:
  intake: IntakeResult
  world: WorldIndex
  acquired: KnowledgeIndex
returns: DialogueDecision[]
---
1. Inventory dialogue assertions in selected events and their asserted fact IDs. Keep guesses,
   questions, recalled possibilities and confident factual claims as different speech acts.
2. For each assertion, find the actor's BeliefState immediately before its event. A later successful
   rescue cannot retrospectively license how the actor knew where to go.
3. Select acquisitions and source-backed temporal paths that can actually reach this prior state.
   Exclude knowledge occurring only in an unselected memory or a proposal document.
4. Call resolveLicensing(event, actor, factIds, priorState, relevantAcquisitions, world).
   Consume its returned DialogueDecision unchanged as this assertion's licensing contribution.
5. A confident factual assertion without a supported route is unsupported. A marked guess may be
   licensed as a guess, while an unresolved temporal premise stays unresolved rather than false.
6. Include dialogue whose wording claims how information was acquired. Knowing the destination
   does not license claiming to have read a letter when the only supported route was oral testimony.
7. Return all decisions, preserving sceneId, actor, exact source witnesses and actual informationPath.
   Do not filter out unsupported lines: those are the downstream obligations that make this useful.
8. The finite assertion inventory is the traversal frontier. One line can carry multiple facts and
   provenance claims; resolve their conjunction rather than checking only its most convenient fact.
