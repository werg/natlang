---
description: Parse language into issue terms while preserving commitment ambiguity.
args:{scenario:ArenaScenario;utterance:string;turns:ArenaTurn[];sourceRecord:NegotiationExercise;sourceEvidence:EvidenceItem[]}
returns:ParsedArenaOffer
---
Use the exercise top-level learningObjectives, participantIds, scenarioFamily, instructorId, conductPolicyId and workflowStatus to bind the actual exercise and debrief scope. Validate that the acting role belongs to a named participant and is governed by the sourced conduct policy. Do not accept scenario defaults that erase the source exercise’s training goals or participant authority.

Extract each issue separately with units, conditions and stated expiry. Classify exploratory, firm, conditional acceptance or withdrawal only when wording establishes it. Ask a neutral clarification for any missing term that changes feasibility, timing or commitment. Preserve the original utterance and turn identity.
