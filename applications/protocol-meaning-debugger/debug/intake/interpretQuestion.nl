---
description: Specify the target proposition before comparing implementation numbers.
args:
  request: ProtocolRequest
  index: EvidenceIndex
returns: Question
---
1. Consume actual index.definitions and index.unitResults as the interpreted prerequisite. Select clauses
   controlling request.targetProtocolId under explicit authority and revision rules.
   Read their references to population, temporal anchor, measurement interval and outcome coding.
2. State populationMeaning as a predicate over trial units, not over available reading rows. Distinguish
   commissioned equipment from equipment that happened to transmit data by the launch timestamp.
3. State anchorMeaning as the named accepted event and its occurrence scope. A calibration, retry,
   commissioning acknowledgment and actual accepted commissioning need not identify the same instant.
4. State windowMeaning with units, bounds and inclusivity. Separate relative offsets from calendar
   timestamps; a window after an anchor does not default to after the first recorded reading.
5. State metricMeaning as numerator proposition, denominator population and aggregation. Success
   percentage and failure percentage can coincide at50percent while answering opposite questions.
6. Keep every controlling clause in requirements with its exact witness. Preserve scope and probe
   restrictions as requirements even when they do not contribute directly to a baseline denominator.
7. Record missing or conflicting target meanings in unknowns. Do not make the better-looking
   implementation into the intended protocol merely because its narrative explanation sounds plausible.
8. Missing accepted definitions stay in unknowns with their actual failed/unresolved unit identities.
   Original documents verify provenance only in this stage; role/owner/meaning extraction belongs to the
   readEvidence frontier. A missing typed prerequisite is repaired there, not synthesized from raw documents
   as a new semantic index inside this function.
9. Return the typed question. This proposition is the reference for all contrast and repair judgments;
   the baseline number is never its definition.
