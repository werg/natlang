---
description: Interpret inclusion and data availability separately for each trial unit.
args:
  ownerId: string
  trial: TrialData
  lineage: Lineage
  question: Question
returns: PopulationView
---
1. Locate this owner's selected population clause through lineage. Read its predicate and cutoff,
   including whether it refers to accepted commissioning or first available telemetry.
2. Capture that definition, target question and raw unit event fields. In eval declare local Eligibility
   with included: boolean | null, reason: string, witnesses: {sourceId:string;path:string;quote:string}[].
   Construct nl.with<Eligibility> with changing input one TrialUnit from the finite trial inventory.
3. Its instructions are: apply the captured population predicate to this unit; distinguish readiness
   of the physical device from readiness of its recorded data; use accepted event identity and stated
   cutoff; preserve missing decisive evidence as null instead of silently excluding the unit.
4. Directly await the literal for each unit. Consume true into includedUnitIds, false into excluded,
   and null into unresolved with its exact missing-premise reason and witnesses.
5. Record LocalDecision entries for the actual returned unit judgments and inclusion contribution.
   The numeric service receives only these decided IDs; it cannot choose a missing unit's membership.
6. Late telemetry can leave a device in the intended population even when its first available reading
   is later. Conversely an implementation explicitly defined by telemetry may exclude that device.
7. Return implementationId, selected definitionId, includedUnitIds, excluded, localDecisions and unresolved.
   Do not erase an implementation's wrong meaning here to make its numbers match the target.
8. A unit with no eligible reading can remain population-included while contributing no attempts.
   Population and denominator are different stages; record the absence rather than inventing a row.
