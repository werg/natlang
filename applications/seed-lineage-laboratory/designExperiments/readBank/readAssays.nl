---
description: Bind observations to the tissue, stage, regime and resolution they actually measure.
args:
  clauses: SourceClause[]
returns: AssayIndex
---
1. Inventory source assay protocols, keeping compartment, developmental stage, regime, readout
   resolution and visible answer grammar for each. Distinguish sensitivity failure from a dark result.
2. Interpret protocol validity conditions. Return usable null when required calibration is missing,
   false when the source explicitly invalidates the regime, and true when its conditions are established.
3. For each recorded observation, collect its own source clause and the linked protocol/calibration.
   Call judgeObservationScope(observationClauses, protocolClauses), then consume that actual result.
4. Bind observationId, accessionId and reported answer from the original record. Keep a proposed
   experiment separate from an observation already made, even when their readout wording matches.
5. Preserve observations whose scope is invalid or unresolved. Their existence is evidence about the
   record, but their result cannot eliminate a world about a quantity they do not measure.
6. Return AssayIndex with exact witnesses and unresolved scope obligations. Later updates receive
   these typed scopes instead of interpreting a color word in isolation.
