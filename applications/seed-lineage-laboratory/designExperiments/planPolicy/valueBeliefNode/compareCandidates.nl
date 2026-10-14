---
description: Compare exact candidate cost vectors under the source objective.
args:
  candidates: CandidateValue[]
  objective: string
  witnesses: Witness[]
returns: CandidateComparison
---
1. Interpret the supplied objective and tie convention. Identify whether completion, expected cost,
   worst cost or source experiment identity is the next priority; preserve missing priorities as unresolved.
2. Separate incomplete candidates from policies that solve every positive-probability outcome branch.
   An inexpensive branch that abandons a possible mechanism is not a complete candidate.
3. Compare complete candidates using exact rational cross-products for each numeric priority, in the
   source-defined order. Apply a source identity tie rule only after all higher priorities agree.
4. Return the selected actual experimentId, any numerical ties and a source-witnessed explanation.
   Do not prefer a familiar assay, maximum immediate information or a fully resolved genotype unless
   those are actually the supplied objective.
5. If the objective or candidate completeness is unresolved, return null selection and its reason.
   The caller retains this actual comparison instead of silently selecting the first candidate.
