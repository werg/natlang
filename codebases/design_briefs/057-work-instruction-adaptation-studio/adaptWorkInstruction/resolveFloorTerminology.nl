---
args: { terms: TerminologyEntry[], equipment: EquipmentAlias[] }
returns: AmbiguitySet
---
Resolve a local term only when the supplied meaning and evidence identify one control. When an alias names two assets or controls, preserve the alternatives and block only the step that depends on that reference. Never use familiarity or location as identity evidence.
