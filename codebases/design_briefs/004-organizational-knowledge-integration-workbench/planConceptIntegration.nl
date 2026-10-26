---
args: {"inquiry": "IntegrationInquiry", "atlas": "IntegrationAtlas | null"}
returns: "IntegrationBlueprint"
---
1. Bind senses = readOrganizationalUsage(inquiry). Preserve source organization aliases and distinct senses under identical acronyms. Context-free glossary labels cannot establish equivalence.
2. For each plausible cross-organization sense pair bind bridge = testOperationalSubstitution(left, right, inquiry.usageSources). Collect bridges, retaining asymmetry, partial overlap and explicit team rejection. If a bridge is disputed, keep that operational boundary separate rather than forcing a universal definition.
3. For supported bridges bind handoffs = negotiateRoleInterfaces(bridges, senses, inquiry). Work at role level when personnel access is restricted. A similar job title or shared acronym is not a responsibility agreement.
4. Bind scenarios = walkCrossTeamTransactions(senses, bridges, handoffs, inquiry). Use their unowned steps and lost distinctions to revise the affected handoff proposals; no critical gap may disappear under a shared term.
5. Bind waves = planReversibleFamilies(bridges, handoffs, scenarios, inquiry, atlas). Any ambiguous meaning, unaccepted ownership, failed semantic round trip or missing exact integrity receipt holds that family. Independent ready families remain separate.
6. Assemble the revised atlas, preserving actual executed waves, historical aliases and already effective decisions. Return readyWaveKeys only for fully supported reversible families and roleAgreements still requiring the identified owners' acceptance. Keep all disputed boundaries and smallest source requests visible; no plan is an executed migration.
