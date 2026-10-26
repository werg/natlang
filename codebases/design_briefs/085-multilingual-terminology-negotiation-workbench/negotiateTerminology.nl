---
description: Negotiate context-scoped multilingual terminology by concept and stage coherent versions into released content.
args: { dossier: TerminologyDossier; change: TerminologyChange; owner: TerminologyOwner }
returns: TerminologyProposal
---
Treat each occurrence as a language form in a specific product context, grammatical pattern and regional scope. Do not decide concept identity from string similarity or frequency.

Bind occurrences = `extractContextualOccurrences(dossier, change)` and bind sense, language, usage and source version. Bind concepts = `alignConceptBoundaries(dossier, change, occurrences)`; preserve multiple meanings and expert disagreement instead of choosing the most common form.

Bind termOptions = `constructScopedTermOptions(dossier, change, concepts)`. Keep regional variants scoped and generate grammatical inflections only for evidenced contexts. Each form remains pending review until checked.

Bind decision = `recordTerminologyDecision(dossier, change, concepts, termOptions.options, termOptions.inflections)` only when concept boundary, owner authority and accepted form are explicit. Otherwise keep the decision open and return the unresolved issue.

When an accepted decision affects released content, Bind migration = `stageTerminologyMigration(dossier, change, decision)`. Show affected strings and workflow dependencies; require versioned review and prevent a workflow from mixing old and new forms. Preserve prior released material until migration authorization.

Return `TerminologyProposal` for `owner.id` with context-linked occurrences, concept boundaries, reviewed forms, decision status, migration plan and resumable disagreement ledger. A proposed decision or migration is not a glossary release.
