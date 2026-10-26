---
description: Require each action prerequisite to be available in this character's authorized perspective.
args: { perspective: CharacterPerspective; action: LegalAction }
returns: ActionCheck
---
Require action.kind to occur in perspective.availableActions. Treat action.requires as exact prerequisite identifiers. Each identifier must occur in perspective.visibleFacts or perspective.permittedMemories. Do not substitute a similarly worded fact or another character's knowledge. Return every absent prerequisite in blockers and eligible only when none are absent. Use source-event identifiers attached to those permitted facts when supplied; never invent evidence IDs.
