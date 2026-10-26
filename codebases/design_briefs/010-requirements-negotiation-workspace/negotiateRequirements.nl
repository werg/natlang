---
description: Reconcile stakeholder positions into a phased specification with exact conflict witnesses and replayed acceptance cases.
args: { request: NegotiationRequest; dossier: NegotiationDossier }
returns: NegotiationProposal
---
Bind scope = `frameNegotiation(request, dossier.project)`. Keep product boundary, release target, sponsor authority and prior baseline version explicit. Bind positions = `extractPositionsDomain(scope, dossier)`; classify desired outcome, implementation suggestion, mandatory constraint and preference separately. A successor role must confirm future changes, and verbal-only assent remains pending.

Bind conflicts = `findRequirementConflicts(scope, positions, dossier.constraintEvidence)`; accept a hard conflict only with a supplied constraint witness, and retain semantic counterexamples as explanation rather than claiming exact impossibility. Bind compromises = `designNegotiationOptions(scope, positions, conflicts, dossier.agreementRecords, dossier.interfaceOwners)`; each option names preserved outcomes, concessions, assumptions, affected parties and confirmation state, including no agreement.

Bind specification = `compileAcceptedRequirements(scope, positions, compromises, dossier.agreementRecords)`. Only confirmed outcomes and authorized concessions become atomic clauses; unresolved choices remain outside baseline and each clause carries owner, release phase, failure behavior and source position. Bind scenarios = `replayRequirementAcceptance(scope, specification, dossier.simulatorSchemas, dossier.simulatorResults)`. Bindings must match scenario and schema versions. Reopen only clauses named by a counterexample; preserve unrelated stakeholder acceptance.

Return a review proposal with pending successor/participant confirmations, affected-interface reviews, phase-tagged clauses and unexplained counterexamples. Set commitAllowed to false: negotiation does not persist a baseline or dispatch work.
