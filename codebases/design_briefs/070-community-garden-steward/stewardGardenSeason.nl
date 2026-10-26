---
description: Steward one garden season through consent-bound plot plans, shared resources and accepted task transfers.
args: { garden: GardenDossier; change: GardenChange; steward: GardenSteward }
returns: GardenProposal
---
Begin from the current accepted agreements in `garden`. Do not convert member preferences or missed work into obligations.

Bind preferences = `interpretMemberAgreements(garden, change)` and bind the attributed preference ledger. Bind conditions = `inspectPlotConditions(garden, change)` and keep untested conditions visibly uncertain.

Bind alternatives = `coDesignPlotAlternatives(garden, change, preferences, conditions)`. Preserve multiple feasible plans and name tradeoffs across all affected members. If the plot condition is uncertain, exclude only alternatives whose safety or success depends on the unknown and request a suitable assessment.

Bind negotiation = `negotiatePlotAllocation(garden, change, alternatives)`. Bind explicit responses to exact proposed terms. Keep declined or unanswered duties unassigned; a reversible trial requires every affected member's explicit acceptance and a review condition.

Bind board = `rollSeasonBoard(garden, change, negotiation)`. Schedule only accepted duties and current resource reservations. Hold tasks dependent on broken tools, and transfer care or harvest rights only through a documented accepted agreement.

Return `GardenProposal` for `steward.id` with dissent, held tasks, uncertainty and complete `resumeState`. Ready means the proposal is reviewable under current consent. It is not itself an assignment, agreement or completion receipt.
