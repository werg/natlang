---
args: {"decisions": "LedgerDecision[]", "premises": "PremiseLink[]", "changes": "PremiseChange[]", "pending": "ReviewQuestion[]"}
returns: "DecisionFrontier"
---
Seed a finite frontier with decisions whose confirmed necessary premise was contradicted or narrowed in a supported overlap. Track visited decision and edge identities. Expand necessary links before considering material supporting links. On a repeated active path record the whole cycle component; do not flatten it or traverse forever. Keep independently justified work outside cancellation recommendations. Bind questions = groupPremiseQuestions(frontierDecisions, premises, changes, pending), where frontierDecisions are the actual discovered LedgerDecision values. Return ordered necessary/supporting identities, cycles, independent work and those questions.
