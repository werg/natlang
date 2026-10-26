---
description: Apply scenario acceptance, expiry, conditional-term and withdrawal rules to one immutable turn.
args:{scenario:ArenaScenario;offer:ParsedArenaOffer;response:CounterpartChoice;turns:ArenaTurn[]}
returns:ArenaTurn[]
---
Read the scenario's exact acceptanceRule and each issue's term conditions against the ordered turn ledger. A response accepts only the same open offer before expiresAt and only for terms the counterpart has authority to accept. Keep conditional acceptance open until every named condition has actual supplied satisfaction evidence; never infer satisfaction from silence. A withdrawal closes only an offer not already accepted under the timing rule. Append exactly one immutable turn bound to the parsed offer and response, preserve every prior turn, and return the full ledger with the correct open/accepted/withdrawn/rejected/clarification-needed state.
