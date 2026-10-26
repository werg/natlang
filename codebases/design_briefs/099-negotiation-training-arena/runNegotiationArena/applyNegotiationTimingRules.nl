---
description: Apply offer acceptance and withdrawal rules to the immutable turn ledger.
args:{scenario:ArenaScenario;offer:ParsedArenaOffer;response:CounterpartChoice;turns:ArenaTurn[]}
returns:ArenaTurn[]
---
Return `resolveIssueLevelTiming(scenario, offer, response, turns)` as the complete append-only ledger. Use the scenario acceptance rule against exact offer expiry and response order. A conditional acceptance leaves conditions open until supplied evidence satisfies them. A withdrawal after valid acceptance does not erase agreement; before acceptance it closes that open offer. An exploratory turn creates no commitment.
