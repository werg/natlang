---
description: Apply offer acceptance and withdrawal rules to the immutable turn ledger.
args:{scenario:ArenaScenario;offer:ParsedArenaOffer;response:CounterpartChoice;turns:ArenaTurn[]}
returns:ArenaTurn[]
---
Append a new turn; never rewrite prior utterances. Apply the scenario acceptance rule to exact offer expiry and response order. A conditional acceptance leaves conditions open until explicitly satisfied. A withdrawal after a valid acceptance does not retroactively erase agreement; before acceptance it closes the open offer. Create no commitment from an exploratory turn.
