---
description: Judge whether source-defined downstream consumers distinguish two actual rewrite histories.
args:
  first: OrderPreview
  second: OrderPreview
  beforeBoard: BoardSnapshot
  source: TurnSource
returns: OrderDifference
---
1. Join final parts by original lineage, preserving both actual replacement histories and capacities.
   Identify changed kind, quantity, target, recipient or history. Keep missing and split lineages
   explicit; do not compare only the intersection of returned parts.
2. Capture source outcome/departure/trigger meanings, the two typed projections and their common
   beforeBoard. In eval declare ConsequenceComparison as {equivalent:boolean|null;
   affectedLineageIds:string[];consequenceRuleIds:string[];reason:string;witnesses:Witness[]}.
   Directly await a scoped NL literal judging the actual projection pair.
3. The literal asks whether a source consumer can distinguish damage, growth, departures, converted
   quantity, bound recipient or remaining shared capacity. Explanatory wording alone is not a game
   difference. Missing decisive meaning returns null with its premise rather than presumed equivalence.
4. Consume the actual result; validate lineage/rule domains and nonempty exact original support.
   Parent-bind both candidate identities from inputs. The literal decides semantic distinguishability.
5. Return the witnessed OrderDifference. Do not commit hypothetical outcomes or activate hypothetical
   trigger work slots while comparing these branch-local projections.
