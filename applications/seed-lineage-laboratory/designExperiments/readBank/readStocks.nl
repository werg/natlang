---
description: Interpret material capacities and source-defined experiment slots.
args:
  clauses: SourceClause[]
returns: StockIndex
---
1. Own the StockIndex inventory contract: original stock, offered ExperimentPlans, source consumption
   rules and missing source roles. Read original living-material identities and available quantities.
   Separate a plant's pollen,
   flower, leaf and already archived tray when the source gives them distinct capacities.
2. For each proposed experiment, bind its original experimentId and one-use slotId, kind, subject,
   oriented parent IDs, assayId, exact rational cost and required material amounts.
3. Interpret whether consumed material is restored, reusable or branch-local under the source rules.
   A hypothetical preview is not an executed cross and does not reduce stock in sibling alternatives.
4. Keep an invalid assay experiment in the inventory with its source identity. The planning judgment
   can reject it for a witnessed reason; the reader must not silently erase a supplied alternative.
5. Preserve missing costs or material roles as unresolved source obligations. Exact integer/nonnegative
   checks validate quantities; they do not decide which biological material an experiment requires.
6. Return stock, actual experiment plans, consumption/slot rules and unresolved source issues.
   Offered costs/materials are recorded inputs. predictExperiments later supplies answer partitions,
   and planPolicy derives adaptive choices and accumulated costs from these actual inputs. Generated
   experiment aliases cannot create additional source capacity or permit repeating a spent slot.
