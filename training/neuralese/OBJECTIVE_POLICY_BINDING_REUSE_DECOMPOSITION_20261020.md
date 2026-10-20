# Reuse identical policy bindings within one continuation verification

The supervisor's existing continuation policy preserves source classes, held data,
objective authority and earned training state. This change only removes repeated
filesystem authentication of the same seven-file policy map during one call.

1. A crisp call-local verifier authenticates every reference in each distinct
   complete policy map with the existing strict classifier binding mechanism.
2. Identical maps reuse that verified digest while checking source/index entries.
   Paths, SHA values, optional byte declarations and all other reference fields
   remain part of the map identity; equivalent content at another path is distinct.
3. Before granting continuation, authenticate every cached map again. A mutation
   during verification fails; no cache or trust survives into the next call.
4. Existing classifier identity, seven intrinsic policy files, objective adapter,
   row classes, source inventories, withdrawals and saved-state rules are unchanged.
   This metadata optimization grants no training exposure or numerical qualification.
