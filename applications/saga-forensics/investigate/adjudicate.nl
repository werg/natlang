---
description: Reduce distinct effect candidates into a ledger of committed and bounded-unknown outcomes.
args:
  index: EvidenceIndex
  correlation: Correlation
returns: EffectLedger
---
1. Build one EffectCandidate per explicitly named committed effect ID. Gather all linked attempts and all atoms
   reporting that effect, including repeated responses. Use the effect ID as its ledgerId.
2. For each nonrejected attempt group with no established effect, create one candidate using a visible attempt
   ID as ledgerId and null effectId. Keep its uncertainty; a future possible effect name is not a current commit.
3. Include separate deposit and refund candidates even when they belong to the same order and tenant.
   Candidates are business effects, not one ledger entry per file or per acknowledgment.
4. Initialize LedgerState with the finite pending candidate list, empty effects, explicit rejected attempts,
   inherited correlation conflicts, index and correlation. Call advanceLedger.iterateOn(initialState) with
   measure state.pending.length and stop when pending is empty. Each step removes exactly one candidate.
5. Consume the final state's effects and conflicts. Every candidate contributes an effect or a witnessed reason
   it was absent/rejected. No pending candidate may disappear merely because its outcome was unknown.
6. Return EffectLedger without the iteration context. Conservation and cancellation policy are later stages;
   this ledger establishes what the evidence permits, including a committed-or-absent refund possibility.
