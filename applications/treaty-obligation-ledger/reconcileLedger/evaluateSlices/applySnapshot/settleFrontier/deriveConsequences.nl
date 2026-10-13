---
description: Materialize supported or unresolved proof edges from one applicability judgment.
args:
  application: Applicability
  state: ClosureState
returns: ProofDelta
---
1. Find the application's norm and source authority. If applies=false, return no positive proof and
   retain its exclusion reason through the application; a false prerequisite cannot support a consequence.
2. For an applicable rate norm, return a SelectedTerm for the target norm and business key. Keep
   its amount, sourceApplicationIds and exact amendment/event premises. Resolve competing amendments
   by source authority and effective scope. Preserve a null amount for open applicability or unresolved
   competing terms; such a present term blocks fallback to a source base amount.
3. For a require or permit norm, consume the actual state.selectedTerms for this target/key before
   constructing its Justification. Use the selected amount when present, otherwise the source base amount.
   Retain the selected term's amendment premises on that proof. Set supported=true only for established
   prerequisites and supported=null for open ones. A rate application creates a term, not a duplicate duty.
4. For an exemption, reservation or closure, locate the precise target proofs by norm relation and
   business roles. Return defeatCandidates instead of deleting all duties involving the same party.
5. A certified-rescue permission creates an independently justified passage proof whose scope is
   protected by the source exception. A closure can defeat the ordinary-permission proof while leaving
   that protected proof intact. Keep these proof scopes distinct; do not erase the closure instrument.
6. Every proof records clause, fact, authority and prior-proof references with typed PremiseRef tags.
   Include evidence-availability premises whenever a late document controls the result.
7. Return new proofs/candidates and unresolved interpretation. Proposed IDs are stable within this
   result; their spelling supplies no fact and never replaces a source-supported premise.
