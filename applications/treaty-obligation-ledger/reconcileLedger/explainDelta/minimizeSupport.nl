---
description: Derive inclusion-minimal newly available evidence for one observed certificate change.
args:
  delta: CertificateDelta
  before: SliceLedger
  after: SliceLedger
  archive: Archive
returns: MinimalSupport
---
1. Trace the changed entry's after-proof cone to source facts and their availability assertions.
   Partition premises into newly available between snapshots and already available background.
2. Initialize SupportPassState with the complete new candidateIds, empty removedIds/reasons,
   backgroundPremises, actual before/after ledgers, delta, archive and settled=false.
3. Include the substantive certification and its source availability assertion separately. Derived
   availableFrom fields depend on availabilityPremiseIds and must be recomputed if those premises are removed.
4. Invoke pruneSupportPass.iterateOn(initialState) until settled is true, with candidateIds.length
   as the structural measure. Every continuing pass removes at least one candidate; a pass with no removal settles.
5. Consume the actual returned candidateIds. If unresolved is empty, report inclusion-minimal; if
   any decisive sufficiency judgment remains unknown, preserve minimality unresolved and its missing premise.
6. Return MinimalSupport with retained newPremiseIds, unchanged backgroundPremises, removedCandidates,
   source-grounded reasons and exact witnesses. Inclusion minimality is not minimum cardinality.
