---
description: Recover source ownership and the intended measurement question.
args:
  request: ProtocolRequest
  input: ProtocolInput
uses: [readEvidence]
returns: Intake
---
1. Confirm request.caseId and input.caseId agree. Keep a mismatch explicit rather than joining trials
   from different source-world groups merely because unit identifiers happen to match.
2. Call readEvidence(input.documents). Name its actual returned index evidenceIndex. Preserve its
   accepted definitions, per-unit unresolved/failed results and original source ownership.
3. Call interpretQuestion(request, evidenceIndex). Recover target population, event anchor, window
   and outcome proposition separately; preserve hard restrictions on edits and admissible interventions.
4. Check baseline and each probe seed declare unique unit and reading identities. A reading references
   a unit in its own trial; shared names across trials identify counterparts, not simultaneous observations.
5. Keep probeSeed.allowed and changedPremiseIds as visible authorization claims to assess downstream.
   The label 'population probe' does not itself prove that a seed isolates population meaning.
6. Retain every input document and its hash in evidenceIndex.documents. Semantic clauses carry exact witnesses
   back to those originals rather than only a normalized summary that discards provenance.
7. Return Intake with input, index=evidenceIndex and the actual question. Unclassified or conflicting definitions remain unresolved and
   cannot become default inclusion or default time zero simply to make an executable plan available.
8. This stage inventories authority. Later lineage resolves which definition each implementation actually
   used; do not assume the implementation named modern follows the controlling modern protocol.
