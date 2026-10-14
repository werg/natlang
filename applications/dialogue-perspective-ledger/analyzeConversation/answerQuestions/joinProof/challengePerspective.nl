---
description: Locally challenge a proof for quotation, privacy and temporal leakage.
args:
  focus: QueryFocus
  proof: ProofGraph
  source: DialogueSource
  scoped: ScopedConversation
  evolution: PerspectiveEvolution
returns: PerspectiveChallenge
---
1. Capture the actual focus, graph, supporting original spans and actual actor view/time. Construct a
   scoped NL PerspectiveChallenge literal and directly invoke it on this candidate proof, not a gold answer.
2. Its first local instruction checks every source/availability path belongs to the named participant
   at the requested time. Private notes of another actor and future adopted corrections are inadmissible.
3. Then check original-speaker and quote-frame anchoring. A nested quoted promise is attributable to
   its original issuer; a reporter-only act is not a new promise even if its substring contains will.
4. Challenge each knowledge path for both actual endorsement and available verified corroboration.
   Hearing, doubt or source-correct truth alone is insufficient under the explicit model.
5. Challenge commitment proofs for independent actual prerequisites, exact acknowledgment version and
   scope of any alleged retraction. Source absence supported by a complete log differs from bare silence.
6. Return actual passed/null/false, unsupported generated proof IDs and nonempty supporting witnesses.
   No child silently repairs the graph; the parent retains actual failures for query and report audits.
