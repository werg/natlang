---
description: Build and traverse the finite semantic dependency graph for revision.
args:
  analysis: Analysis
returns: ImpactResult
---
1. Inventory obligations for the requested change, all hard preservation rules, dialogue licenses,
   physical transitions, action motivations, protected uncertainties and meaningful callbacks.
   Use source-declared obligation IDs and required coverage conventions when present; otherwise give
   stable derived IDs with explicit provenance. Preserve scene membership, meaning and exact witnesses.
2. Build MeaningDependencies from premise fact or obligation IDs to consuming obligations. A
   knowledge premise supports a line; a custody premise supports a physical action; a retained
   experience supports a callback. Keep relation kinds and necessity judgments explicit. Initialize affected:null until the scoped
   delta judgment is performed; each link retains both necessary and affected as typed nullable fields.
3. Include all relevant inventoried obligations before iteration. Discovering a link later may add
   an edge between these nodes, but cannot secretly grow the node universe during measured closure.
4. Seed affectedNodeIds with obligations whose premise is removed or changed by the requested delta.
   Include the source-required guardrail obligations in their affected component even when their
   protected premise is unchanged: candidate generation can violate them and must check preservation.
   Initialize pendingNodeIds to every node ID and expandedNodeIds empty and localDecisions empty. Carry source unknowns.
5. Invoke closureStep.iterateOn(frontier, analysis) with measure pendingNodeIds.length and terminate
   when pendingNodeIds is empty. Consume the actual returned graph judgments and expanded state.
6. From the judged necessary links, compute affected closure over the finite node set. Add only new
   affected IDs until no new ID can be reached. Links with necessary:null or affected:null remain possible exposure: include
   their connected consumers for unresolved review and preserve the witnessed missing premise.
7. Form connected affected components using dependencies plus shared protected rules. Components
   that share an edit path or a necessary premise must be settled together, not independently patched.
8. Return affected obligations, components, judged links, final frontier localDecisions and unaffectedSceneIds. A scene with no
   changed semantic obligation can remain untouched even if it belongs to the same act directory.
9. Physical, epistemic and callback obligations can meet at one component. Keep each distinct in
   coverage so one plausible new utterance cannot masquerade as satisfaction of the whole graph.
