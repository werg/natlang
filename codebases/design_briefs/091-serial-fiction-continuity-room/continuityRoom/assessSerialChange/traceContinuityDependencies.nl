---
description: Trace which later scenes depend on altered facts, knowledge exposures, motivations or open promises.
args:{edition:SerialEdition;proposal:SerialScene;canon:CanonAssertion[];knowledge:KnowledgeEvent[];obligations:ArcObligation[]}
returns:{changedFactIds:string[];knowledgeEventIds:string[];obligationIds:string[];dependentSceneIds:string[];dependencyPaths:string[][];uncertainLinks:string[]}
---
Start from the proposal's asserted facts, explicitly revealed secrets, character actions and consequences. Match each seed only to canon assertions, knowledge events and open obligations with exact subject or identifier lineage. Follow later scene references transitively, preserving each dependency path and the originating scene. A matching theme or character name without a shared fact, exposure or obligation is not a dependency. Return uncertain links separately when the edition lacks a reference needed to determine impact; do not mark descendants unaffected until their source lineage is checked.
