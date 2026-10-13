# Sol semantic program slate

100 distinct program operations across 20 domains. Canonical machine-readable catalog: `training/semantic_stack/sol_programs.json`. Each entry defines typed directory inputs, four named NL stages, an ad hoc runtime NL literal, structural iteration, source choice, and exact plus independent teacher checks.

All entries initially remain source candidates. Seed bundles are not claimed sizable execution gold. Expanded fixtures, exact source replay, semantic review, about 48 live samples, full trajectories, and explicit split/admission checks are required before full collection and training.

Crisp helpers provide immutable directory snapshots, finite frontier mechanics, source/span and trace verification, and exact arithmetic. Meaning, policy, matching, uncertainty, and repair decisions belong to the NL stages.

## journalism

### sol-001 — Correction ripple ledger

**Typed inputs:** `ClaimDossier = {files:DirectorySnapshot, records:Claim{id,speaker,utterance,date,origin,access}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Propagate a corrected premise only to dependent claims; preserve direct observations.

**NL decomposition:** Trace claims to earliest source. → Build explicit borrowing graph. → Resolve corrected proposition scope. → Traverse dependent branches into minimal correction notices.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare borrowed propositions with the captured correction and decide whether downstream meaning still follows.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional interviews, corrections, briefings and documentary inquiries. Positive witness: A reporter copies a draft asserting permanent closure; a signed correction says the closure was only for repairs. Contrast: A witness independently saw locked gates that afternoon without asserting permanent closure.

**Output:** `CorrectionLedger{correctedClaims,unaffectedClaims,edges}`. **Exact:** Every corrected claim has a dependency path; direct observations remain unchanged. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-002 — Witness vantage reconstruction

**Typed inputs:** `ClaimDossier = {files:DirectorySnapshot, records:Claim{id,speaker,utterance,date,origin,access}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Reconcile accounts by perception access and timing rather than choosing the most confident witness.

**NL decomposition:** Extract access and time anchors. → Align shared perceived events. → Intersect compatible partial orders. → Separate perspective gaps from factual conflicts.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Use captured vantage cards to test whether both accounts can hold without changing either.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional interviews, corrections, briefings and documentary inquiries. Positive witness: Upstairs smoke precedes the alarm; the lobby witness hears the alarm before reaching a window. Contrast: A rumor repeats the smoke claim without a personal vantage.

**Output:** `VantageMap{eventIntervals,compatibleAccounts,conflicts}`. **Exact:** All inferred orders have stated temporal witnesses; perception gaps do not create events. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-003 — Quotation meaning drift

**Typed inputs:** `ClaimDossier = {files:DirectorySnapshot, records:Claim{id,speaker,utterance,date,origin,access}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Audit quotation edits for changed referents, conditions, or certainty.

**NL decomposition:** Pair quotations with interview context. → Recover scope and modality. → Align original and edited propositions. → Draft source-linked corrections by meaning change.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Read captured adjacent turns to identify the precise condition or target the edit changes.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional interviews, corrections, briefings and documentary inquiries. Positive witness: Interview: the plan may help if buses run later; article: the speaker promised it will help everyone. Contrast: A shortened anecdote preserves the conditional conclusion.

**Output:** `QuoteAudit{alignedSpans,meaningChanges,faithfulEdits}`. **Exact:** Every flag cites both spans; formatting differences alone are excluded. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-004 — Anonymous-source independence

**Typed inputs:** `ClaimDossier = {files:DirectorySnapshot, records:Claim{id,speaker,utterance,date,origin,access}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Count independent corroboration rather than retellings of one briefing.

**NL decomposition:** Extract access stories and shared details. → Construct evidence-qualified origin links. → Partition dependence components. → Report support counts and unknown origins.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge whether a shared detail establishes common origin or only shared subject matter.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional interviews, corrections, briefings and documentary inquiries. Positive witness: Three contacts quote an unusual phrase from the same private slide deck. Contrast: Two contacts independently observed separate shifts supporting the claim.

**Output:** `SourceClusters{lineages,independentSupport,unknownOrigins}`. **Exact:** Unknown origin is not automatic independence; counts equal justified lineage components. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-005 — Missing-link inquiry

**Typed inputs:** `ClaimDossier = {files:DirectorySnapshot, records:Claim{id,speaker,utterance,date,origin,access}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Ask questions that separate competing explanations of unresolved evidence.

**NL decomposition:** Construct rival explanations. → Locate divergent predictions. → Group gaps by resolving source. → Drain gap frontier into discriminating questions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Draft a question whose alternative answers separate the captured explanations.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional interviews, corrections, briefings and documentary inquiries. Positive witness: The permit predates the memo said to authorize it; ask who authorized the earlier action. Contrast: A biographical detail does not separate the authorization accounts.

**Output:** `InquiryPlan{accounts,questions,evidenceTargets}`. **Exact:** Every question identifies an unresolved proposition and discriminating answers. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## field-linguistics

### sol-006 — Deictic center tracker

**Typed inputs:** `LanguageNotebook = {files:DirectorySnapshot, records:Utterance{id,speaker,form,gloss,situation,elicitation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Resolve demonstratives across quotations using consultant-defined speaker-relative anchors.

**NL decomposition:** Read supplied deictic rules. → Segment nested discourse. → Push and pop discourse centers. → Resolve uses against active centers.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret a deictic use against the captured speaker-location stack.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented language consultant explanations, conversation and elicitation notebooks. Positive witness: A quoted speaker's near means near that speaker even when a distant narrator reports it. Contrast: Near in the narrator's unquoted sentence uses the narrator's location.

**Output:** `DeicticTrace{anchors,referents,unknownUses}`. **Exact:** Anchor changes occur only at discourse boundaries; referents belong to the stated scene. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-007 — Elicitation artifact separation

**Typed inputs:** `LanguageNotebook = {files:DirectorySnapshot, records:Utterance{id,speaker,form,gloss,situation,elicitation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate spontaneous grammatical evidence from forms induced by leading questions.

**NL decomposition:** Classify elicitation histories. → Group scene-matched forms. → Compare contamination-sensitive contrasts. → Design nonleading follow-up scenes.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Determine whether a matched pair differs in meaning or only question wording.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented language consultant explanations, conversation and elicitation notebooks. Positive witness: A consultant accepts a questioned phrase but spontaneously describes the same scene differently. Contrast: A spontaneous retelling independently uses the questioned construction.

**Output:** `ContrastPlan{rules,artifacts,pairs,followupScenes}`. **Exact:** Each contrast has matched context; follow-up scenes omit the tested form. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-008 — Morphological exception notebook

**Typed inputs:** `LanguageNotebook = {files:DirectorySnapshot, records:Utterance{id,speaker,form,gloss,situation,elicitation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Construct an invented inflection hierarchy with narrow contextual exceptions.

**NL decomposition:** Extract form-meaning pairs. → Order broad and narrow scopes. → Attach exceptions to smallest supported scope. → Apply hierarchy and expose unlicensed predictions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Choose the narrowest supplied context licensing an exception.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented language consultant explanations, conversation and elicitation notebooks. Positive witness: Animate nouns take nu except respectful-address kin terms take li. Contrast: A place name has li only in a quotation with no stated kin relation.

**Output:** `MorphologyNotebook{rules,exceptionScopes,unknowns}`. **Exact:** Rules reproduce documented forms; exceptions cannot widen beyond stated evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-009 — Repair-sequence meaning finder

**Typed inputs:** `LanguageNotebook = {files:DirectorySnapshot, records:Utterance{id,speaker,form,gloss,situation,elicitation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Infer semantic distinctions from conversational repair while separating acoustic repetition.

**NL decomposition:** Pair initiations with responses. → Recover misunderstood propositions. → Contrast forms under supplied glosses. → Cluster repaired semantic dimensions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Use captured interlocutor questions to distinguish meaning repair from sound repair.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented language consultant explanations, conversation and elicitation notebooks. Positive witness: A speaker changes direct-hearsay form after being asked who told them. Contrast: A speaker repeats the same form louder after a volume request.

**Output:** `RepairAnalysis{pairs,dimensions,acousticRepairs}`. **Exact:** Semantic clusters contain explicit contrasts; louder repetition stays separate. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-010 — Lexical gap circumlocution

**Typed inputs:** `LanguageNotebook = {files:DirectorySnapshot, records:Utterance{id,speaker,form,gloss,situation,elicitation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare circumlocutions without inventing an unattested translation.

**NL decomposition:** Decompose contextual descriptions. → Compare referents and functions. → Find shared feature coverage. → State gap hypothesis and open lexical questions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Check whether a description covers the captured target function or an incidental attribute.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented language consultant explanations, conversation and elicitation notebooks. Positive witness: Two speakers describe a vessel that keeps liquids cool with different multiword phrases. Contrast: A color description does not cover the cooling function.

**Output:** `LexicalGap{features,descriptions,unobservedLexeme}`. **Exact:** Unobserved is not nonexistent; features require consultant evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## maintenance

### sol-011 — Procedure precondition repair

**Typed inputs:** `MaintenanceCase = {files:DirectorySnapshot, records:ManualNote{id,component,instruction,observation,precondition,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Reorder a fictional manual by explicit prerequisite states.

**NL decomposition:** Extract required component states. → Link state-establishing steps. → Detect cycles and missing prerequisites. → Topologically assemble a minimal patch.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge whether two steps have a mandatory order under captured state requirements.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional device manuals, service diaries and explicitly supplied safety conditions. Positive witness: Replacement requires a drained reservoir, but the revised drain step occurs later. Contrast: A labeling step has no drainage dependency.

**Output:** `ProcedurePatch{graph,order,conflicts}`. **Exact:** Patched order satisfies all stated preconditions; missing safety facts remain unresolved. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-012 — Fault symptom differential

**Typed inputs:** `MaintenanceCase = {files:DirectorySnapshot, records:ManualNote{id,component,instruction,observation,precondition,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate faults sharing symptoms using described discriminating observations.

**NL decomposition:** Read symptom and exclusion descriptions. → Instantiate compatible faults. → Eliminate observed contradictions. → Choose tests separating survivors.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Explain which possible test outcomes remove which captured faults.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional device manuals, service diaries and explicitly supplied safety conditions. Positive witness: Intake blockage and worn seal reduce flow; only seal wear leaves wet housing. Contrast: Low flow alone identifies neither fault uniquely.

**Output:** `FaultInquiry{survivors,tests,eliminations}`. **Exact:** Elimination needs contradictory evidence; missing observations are not negative tests. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-013 — Service history reset detector

**Typed inputs:** `MaintenanceCase = {files:DirectorySnapshot, records:ManualNote{id,component,instruction,observation,precondition,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Track which clues survive part replacement or recalibration.

**NL decomposition:** Group events by component and time. → Recognize reset scopes. → Fold clues into component epochs. → Publish current and retired evidence.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Apply a service event only to clues inside its captured component scope.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional device manuals, service diaries and explicitly supplied safety conditions. Positive witness: A seal replacement separates an earlier leak from the new pressure observation. Contrast: Sensor calibration does not erase a mechanical rattle.

**Output:** `ServiceState{epochs,activeClues,retiredClues}`. **Exact:** Retired clues are covered by reset scope; unrelated clues persist. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-014 — Manual revision meaning merge

**Typed inputs:** `MaintenanceCase = {files:DirectorySnapshot, records:ManualNote{id,component,instruction,observation,precondition,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Merge revisions while checking interacting changed assumptions.

**NL decomposition:** Align base and revised units. → Extract changed conditions. → Compose compatible edits and test interactions. → Return snapshot and unresolved conflicts.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Check whether captured assumptions make two edits compatible.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional device manuals, service diaries and explicitly supplied safety conditions. Positive witness: One edit changes gasket material; another changes torque for that new material. Contrast: Identical heading renames do not change procedure.

**Output:** `ManualMerge{instructions,assumptions,conflicts}`. **Exact:** Merged instructions retain revision provenance; incompatible requirements stay explicit. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-015 — Inspection route explanation

**Typed inputs:** `MaintenanceCase = {files:DirectorySnapshot, records:ManualNote{id,component,instruction,observation,precondition,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Choose nonredundant inspections covering described failure modes.

**NL decomposition:** Extract modes and detection capabilities. → Build semantic coverage sets. → Compare routes by uncovered modes. → Order selected checks with coverage witnesses.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Decide whether a check detects the captured failure mode under supplied conditions.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional device manuals, service diaries and explicitly supplied safety conditions. Positive witness: A seam check detects cracking and seepage; vibration reveals looseness only. Contrast: Two differently named checks inspect the same seam appearance.

**Output:** `InspectionRoute{checks,witnesses,deferredModes}`. **Exact:** Coverage needs stated capabilities; unsupported modes remain deferred. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## peer-review

### sol-016 — Claim evidence ladder

**Typed inputs:** `ReviewPacket = {files:DirectorySnapshot, records:ReviewItem{id,paper,section,claim,evidence,reviewer}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate measured results, interpretations, and extrapolations.

**NL decomposition:** Extract populations and certainty. → Link claims to evidence. → Order scope relationships. → Locate added inferential assumptions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Identify the new assumption separating a claim from captured results.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original miniature papers, appendices, referee comments and author responses. Positive witness: Three tested settings improve; the abstract asserts improvement in every setting. Contrast: Repeating the measured scope is directly supported.

**Output:** `ClaimLadder{claims,links,expansions}`. **Exact:** Every expansion names its added scope or certainty; measured results are preserved. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-017 — Referee contradiction resolver

**Typed inputs:** `ReviewPacket = {files:DirectorySnapshot, records:ReviewItem{id,paper,section,claim,evidence,reviewer}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Resolve apparent conflicting requests by target and rationale.

**NL decomposition:** Parse targets and rationales. → Align comments to passages. → Compose disjoint-scope requests. → Expose same-target conflicts.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Test whether one scoped revision can satisfy both captured rationales.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original miniature papers, appendices, referee comments and author responses. Positive witness: One referee wants fewer result anecdotes; another wants context in limitations. Contrast: Opposite definitions of one central term in one section truly conflict.

**Output:** `ResponseMatrix{compatibleEdits,conflicts,questions}`. **Exact:** Compatible edits preserve both reasons; true contradictions are not hidden. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-018 — Replication boundary dossier

**Typed inputs:** `ReviewPacket = {files:DirectorySnapshot, records:ReviewItem{id,paper,section,claim,evidence,reviewer}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Explain procedural differences without claiming unmeasured causal proof.

**NL decomposition:** Align steps by purpose. → Compare conditions. → Trace changes to methodological rationales. → Report conditional consequences.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Distinguish renamed steps from changed conditions using captured methods.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original miniature papers, appendices, referee comments and author responses. Positive witness: A replication omits the warm-up said to change participant familiarity. Contrast: Renaming the same warm-up does not change procedure.

**Output:** `ReplicationBoundary{matches,differences,hypotheses}`. **Exact:** Differences cite both methods; consequences remain hypotheses without evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-019 — Response fulfillment audit

**Typed inputs:** `ReviewPacket = {files:DirectorySnapshot, records:ReviewItem{id,paper,section,claim,evidence,reviewer}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Verify promised revisions against actual manuscript passages.

**NL decomposition:** Parse requested outcomes. → Align author promises. → Locate corresponding edits. → Describe residual substantive gaps.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Determine which captured request outcome the edit actually supplies.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original miniature papers, appendices, referee comments and author responses. Positive witness: A reply promises confound discussion and the new limitations paragraph supplies it. Contrast: A fixed assertion plus unrelated citation does not address the confound.

**Output:** `FulfillmentLedger{fulfilled,partial,unsupported}`. **Exact:** Fulfilled status needs a matching revision span; author assurance alone is insufficient. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-020 — Reviewer expertise routing

**Typed inputs:** `ReviewPacket = {files:DirectorySnapshot, records:ReviewItem{id,paper,section,claim,evidence,reviewer}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Route questions using demonstrated expertise rather than titles.

**NL decomposition:** Extract question knowledge needs. → Read supplied expertise. → Build justified compatibility edges. → Assign or preserve gaps.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Match demonstrated expertise to captured question substance.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original miniature papers, appendices, referee comments and author responses. Positive witness: Prior substantive comments explain sampling bias relevant to the open question. Contrast: A shared department title does not show knowledge of that technique.

**Output:** `ReviewerRoute{assignments,evidence,unmatched}`. **Exact:** Assignments require visible expertise; unknowns remain unmatched. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## observing-logs

### sol-021 — Instrument versus sky change

**Typed inputs:** `ObservingNight = {files:DirectorySnapshot, records:Observation{id,target,time,conditions,note,calibration}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate apparent target change from instrument or conditions change.

**NL decomposition:** Extract target and reference states. → Segment instrument transitions. → Compare simultaneous changes. → Write bounded interpretations.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge target specificity using captured comparison and calibration states.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional observing diaries and instrument capabilities supplied entirely in text. Positive witness: The target and comparison targets dim while the calibration lamp becomes unstable. Contrast: Target-only dimming with stable supplied references supports specificity.

**Output:** `ChangeReport{segments,targetChanges,confounds}`. **Exact:** Specificity cites stable references; confounds preserve uncertainty. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-022 — Observer vocabulary alignment

**Typed inputs:** `ObservingNight = {files:DirectorySnapshot, records:Observation{id,target,time,conditions,note,calibration}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Align qualitative adjectives through observer-specific calibration examples.

**NL decomposition:** Collect descriptors and examples. → Build local ordinal scales. → Anchor through shared references. → Translate with uncertainty intervals.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Map one descriptor through captured observer reference examples.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional observing diaries and instrument capabilities supplied entirely in text. Positive witness: One observer calls barely split doubles sharp; another reserves sharp for clear separation. Contrast: Identical adjectives without shared examples do not establish equality.

**Output:** `VocabularyCrosswalk{scales,mappings,unknowns}`. **Exact:** Equivalence requires shared anchors; unanchored adjectives remain uncertain. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-023 — Night interruption resumption

**Typed inputs:** `ObservingNight = {files:DirectorySnapshot, records:Observation{id,target,time,conditions,note,calibration}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Recover unfinished tasks and reusable observations after interruptions.

**NL decomposition:** Parse observation dependencies. → Fold completion and interruption. → Check reuse conditions. → Resume outstanding frontier.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Determine whether captured condition changes permit reuse of a partial observation.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional observing diaries and instrument capabilities supplied entirely in text. Positive witness: Blue comparison completes before clouds; only target exposure restarts; red comparison remains unfinished. Contrast: Resumed work does not prove every planned observation completed.

**Output:** `NightPlan{completed,reusable,outstanding}`. **Exact:** Completion needs evidence; vague resumption cannot close dependencies. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-024 — Transient account association

**Typed inputs:** `ObservingNight = {files:DirectorySnapshot, records:Observation{id,target,time,conditions,note,calibration}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Associate reports of one fictional transient using evolution and relative position.

**NL decomposition:** Extract position and sequence. → Compare identity constraints. → Build qualified association edges. → Cluster supported identities.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge event identity using captured location and evolution constraints.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional observing diaries and instrument capabilities supplied entirely in text. Positive witness: Two nicknames share target-relative position and fading sequence. Contrast: A simultaneous flash in another region has incompatible evolution.

**Output:** `TransientCatalog{clusters,uncertainLinks,separateEvents}`. **Exact:** Clusters require compatible anchors; simultaneity alone is insufficient. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-025 — Observation request usefulness

**Typed inputs:** `ObservingNight = {files:DirectorySnapshot, records:Observation{id,target,time,conditions,note,calibration}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Design observations that separate surviving interpretations.

**NL decomposition:** Enumerate consistent interpretations. → Derive contrasting predictions. → Match feasible observations. → Order by newly resolved distinctions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Draft a request with divergent captured predicted outcomes.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional observing diaries and instrument capabilities supplied entirely in text. Positive witness: A confound explanation predicts comparison-target variation while target-only change does not. Contrast: Repeating a decisive description under identical conditions adds no distinction.

**Output:** `FollowupPlan{hypotheses,observations,conditions}`. **Exact:** Requests use supplied instrument capabilities and identify discriminating outcomes. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## family-correspondence

### sol-026 — Promise lifecycle ledger

**Typed inputs:** `LetterBundle = {files:DirectorySnapshot, records:Letter{id,sender,recipient,date,text,enclosure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Track commitments, contingencies, renegotiations and completion across letters.

**NL decomposition:** Distinguish promises from hopes. → Link conditions and replies. → Fold amendment and completion. → Report unresolved triggers.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Classify a letter's effect on captured earlier commitments.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional family letters, memoirs, address changes and enclosure acknowledgments. Positive witness: Books are promised if an address arrives; later letters supply address and confirm dispatch. Contrast: Wistful thoughts about visiting do not commit a visit.

**Output:** `PromiseLedger{active,fulfilled,withdrawn,conditions}`. **Exact:** Every obligation has a commitment span; completion needs specific evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-027 — Indirect apology interpretation

**Typed inputs:** `LetterBundle = {files:DirectorySnapshot, records:Letter{id,sender,recipient,date,text,enclosure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Find repair attempts without relying on apology vocabulary.

**NL decomposition:** Read paired conflict accounts. → Extract repair speech acts. → Compare replies with grievance. → Interpret grounded intentions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess a reply against captured grievance and recipient choice.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional family letters, memoirs, address changes and enclosure acknowledgments. Positive witness: The writer acknowledges hurt, takes responsibility, and permits refusal of contact. Contrast: Sorry you are upset plus demanded forgiveness and blame does not perform the same repair.

**Output:** `ReconciliationMap{repairs,pressure,unknownIntent}`. **Exact:** Intent remains provisional and source-grounded; polite words alone are insufficient. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-028 — Address identity puzzle

**Typed inputs:** `LetterBundle = {files:DirectorySnapshot, records:Letter{id,sender,recipient,date,text,enclosure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Resolve aliases without merging relatives sharing names.

**NL decomposition:** Extract names kin and moves. → Build identity constraints. → Propagate explicit equivalences. → Retain ambiguous mentions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Weigh a possible alias against captured kin and address constraints.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional family letters, memoirs, address changes and enclosure acknowledgments. Positive witness: Aunt Kit identifies her full name and the same move and sibling. Contrast: Cousin Kit has incompatible parentage and address history.

**Output:** `IdentityRegister{aliases,addressEpochs,unknowns}`. **Exact:** Resolved groups contain no incompatible simultaneous identities. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-029 — Enclosure journey reconstruction

**Typed inputs:** `LetterBundle = {files:DirectorySnapshot, records:Letter{id,sender,recipient,date,text,enclosure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Trace original objects separately from copies and replacements.

**NL decomposition:** Extract artifact identity cues. → Link transfer events. → Traverse custody paths. → Explain missing handoffs.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Decide whether a transfer concerns the captured original or a separate copy.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional family letters, memoirs, address changes and enclosure acknowledgments. Positive witness: A photograph is forwarded; a later copy is sent because the original never arrived. Contrast: Mentioning a photograph does not establish the same physical print.

**Output:** `EnclosureRoute{paths,copies,gaps}`. **Exact:** Transfer edges have explicit evidence; copy receipt cannot close original loss. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-030 — Shared memory disagreement

**Typed inputs:** `LetterBundle = {files:DirectorySnapshot, records:Letter{id,sender,recipient,date,text,enclosure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Preserve perspective-compatible recollections and factual contradictions.

**NL decomposition:** Locate common event anchors. → Recover teller access. → Separate perspective gaps from impossible facts. → Represent unresolved branches.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Test whether captured access differences explain two memories.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional family letters, memoirs, address changes and enclosure acknowledgments. Positive witness: Children recall sudden departure while adults recall preparations hidden from them. Contrast: A borrowed memoir phrase is not another independent memory.

**Output:** `MemoryMap{sharedEvents,branches,conflicts,lineages}`. **Exact:** Perspective needs access evidence; source dependence remains explicit. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## craft-knowledge

### sol-031 — Tacit readiness cue ladder

**Typed inputs:** `WorkshopNotebook = {files:DirectorySnapshot, records:CraftNote{id,maker,material,step,cue,outcome}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Turn supplied sensory readiness descriptions into a staged guide without invented thresholds.

**NL decomposition:** Extract stages and cues. → Order transformations. → Link decisions to confirming observations. → Build environment-aware cue guide.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Evaluate a sample against captured stage-specific sensory cues.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original workshop notes and sample accounts with stated material behavior. Positive witness: Press after wet shine disappears and a lifted edge briefly holds. Contrast: Clock estimates are stated unreliable under humidity changes.

**Output:** `ReadinessGuide{stages,cues,checks,exceptions}`. **Exact:** Readiness cites supplied cues; numeric thresholds are not invented. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-032 — Material substitution boundaries

**Typed inputs:** `WorkshopNotebook = {files:DirectorySnapshot, records:CraftNote{id,maker,material,step,cue,outcome}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Assess substitutes by required functional properties rather than name or appearance.

**NL decomposition:** Extract functional requirements. → Read sample behavior. → Match evidence under conditions. → Explain compatibility and missing tests.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Check each captured required property, distinguishing absent from conflicting evidence.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original workshop notes and sample accounts with stated material behavior. Positive witness: The process needs wet tension retention and the sample diary shows the substitute retains it. Contrast: A similar-colored fiber relaxes when wet.

**Output:** `SubstitutionMatrix{requirements,compatible,unknowns}`. **Exact:** Compatibility has witnesses for all required properties. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-033 — Repair provenance signature

**Typed inputs:** `WorkshopNotebook = {files:DirectorySnapshot, records:CraftNote{id,maker,material,step,cue,outcome}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Recover partial repair chronology from overlap and intervention narratives.

**NL decomposition:** Extract overlaps. → Build before-after constraints. → Align documented interventions. → Return chronology with uncertain attribution.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Associate an intervention with captured layer evidence.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original workshop notes and sample accounts with stated material behavior. Positive witness: New adhesive covers an earlier stitch and matches the repair diary. Contrast: Stitch style alone cannot date or identify the repairer.

**Output:** `RepairLayers{order,hypotheses,unknownAttributions}`. **Exact:** Order edges require overlaps or dates; attribution needs more than stylistic similarity. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-034 — Technique transmission divergence

**Typed inputs:** `WorkshopNotebook = {files:DirectorySnapshot, records:CraftNote{id,maker,material,step,cue,outcome}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Distinguish reasoned adaptation from unexplained changes in teaching lineages.

**NL decomposition:** Align actions by purpose. → Recover contexts and reasons. → Construct teaching lineage. → Locate first unsupported change.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge altered instruction against captured material conditions.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original workshop notes and sample accounts with stated material behavior. Positive witness: An apprentice shortens soaking only for thinner material and explains why. Contrast: Different names for the same action are not procedural divergence.

**Output:** `TransmissionTree{core,adaptations,gaps}`. **Exact:** Adaptation needs a reason consistent with supplied behavior. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-035 — Failed sample recovery notebook

**Typed inputs:** `WorkshopNotebook = {files:DirectorySnapshot, records:CraftNote{id,maker,material,step,cue,outcome}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Propose discriminating retries from matched success and failure histories.

**NL decomposition:** Extract histories and outcomes. → Find matched contrasts. → Remove contradicted causes. → Design one-distinction retries.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Choose a retry against captured contrasts and predict separating outcomes.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original workshop notes and sample accounts with stated material behavior. Positive witness: Cracked samples were pressed before readiness cues; intact matched samples met them. Contrast: Both failed and intact samples use blue pigment.

**Output:** `RecoveryNotebook{causes,contrasts,retries}`. **Exact:** Retries isolate documented differences; correlation is not proof. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## travel-literature

### sol-036 — Route versus narrative order

**Typed inputs:** `TravelTextBundle = {files:DirectorySnapshot, records:Passage{id,narrator,place,time,text,genre}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Recover actual chronology through flashbacks and imagined returns.

**NL decomposition:** Segment actual and imagined episodes. → Extract movement anchors. → Assemble partial actual route. → Contrast narration and chronology.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Classify an episode using captured tense and framing.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional journeys, route descriptions and retrospective travel passages. Positive witness: Port opening recalls a mountain crossing then the road linking mountain to port. Contrast: An imagined future return is not a completed visit.

**Output:** `JourneyGraph{actualStops,timeEdges,imaginedStops}`. **Exact:** Actual visits need assertions; imagination does not create route nodes. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-037 — Place nickname crosswalk

**Typed inputs:** `TravelTextBundle = {files:DirectorySnapshot, records:Passage{id,narrator,place,time,text,genre}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Resolve evocative names through landmarks and route relations.

**NL decomposition:** Extract names and landmarks. → Link explicit equivalences. → Propagate along route. → Retain ambiguous descriptions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess name links against captured landmarks and route position.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional journeys, route descriptions and retrospective travel passages. Positive witness: Silver mouth is explicitly identified as the inlet below the ruined tower. Contrast: A silver-looking inlet beyond another headland is distinct.

**Output:** `PlaceIndex{names,relations,unknownMentions}`. **Exact:** Resolved places satisfy stated relations; adjective similarity does not merge them. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-038 — Host gaze inversion

**Typed inputs:** `TravelTextBundle = {files:DirectorySnapshot, records:Passage{id,narrator,place,time,text,genre}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare traveler and host meanings of the same encounter without importing stereotypes.

**NL decomposition:** Align encounter actions. → Extract local conventions. → Compare context-specific interpretations. → Write grounded alternative readings.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret an action through captured encounter-specific host explanations.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional journeys, route descriptions and retrospective travel passages. Positive witness: The traveler sees embarrassment; the host explains a pause reserved for elders. Contrast: A different encounter explicitly involves embarrassment.

**Output:** `EncounterReadings{actions,meanings,assumptions}`. **Exact:** Alternative readings cite local evidence, not universal cultural reversal. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-039 — Sensory motif itinerary

**Typed inputs:** `TravelTextBundle = {files:DirectorySnapshot, records:Passage{id,narrator,place,time,text,genre}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Trace changes in a recurring sensory motif's meaning.

**NL decomposition:** Locate motif candidates. → Interpret contextual roles. → Connect supported recurrences. → Describe changes along route.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare recurrences using captured intervening events.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional journeys, route descriptions and retrospective travel passages. Positive witness: Bells mark exclusion at a locked gate then welcome after the town opens. Contrast: An unrelated dream bell need not be the journey motif.

**Output:** `MotifItinerary{occurrences,meanings,transitions}`. **Exact:** Transitions cite occurrences and context; repeated words alone are insufficient. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-040 — Map legend mismatch critique

**Typed inputs:** `TravelTextBundle = {files:DirectorySnapshot, records:Passage{id,narrator,place,time,text,genre}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Find distinctions erased by the map categories relative to the narrated route.

**NL decomposition:** Read map legend meanings. → Align narrative segments. → Compare category and access scope. → Draft minimal legend changes.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess a map label against captured use and access evidence.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional journeys, route descriptions and retrospective travel passages. Positive witness: Road covers a public route and a ritual path explicitly closed to visitors. Contrast: Two differently named public routes legitimately share one category.

**Output:** `MapCritique{collapses,preserved,corrections}`. **Exact:** Corrections cite actual segments and narrative distinctions. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## film-continuity

### sol-041 — Prop state branch audit

**Typed inputs:** `ContinuityBundle = {files:DirectorySnapshot, records:ShotNote{id,scene,shot,time,description,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Track prop states through selected takes and story chronology.

**NL decomposition:** Extract prop transitions. → Separate selected and alternate takes. → Order story scenes. → Fold states and detect unsupported reversals.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge appearance against captured branch and prior state.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original screenplays, shot accounts, edit revisions and prop-state cards. Positive witness: A cup breaks in the selected take then appears intact with no replacement. Contrast: An intact cup in an earlier flashback is compatible.

**Output:** `PropAudit{timelines,takeBranches,conflicts}`. **Exact:** Discarded takes cannot change selected-branch states. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-042 — Character knowledge leakage

**Typed inputs:** `ContinuityBundle = {files:DirectorySnapshot, records:ShotNote{id,scene,shot,time,description,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Check dialogue against facts each character could have learned.

**NL decomposition:** Extract asserted facts. → Identify witness and recipient access. → Propagate explicit transfers. → Check speaker-local knowledge.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess a line against captured speaker acquisition paths.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original screenplays, shot accounts, edit revisions and prop-state cards. Positive witness: A character names a hiding place before the first information-transfer scene. Contrast: Audience knowledge alone does not inform the character.

**Output:** `KnowledgeAudit{acquisitionGraph,licensedLines,leaks}`. **Exact:** Every licensed fact has a prior access path. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-043 — Parallel edit synchronization

**Typed inputs:** `ContinuityBundle = {files:DirectorySnapshot, records:ShotNote{id,scene,shot,time,description,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Check intercut timelines using explicit shared event anchors.

**NL decomposition:** Recover thread orders. → Locate shared events. → Join explicit anchors. → Find minimal inconsistent cycles.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Decide whether two events share identity under captured dialogue and action.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original screenplays, shot accounts, edit revisions and prop-state cards. Positive witness: One thread places a call before dawn; the other answers it after later sunrise. Contrast: Alternating shots alone do not assert simultaneity.

**Output:** `ParallelTimeline{anchors,constraints,cycles}`. **Exact:** Only supported temporal edges create cycles; adjacency gives no time relation. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-044 — Unreliable flashback variants

**Typed inputs:** `ContinuityBundle = {files:DirectorySnapshot, records:ShotNote{id,scene,shot,time,description,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Distinguish intentional remembered variants from within-branch continuity errors.

**NL decomposition:** Read epistemic framing. → Assign shots to account branches. → Compare within and across branches. → Explain grounded differences.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret mismatch using captured framing and branch owner.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original screenplays, shot accounts, edit revisions and prop-state cards. Positive witness: Two characters' explicitly framed memories assign different meeting seats. Contrast: An objective scene changes seats between consecutive selected shots.

**Output:** `FlashbackMatrix{branches,intentionalDifferences,defects}`. **Exact:** Intentional difference requires explicit account framing. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-045 — Revision cascade planner

**Typed inputs:** `ContinuityBundle = {files:DirectorySnapshot, records:ShotNote{id,scene,shot,time,description,revision}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Trace consequences of changing a scene fact through knowledge, props, and callbacks.

**NL decomposition:** Parse fact delta. → Trace direct consumers. → Expand secondary dependencies. → Plan minimal repairs.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Test whether removed captured facts are necessary to a downstream unit.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original screenplays, shot accounts, edit revisions and prop-state cards. Positive witness: Removing letter delivery removes later knowledge and a seal callback. Contrast: Sharing a location without dependence does not require change.

**Output:** `RevisionCascade{affected,paths,safe,alternatives}`. **Exact:** Every affected unit has a dependency path, not thematic proximity. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## music-rehearsal

### sol-046 — Interpretive instruction ancestry

**Typed inputs:** `RehearsalBundle = {files:DirectorySnapshot, records:RehearsalNote{id,section,performer,passage,instruction,observation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Resolve artistic markings by explicit scope and supersession.

**NL decomposition:** Extract scope and reasons. → Order revisions. → Apply explicit local overrides. → Report current intentions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Resolve one passage using captured broad and narrow markings.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented score annotations, rehearsal attempts and explicit artistic intentions. Positive witness: General restraint is overridden by later intensity at one returning phrase. Contrast: A change elsewhere does not globally override restraint.

**Output:** `InterpretationMap{active,overrides,conflicts}`. **Exact:** Overrides match stated passage scope; later is not automatically global. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-047 — Ensemble listening dependencies

**Typed inputs:** `RehearsalBundle = {files:DirectorySnapshot, records:RehearsalNote{id,section,performer,passage,instruction,observation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Recover entrance chains based on hearing other performers.

**NL decomposition:** Extract entrances and cues. → Link cue producers. → Trace performer chains. → Expose cycles and missing cues.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Check whether an action supplies the captured entrance cue.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented score annotations, rehearsal attempts and explicit artistic intentions. Positive witness: Flute follows cello release; cello waits for soloist breath. Contrast: A practice count does not replace the live cue.

**Output:** `CueGraph{dependencies,cues,gaps}`. **Exact:** Edges cite explicit listening instructions, not assumed beat counts. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-048 — Rehearsal diagnosis contrasts

**Typed inputs:** `RehearsalBundle = {files:DirectorySnapshot, records:RehearsalNote{id,section,performer,passage,instruction,observation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate execution repair from a change in artistic intention.

**NL decomposition:** Extract goals and outcomes. → Align repeated attempts. → Contrast stable goals with realization. → Explain revision kind.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Classify a note against captured prior intent and observed result.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented score annotations, rehearsal attempts and explicit artistic intentions. Positive witness: Timing and balance are met but the conductor changes mood. Contrast: A missed agreed entrance receives a corrective instruction.

**Output:** `RehearsalDiagnosis{goalChanges,executionGaps,contrasts}`. **Exact:** Execution gap requires a stated unmet goal. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-049 — Motif character continuity

**Typed inputs:** `RehearsalBundle = {files:DirectorySnapshot, records:RehearsalNote{id,section,performer,passage,instruction,observation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Interpret recurring musical motifs through supplied dramatic changes.

**NL decomposition:** Align supplied motif identities. → Recover dramatic context. → Trace role transitions. → Derive supported performance implications.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare one recurrence with captured earlier roles and changed context.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented score annotations, rehearsal attempts and explicit artistic intentions. Positive witness: Hesitation motif returns confidently after a stated dramatic turn. Contrast: Similar contour without identity evidence is not the same motif.

**Output:** `MotifRoles{occurrences,roles,implications}`. **Exact:** Motif links need identity evidence; implications follow supplied intentions. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-050 — Recording take assembly

**Typed inputs:** `RehearsalBundle = {files:DirectorySnapshot, records:RehearsalNote{id,section,performer,passage,instruction,observation}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Choose repair takes preserving the declared interpretation with edit disclosure.

**NL decomposition:** Extract take intentions and defects. → Compare boundary and phrase meaning. → Partition compatible repairs. → Publish proposal and disclosure.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge a replacement against captured neighbor intentions and boundary notes.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented score annotations, rehearsal attempts and explicit artistic intentions. Positive witness: Matching phrase shape and tempo permit a clean entrance repair. Contrast: A clean take with different character cannot silently replace the chosen reading.

**Output:** `TakeAssembly{segments,conflicts,disclosure}`. **Exact:** Replacements have compatibility evidence and explicit disclosure. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## software-issues

### sol-051 — Duplicate issue causal graph

**Typed inputs:** `IssueBundle = {files:DirectorySnapshot, records:IssueNote{id,product,version,statement,repro,provenance}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Cluster bugs by triggering behavior rather than symptom vocabulary.

**NL decomposition:** Extract trigger sequences. → Normalize supplied action synonyms. → Compare causal paths. → Cluster justified duplicates.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare repros against captured product action meanings.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Authored fictional bugs, release notes, repro paths and developer discussions. Positive witness: Different labels describe failure after draft rename and session restore. Contrast: Blank screens from network refusal and rename have different triggers.

**Output:** `IssueClusters{clusters,repros,surfaceMatches}`. **Exact:** Shared symptoms alone cannot merge reports. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-052 — Regression introduction narrative

**Typed inputs:** `IssueBundle = {files:DirectorySnapshot, records:IssueNote{id,product,version,statement,repro,provenance}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Localize candidate changes through version-specific reproduction evidence.

**NL decomposition:** Align version results. → Map changes to paths. → Remove incompatible changes. → Choose separating comparisons.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess one change against captured failing path.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Authored fictional bugs, release notes, repro paths and developer discussions. Positive witness: Success before autosave redesign and failure after it implicate the changed action path. Contrast: Icon recoloring does not touch the failing path.

**Output:** `RegressionDossier{candidates,ruledOut,nextComparisons}`. **Exact:** Localization remains candidate until discriminating repro evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-053 — Workaround semantic risk

**Typed inputs:** `IssueBundle = {files:DirectorySnapshot, records:IssueNote{id,product,version,statement,repro,provenance}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Check whether symptom suppression preserves the user's goal.

**NL decomposition:** Extract user invariants. → Decompose workaround behavior. → Compare every requirement. → Report tradeoffs and unknowns.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge a step against captured user goals.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Authored fictional bugs, release notes, repro paths and developer discussions. Positive witness: Disabling restoration stops error but loses required unsaved draft recovery. Contrast: Manual saved-draft reopen preserves all stated requirements.

**Output:** `WorkaroundAudit{preserved,sacrificed,unknown}`. **Exact:** No error is not goal completion; preservation needs evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-054 — Issue thread decision extraction

**Typed inputs:** `IssueBundle = {files:DirectorySnapshot, records:IssueNote{id,product,version,statement,repro,provenance}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Recover authorized current decisions through proposals and withdrawals.

**NL decomposition:** Extract options and authority. → Link approvals and withdrawals. → Fold scoped decisions. → Retain unresolved objections.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Apply one thread event to captured active options and authority.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Authored fictional bugs, release notes, repro paths and developer discussions. Positive witness: Maintainer approves compatibility plan after withdrawing breaking change. Contrast: Confident unauthorized comments remain proposals.

**Output:** `DecisionRecord{current,superseded,objections}`. **Exact:** Current decisions need authorized approval; objections are not silently closed. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-055 — Missing-step report tutor

**Typed inputs:** `IssueBundle = {files:DirectorySnapshot, records:IssueNote{id,product,version,statement,repro,provenance}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Ask the smallest repro questions that separate supplied action branches.

**NL decomposition:** Parse actions and results. → Compare alternative paths. → Locate missing branch conditions. → Draft discriminating questions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Ask for one gap using captured paths and explain answer consequences.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Authored fictional bugs, release notes, repro paths and developer discussions. Positive witness: Import mode is omitted, and merge versus replace behave differently. Contrast: Favorite theme does not distinguish import failure.

**Output:** `ReportQuestions{known,branchGaps,questions}`. **Exact:** Every question resolves a branch or missing observation. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## accounting-narratives

### sol-056 — Payment purpose reconciliation

**Typed inputs:** `LedgerBundle = {files:DirectorySnapshot, records:LedgerNote{id,account,period,amount,explanation,evidence}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Match payments to obligations by documented purpose before arithmetic.

**NL decomposition:** Extract obligations. → Interpret payment purposes. → Join compatible purpose and period. → Apply exact amounts.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare one payment with captured service obligations.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional bookkeeping under a fully supplied policy, not financial advice. Positive witness: Remittance settles March storage, matching the service-period invoice. Contrast: Equal payment for April repair is a different obligation.

**Output:** `PaymentLedger{matches,partial,unmatched}`. **Exact:** Amounts reconcile after semantic matching; equality is not identity. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-057 — Restricted fund purpose audit

**Typed inputs:** `LedgerBundle = {files:DirectorySnapshot, records:LedgerNote{id,account,period,amount,explanation,evidence}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Assess fictional expenses under donor-described purpose and supplied policy.

**NL decomposition:** Read restrictions and exceptions. → Extract actual usage. → Compare usage to purpose. → Aggregate grounded categories.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge an expense against captured purpose and documented use.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional bookkeeping under a fully supplied policy, not financial advice. Positive witness: Shared workshop grant covers documented shared bench tools. Contrast: Tools for private renovation do not fit the same purpose.

**Output:** `PurposeAudit{eligible,incompatible,unresolved}`. **Exact:** Eligibility needs purpose and use evidence; absent use stays unknown. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-058 — Accrual story chronology

**Typed inputs:** `LedgerBundle = {files:DirectorySnapshot, records:LedgerNote{id,account,period,amount,explanation,evidence}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate service, billing, and payment periods under supplied rules.

**NL decomposition:** Extract three event kinds. → Interpret deposits and settlements. → Assign documentary periods. → Explain exact bridges.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Resolve narrative using captured service evidence and policy.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional bookkeeping under a fully supplied policy, not financial advice. Positive witness: May invoice says April service completed; policy attributes costs to service period. Contrast: Deposit for June work does not establish May service.

**Output:** `PeriodBridge{service,billing,payment}`. **Exact:** Service needs performance evidence; totals contain no duplicates. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-059 — Expense split rationale ledger

**Typed inputs:** `LedgerBundle = {files:DirectorySnapshot, records:LedgerNote{id,account,period,amount,explanation,evidence}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Reconstruct allocations from actual beneficiaries and explicit split basis.

**NL decomposition:** Read allocation basis. → Extract beneficiaries and quantities. → Assign evidence-grounded units. → Compute shares and remainder.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge one activity against captured beneficiary criteria.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional bookkeeping under a fully supplied policy, not financial advice. Positive witness: Two service days benefit shared project and one private project under day-based policy. Contrast: Department label does not establish beneficiary.

**Output:** `AllocationLedger{units,shares,unresolved}`. **Exact:** Allocated plus unresolved equals original; every unit has evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-060 — Refund causal chain

**Typed inputs:** `LedgerBundle = {files:DirectorySnapshot, records:LedgerNote{id,account,period,amount,explanation,evidence}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Avoid counting cancellation, credit, and refund as separate corrections.

**NL decomposition:** Extract causes and references. → Link correction stages. → Fold balances by cause. → Return exact open and settled amounts.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Determine whether an event advances a captured chain or starts another.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional bookkeeping under a fully supplied policy, not financial advice. Positive witness: Canceled invoice creates credit then cash settles that same credit. Contrast: Later courtesy credit has a distinct cause.

**Output:** `CorrectionLedger{chains,settled,openCredits}`. **Exact:** Cause-specific corrections cannot be double-counted. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## employment-negotiation

### sol-061 — Offer conditionality ledger

**Typed inputs:** `NegotiationBundle = {files:DirectorySnapshot, records:OfferNote{id,party,date,term,statement,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Track binding, contingent, suggested and withdrawn terms under supplied authority.

**NL decomposition:** Extract content modality and authority. → Link conditions. → Fold approvals and amendments. → Return current terms.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret new statement against captured conditional commitments and authority.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional offers, preferences, contingent concessions and authorized agreements. Positive witness: Authorized remote-work commitment becomes active after written location approval. Contrast: Unauthorized might be possible remains a suggestion.

**Output:** `OfferLedger{committed,conditions,proposals,superseded}`. **Exact:** Commitments need authorization; suggestions cannot satisfy conditions. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-062 — Preference tradeoff interpreter

**Typed inputs:** `NegotiationBundle = {files:DirectorySnapshot, records:OfferNote{id,party,date,term,statement,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare offers using stated partial and conditional preferences.

**NL decomposition:** Extract preferences. → Build partial order. → Activate applicable criteria. → Explain dominance and unresolved tradeoffs.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge offer differences against captured preference conditions.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional offers, preferences, contingent concessions and authorized agreements. Positive witness: Predictable hours rank above bonus; commute matters only after late shifts. Contrast: Assuming salary always dominates contradicts supplied priorities.

**Output:** `TradeoffReport{order,conditions,comparisons,unknowns}`. **Exact:** No ranking is invented for incomparable criteria. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-063 — Negotiation misunderstanding repair

**Typed inputs:** `NegotiationBundle = {files:DirectorySnapshot, records:OfferNote{id,party,date,term,statement,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Expose shared terms hiding different concrete arrangements.

**NL decomposition:** Extract usage examples. → Compare rights and obligations. → Locate meaning mismatches. → Draft transparent definitions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Explain concrete arrangements behind captured uses of one term.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional offers, preferences, contingent concessions and authorized agreements. Positive witness: Flexible means choosing start time to candidate and accepting last-minute shifts to employer. Contrast: Different labels for choosing start time are substantive agreement.

**Output:** `TermClarification{shared,divergent,definitions}`. **Exact:** Definitions preserve positions; shared vocabulary is not agreement. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-064 — Concession dependency planner

**Typed inputs:** `NegotiationBundle = {files:DirectorySnapshot, records:OfferNote{id,party,date,term,statement,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Find coherent packages among conditional concessions.

**NL decomposition:** Extract concessions and conditions. → Build authorized dependency graph. → Enumerate finite combinations. → Explain remaining packages.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Check a package against captured concession prerequisites.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional offers, preferences, contingent concessions and authorized agreements. Positive witness: Extra leave depends explicitly on earlier start acceptance. Contrast: Nearby equipment allowance is not conditional on start date.

**Output:** `PackagePlan{edges,coherent,incompatible}`. **Exact:** All required conditions are present; adjacency creates no dependency. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-065 — Meeting-to-contract scope audit

**Typed inputs:** `NegotiationBundle = {files:DirectorySnapshot, records:OfferNote{id,party,date,term,statement,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare draft rights and obligations with authorized meeting agreement.

**NL decomposition:** Extract authorized agreements. → Align corresponding clauses. → Compare scope and discretion. → Draft documentary clarification questions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Evaluate one clause against captured agreed examples.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional offers, preferences, contingent concessions and authorized agreements. Positive witness: One weekend shift per quarter becomes unrestricted weekend assignment. Contrast: A paraphrase preserving bound and notice is faithful.

**Output:** `AgreementAudit{faithful,changes,omissions}`. **Exact:** Flags cite both spans; synonyms alone are not changes. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## complaint-reconciliation

### sol-066 — Complaint event braid

**Typed inputs:** `ComplaintBundle = {files:DirectorySnapshot, records:Contact{id,customer,date,channel,text,event}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Unify contacts for one incident without merging replacements or recurrence.

**NL decomposition:** Extract incident anchors. → Align channel references. → Separate replacements. → Build chronological incident braids.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare contacts using captured parcel history rather than vocabulary.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional contacts, status notes, policies and agreed remedy evidence. Positive witness: Call and email concern same missing parcel; later damage concerns its replacement. Contrast: Repeated failed delivery words do not establish event identity.

**Output:** `IncidentBraid{groups,contacts,separateIncidents}`. **Exact:** Groups require compatible anchors; stated replacements stay distinct. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-067 — Remedy completeness checker

**Typed inputs:** `ComplaintBundle = {files:DirectorySnapshot, records:Contact{id,customer,date,channel,text,event}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Check every promised remedy component without inventing entitlements.

**NL decomposition:** Extract actual commitments. → Decompose remedy parts. → Join completion evidence. → Return unresolved frontier.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Check an event against captured remedy conditions.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional contacts, status notes, policies and agreed remedy evidence. Positive witness: Replacement and postage reimbursement promised; only dispatch is documented. Contrast: Mentioned inconvenience has no supplied compensation promise.

**Output:** `RemedyLedger{promised,completed,remaining}`. **Exact:** Completion needs specific witnesses; unpromised compensation is not added. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-068 — Apology-action consistency

**Typed inputs:** `ComplaintBundle = {files:DirectorySnapshot, records:Contact{id,customer,date,channel,text,event}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Audit later requests against earlier scoped assurances.

**NL decomposition:** Extract assurance scope. → Link actions to purpose. → Compare imposed duties. → Explain contradictions and new gaps.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge a request against captured promise and new evidence need.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional contacts, status notes, policies and agreed remedy evidence. Positive witness: Agent promises no repeat form then demands the same form. Contrast: A photograph for a new documented gap may be outside that assurance.

**Output:** `AssuranceAudit{assurances,actions,contradictions}`. **Exact:** Contradictions concern same condition; new purpose requires evidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-069 — Escalation reason handoff

**Typed inputs:** `ComplaintBundle = {files:DirectorySnapshot, records:Contact{id,customer,date,channel,text,event}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Condense a case around its unresolved decision and decisive evidence.

**NL decomposition:** Recover unresolved proposition. → Trace relevant attempts. → Select decision-changing evidence. → Write source-linked handoff.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Retain a detail only if it changes the captured requested decision.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional contacts, status notes, policies and agreed remedy evidence. Positive witness: Departments disagree which parcel the refund covers. Contrast: An unrelated old complaint does not explain current ambiguity.

**Output:** `EscalationPacket{question,facts,attempts,decision}`. **Exact:** Decisive source evidence survives condensation; irrelevant history is excluded. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-070 — Policy exception precedent

**Typed inputs:** `ComplaintBundle = {files:DirectorySnapshot, records:Contact{id,customer,date,channel,text,event}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare exceptions by policy rationale rather than superficial outcomes.

**NL decomposition:** Read policy and precedents. → Extract present cause. → Compare rationale dimensions. → Report analogy and authority gaps.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Name relevant matches and disanalogies against captured rationale.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional contacts, status notes, policies and agreed remedy evidence. Positive witness: Agent-caused missed-deadline precedent matches current documented cause. Contrast: Illness exception with equal delay has a different rationale.

**Output:** `ExceptionComparison{precedents,matches,disanalogies,authority}`. **Exact:** Shared outcome alone does not establish applicable precedent. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## voyage-logs

### sol-071 — Route account reconciliation

**Typed inputs:** `VoyageBundle = {files:DirectorySnapshot, records:VoyageNote{id,vessel,date,location,text,author}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Align voyage descriptions through supplied landmarks without inventing coordinates.

**NL decomposition:** Extract movement relations. → Align chart names. → Build leg-specific graph. → Compare accounts and unknowns.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret a statement within captured leg and landmark context.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented voyage diaries, cargo notes, relative charts and orders, not navigation. Positive witness: Port-side island after rounding matches mate account under supplied legend. Contrast: Similar island name on another leg does not resolve this leg.

**Output:** `RouteAccount{links,constraints,compatible,conflicts}`. **Exact:** Links cite supplied landmarks; no coordinates are fabricated. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-072 — Cargo condition responsibility

**Typed inputs:** `VoyageBundle = {files:DirectorySnapshot, records:VoyageNote{id,vessel,date,location,text,author}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Locate damage intervals through handoffs without unsupported blame.

**NL decomposition:** Extract handoffs and conditions. → Separate packaging and cargo. → Bound first change. → State responsibility questions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare observation with captured condition and custody epoch.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented voyage diaries, cargo notes, relative charts and orders, not navigation. Positive witness: Cargo dry at loading, wet at unloading, intermediate hold leak documented. Contrast: Old-looking packaging does not prove wet contents.

**Output:** `CargoTimeline{epochs,changes,questions}`. **Exact:** Damage onset is bounded evidence; blame needs causal witnesses. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-073 — Command intention versus execution

**Typed inputs:** `VoyageBundle = {files:DirectorySnapshot, records:VoyageNote{id,vessel,date,location,text,author}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate issued orders, acknowledgment and completed maneuver.

**NL decomposition:** Extract orders and conditions. → Link acknowledgments. → Locate execution evidence. → Explain delays.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Classify a log event against captured order content.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented voyage diaries, cargo notes, relative charts and orders, not navigation. Positive witness: Course change acknowledged but helm log states delayed execution. Contrast: Acknowledgment alone does not establish changed course.

**Output:** `CommandLedger{orders,acknowledgments,execution,delays}`. **Exact:** Completion needs execution evidence, not intent. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-074 — Storm narrative perspective

**Typed inputs:** `VoyageBundle = {files:DirectorySnapshot, records:VoyageNote{id,vessel,date,location,text,author}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Explain severity differences by supplied observer exposure.

**NL decomposition:** Extract positions and times. → Align intervals. → Apply exposure context. → Separate explainable and unresolved differences.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare conditions using captured exposure rather than presumed reliability.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented voyage diaries, cargo notes, relative charts and orders, not navigation. Positive witness: Deck spray and sheltered cabin quiet occur in same interval. Contrast: Same exposed station gives incompatible accounts without timing difference.

**Output:** `HazardReadings{contexts,compatible,conflicts}`. **Exact:** Perspective explanation requires explicit access difference. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-075 — Port-call purpose classification

**Typed inputs:** `VoyageBundle = {files:DirectorySnapshot, records:VoyageNote{id,vessel,date,location,text,author}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Distinguish planned mission stops from actual contingent diversions.

**NL decomposition:** Read plans. → Extract arrival evidence. → Match contemporaneous purposes. → Contrast planned and realized mission.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Classify a call against captured arrival and changed reason.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented voyage diaries, cargo notes, relative charts and orders, not navigation. Positive witness: Trading itinerary is followed by an unplanned repairs-only harbor visit. Contrast: Old itinerary mention does not prove an actual visit.

**Output:** `PortLedger{calls,purposes,unvisited,diversions}`. **Exact:** Actual calls need visit evidence; old purpose cannot override change. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## philosophical-arguments

### sol-076 — Hidden premise exposer

**Typed inputs:** `ArgumentBundle = {files:DirectorySnapshot, records:ArgumentNote{id,author,text,premise,conclusion,scope}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Find minimal assumptions needed by an inference without endorsing them.

**NL decomposition:** Extract inference transitions. → Check background commitments. → Construct allowed counterexamples. → State needed bridging assumptions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Propose countermodel and needed premise under captured commitments.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original arguments with explicit commitments and finite counterexample sets. Positive witness: Benefit for everyone is used to infer consent, requiring a benefit-to-consent bridge. Contrast: Restating a premise requires no new consent bridge.

**Output:** `PremiseAudit{premises,bridges,countermodels}`. **Exact:** Added premises explain inference gaps, not asserted truths. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-077 — Equivocation sense graph

**Typed inputs:** `ArgumentBundle = {files:DirectorySnapshot, records:ArgumentNote{id,author,text,premise,conclusion,scope}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Track conclusion-relevant shifts in term meaning.

**NL decomposition:** Collect definitions and examples. → Assign supported senses. → Trace repeated-term inferences. → Expose unsupported substitutions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare uses against captured definitions and proposition meaning.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original arguments with explicit commitments and finite counterexample sets. Positive witness: Free shifts from no price to no coercion while inference treats them alike. Contrast: Different examples of no coercion can preserve the same sense.

**Output:** `SenseGraph{senses,shifts,invalidLinks}`. **Exact:** Flags cite sense evidence; examples alone do not establish shift. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-078 — Objection reply dialectic

**Typed inputs:** `ArgumentBundle = {files:DirectorySnapshot, records:ArgumentNote{id,author,text,premise,conclusion,scope}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Check whether replies answer the challenged proposition.

**NL decomposition:** Decompose challenge targets. → Extract reply propositions. → Match scopes. → Return residual objection ledger.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess reply against captured exact challenge.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original arguments with explicit commitments and finite counterexample sets. Positive witness: Distribution objection receives only total-benefit evidence. Contrast: Evidence each group receives promised share addresses distribution.

**Output:** `DialecticMap{objections,targets,answered,residual}`. **Exact:** Answered status requires proposition coverage, not confidence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-079 — Analogy structural mapper

**Typed inputs:** `ArgumentBundle = {files:DirectorySnapshot, records:ArgumentNote{id,author,text,premise,conclusion,scope}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Assess relations required by analogical conclusions.

**NL decomposition:** Extract inferential structure. → Map source and target roles. → Check required relations. → Bound transfer conclusions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge a mapping against captured conclusion requirements.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original arguments with explicit commitments and finite counterexample sets. Positive witness: Reversible source choice is compared with irreversible target although reversal drives conclusion. Contrast: Color difference is irrelevant to reversibility inference.

**Output:** `AnalogyReport{relations,disanalogies,bounds}`. **Exact:** Relevant disanalogies disrupt used relations; incidental traits stay separate. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-080 — Definition revision fixed point

**Typed inputs:** `ArgumentBundle = {files:DirectorySnapshot, records:ArgumentNote{id,author,text,premise,conclusion,scope}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Refine a stipulation against finite supplied challenges while preserving accepted cases.

**NL decomposition:** Read stipulation and positives. → Classify finite challenges. → Revise smallest supported clause. → Drain challenge frontier into honest unresolved cases.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Propose clause changes preserving captured positives and resolving a named counterexample.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original arguments with explicit commitments and finite counterexample sets. Positive witness: Future intention definition fails a case distinguishing private intent from communicated commitment. Contrast: Excluding conditional promises loses an agreed positive case.

**Output:** `DefinitionRevision{definition,discharged,remaining}`. **Exact:** Each continuing revision discharges a challenge; positives stay covered. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## speculative-worldbuilding

### sol-081 — Canon exception hierarchy

**Typed inputs:** `WorldBible = {files:DirectorySnapshot, records:WorldNote{id,culture,era,claim,rule,narrator}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Resolve authoritative rules and scoped exceptions separately from character rumor.

**NL decomposition:** Extract authoritative scopes. → Separate beliefs. → Build exception hierarchy. → Check scenes under applicable rule.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Apply captured authority and exception scopes to a scene.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented canon rules, belief cards, exceptions and unreliable chronicles. Positive witness: Member-only doors except emergency exits permit a documented emergency opening. Contrast: A character rumor of stranger access is not canon revision.

**Output:** `CanonRegister{rules,exceptions,beliefs,conflicts}`. **Exact:** Overrides need canon evidence; rumor cannot amend rules. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-082 — Magic cost conservation

**Typed inputs:** `WorldBible = {files:DirectorySnapshot, records:WorldNote{id,culture,era,claim,rule,narrator}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Track fictional costs and permitted equivalences as obligations.

**NL decomposition:** Read exchange rules. → Extract incurred obligations. → Link allowed payments. → Fold outstanding balances.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge payment against captured cost and allowed equivalences.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented canon rules, belief cards, exceptions and unreliable chronicles. Positive witness: Spell costs memory or equivalent pledged service; later service payment is explicit. Contrast: Dramatic gesture is not payment unless supplied canon permits it.

**Output:** `MagicLedger{costs,payments,outstanding}`. **Exact:** Every discharge has permitted witness; symbolism does not establish equivalence. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-083 — Culture-specific misunderstanding

**Typed inputs:** `WorldBible = {files:DirectorySnapshot, records:WorldNote{id,culture,era,claim,rule,narrator}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Generate grounded misunderstanding from explicitly divergent invented conventions.

**NL decomposition:** Read conventions. → Find divergent action meanings. → Construct actor and observer knowledge. → Design revealing repair cue.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret gesture under captured actor and observer conventions.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented canon rules, belief cards, exceptions and unreliable chronicles. Positive witness: Returned gift means refusal to one culture and reciprocal request to another. Contrast: Identical refusal meanings supply no misunderstanding.

**Output:** `MisunderstandingScene{action,intent,reading,repair}`. **Exact:** Scene uses supplied conventions and reveals actual mismatch. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-084 — Timeline retcon consequence

**Typed inputs:** `WorldBible = {files:DirectorySnapshot, records:WorldNote{id,culture,era,claim,rule,narrator}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Trace temporal retcon through institutions and characters.

**NL decomposition:** Parse temporal fact delta. → Trace institutional dependencies. → Propagate character consequences. → Offer minimal coherent repair alternatives.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess a fact against captured event and causal chain.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented canon rules, belief cards, exceptions and unreliable chronicles. Positive witness: Moving treaty later invalidates an earlier treaty-founded school and its graduate. Contrast: Mountain existence has no supplied treaty dependency.

**Output:** `RetconPlan{affected,paths,preserved,repairs}`. **Exact:** Affected facts have dependency paths, not thematic association. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-085 — Character belief divergence

**Typed inputs:** `WorldBible = {files:DirectorySnapshot, records:WorldNote{id,culture,era,claim,rule,narrator}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Track deception and correction as local beliefs separately from canon.

**NL decomposition:** Separate world and received claims. → Build character access events. → Fold local beliefs and doubt. → Compare action with local knowledge.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Apply information to captured access and prior belief.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented canon rules, belief cards, exceptions and unreliable chronicles. Positive witness: Forged decree convinces one character gate is closed while another sees it open. Contrast: Audience canon note does not inform characters.

**Output:** `BeliefLedger{facts,beliefs,updates}`. **Exact:** Updates need information paths; narrator knowledge is not character knowledge. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## literary-criticism

### sol-086 — Narrator reliability evidence

**Typed inputs:** `ReadingBundle = {files:DirectorySnapshot, records:LiteraryPassage{id,work,speaker,text,scene,perspective}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Assess reliability locally through source-supported claim conflicts.

**NL decomposition:** Extract claims and depicted facts. → Interpret figurative scope. → Link support and conflict. → Describe local reliability changes.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge claim against captured narrative context and literal scope.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional prose and competing readings with supplied narrative context. Positive witness: Narrator denies anyone waited then finds unopened gifts and admits avoidance. Contrast: Metaphorical empty city is not automatically literal absence.

**Output:** `ReliabilityMap{links,contradictions,figurativeUses}`. **Exact:** Reliability cites specific conflicts; figurative claims retain their scope. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-087 — Metaphor entailment comparison

**Typed inputs:** `ReadingBundle = {files:DirectorySnapshot, records:LiteraryPassage{id,work,speaker,text,scene,perspective}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare projected relations of competing metaphors.

**NL decomposition:** Identify source and target. → Extract projected relations. → Compare interpretive consequences. → Return grounded differences.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Project a relation using captured surrounding actions.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional prose and competing readings with supplied narrative context. Positive witness: Council as machine projects coordination; council as market projects bargaining in supplied scenes. Contrast: Shared loud adjective does not reveal relational structure.

**Output:** `MetaphorMatrix{features,suppressed,textEvidence}`. **Exact:** Relations need textual witnesses; generic associations are not imported. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-088 — Scene omission inference

**Typed inputs:** `ReadingBundle = {files:DirectorySnapshot, records:LiteraryPassage{id,work,speaker,text,scene,perspective}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate necessary gap events from unsupported invented details.

**NL decomposition:** Record before-after states. → Find required transitions. → Constrain using later references. → Separate necessities from optional details.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Build minimal bridge using captured states and name unknown details.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional prose and competing readings with supplied narrative context. Positive witness: Key changes hands and later exchange reference supports transfer. Contrast: No passage supports why the transfer occurred.

**Output:** `EllipsisReport{necessary,possible,unsupported}`. **Exact:** Bridge fits observed change; motive is not invented as fact. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-089 — Intertextual transformation

**Typed inputs:** `ReadingBundle = {files:DirectorySnapshot, records:LiteraryPassage{id,work,speaker,text,scene,perspective}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Explain how supplied later text transforms earlier roles and conclusion.

**NL decomposition:** Compare distinctive structures. → Align narrative roles. → Find changed agency. → Explain transformation and weak links.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess echo using captured earlier relations and later assignment.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional prose and competing readings with supplied narrative context. Positive witness: Repeated oath structure now empowers the earlier excluded character. Contrast: Common nondistinctive phrase is weak allusion evidence.

**Output:** `IntertextReport{links,structures,roleChanges}`. **Exact:** Strong allusion needs distinctive structure; transformation cites changed relation. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-090 — Competing reading adjudication

**Typed inputs:** `ReadingBundle = {files:DirectorySnapshot, records:LiteraryPassage{id,work,speaker,text,scene,perspective}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare interpretations by decisive evidence coverage while preserving honest ties.

**NL decomposition:** Extract reading claims. → Map support and conflict. → Compare decisive coverage. → Write bounded judgment or tie.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess explanation against captured decisive passages.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Original fictional prose and competing readings with supplied narrative context. Positive witness: One reading explains withheld letter and final disclosure; another ignores disclosure. Contrast: Eloquent wording cannot offset contradiction of an explicit scene.

**Output:** `ReadingComparison{coverage,conflicts,assumptions,ties}`. **Exact:** Judgments show coverage; underdetermined questions keep ties. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## weather-diaries

### sol-091 — Local weather vocabulary

**Typed inputs:** `WeatherDiaryBundle = {files:DirectorySnapshot, records:WeatherNote{id,observer,date,place,text,exposure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Interpret diary terms through supplied observer usage rather than modern defaults.

**NL decomposition:** Collect contextual terms. → Build local crosswalk. → Interpret under observer convention. → Retain insufficiently anchored meanings.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Resolve descriptor using captured local examples.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented historical-style local glosses, weather diaries and exposure changes. Positive witness: Soft air means humid warmth without required rain for this observer. Contrast: Another observer uses soft for light rain.

**Output:** `DescriptorIndex{meanings,entries,unknowns}`. **Exact:** Interpretations cite local evidence; modern defaults are not imported. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-092 — Observation exposure confound

**Typed inputs:** `WeatherDiaryBundle = {files:DirectorySnapshot, records:WeatherNote{id,observer,date,place,text,exposure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate weather changes from observer exposure changes.

**NL decomposition:** Extract location and placement events. → Segment stable exposure. → Compare within and across epochs. → State corroboration needs.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Judge contrast using captured exposure history and reference notes.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented historical-style local glosses, weather diaries and exposure changes. Positive witness: Calmer reports begin after moving into sheltered courtyard. Contrast: Unchanged exposed station plus storm-ending note supports local change.

**Output:** `ExposureReport{epochs,confounds,changes}`. **Exact:** Move-confounded change remains unresolved; unchanged weather is not assumed. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-093 — Storm episode boundaries

**Typed inputs:** `WeatherDiaryBundle = {files:DirectorySnapshot, records:WeatherNote{id,observer,date,place,text,exposure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Group episodes by continuation, lull, onset and ending relations.

**NL decomposition:** Extract event-boundary cues. → Align adjacent references. → Build qualified episode chains. → Summarize phases.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Classify quiet interval against captured event references.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented historical-style local glosses, weather diaries and exposure changes. Positive witness: Afternoon calm is called a lull before same storm returns. Contrast: New front after settled days marks another episode.

**Output:** `StormEpisodes{groups,lulls,unknownBoundaries}`. **Exact:** Joins need continuation evidence; missing days do not imply continuity. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-094 — Impact cause disentangler

**Typed inputs:** `WeatherDiaryBundle = {files:DirectorySnapshot, records:WeatherNote{id,observer,date,place,text,exposure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Distinguish weather observations from human-confounded impacts.

**NL decomposition:** Separate weather and impact. → Extract human actions. → Build contrasted explanations. → Report bounded attribution.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Assess cause using captured comparison sites and actions.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented historical-style local glosses, weather diaries and exposure changes. Positive witness: Flooded field had deliberately closed drainage; nearby open-gate field stayed usable. Contrast: Damage is not a direct measurement of weather severity.

**Output:** `ImpactDossier{weather,humanFactors,contrasts,unknowns}`. **Exact:** Impacts are not severity measurements; causes stay evidence-bounded. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-095 — Seasonal memory calibration

**Typed inputs:** `WeatherDiaryBundle = {files:DirectorySnapshot, records:WeatherNote{id,observer,date,place,text,exposure}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Compare remembered duration with diary coverage and event salience.

**NL decomposition:** Extract retrospective scope. → Map contemporary coverage. → Compare literal duration and emphasis. → Explain gaps and supported overstatement.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret memory against captured coverage and salient events.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Invented historical-style local glosses, weather diaries and exposure changes. Positive witness: Endlessly wet memory contrasts short flood and many dry entries. Contrast: Damaging flood may explain salience without continuous rain.

**Output:** `SeasonMemory{claims,coverage,salience,gaps}`. **Exact:** Missing intervals stay unknown; salient harm is not continuous weather. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## project-planning

### sol-096 — Decision dependency closure

**Typed inputs:** `ProjectBundle = {files:DirectorySnapshot, records:ProjectNote{id,team,date,statement,dependency,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Update deliverables whose explicit assumptions depend on a changed decision.

**NL decomposition:** Extract authorized delta. → Map consumed assumptions. → Expand tests and handoffs. → Build minimal update frontier.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Trace one deliverable to captured changed assumption.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional cross-team plans, decision records, handoffs and completion evidence. Positive witness: Export format change affects importer and consuming acceptance test. Contrast: Same-milestone color update does not consume that format.

**Output:** `DecisionImpact{affected,paths,safe}`. **Exact:** Affected work needs dependency path; shared milestone is insufficient. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-097 — Handoff contract interpreter

**Typed inputs:** `ProjectBundle = {files:DirectorySnapshot, records:ProjectNote{id,team,date,statement,dependency,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Expose differing concrete definitions of handoff readiness.

**NL decomposition:** Extract readiness definitions. → Align artifacts and evidence. → Locate absent obligations. → Draft explicit shared contract.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Compare statements against captured consumer requirements.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional cross-team plans, decision records, handoffs and completion evidence. Positive witness: Sender means file exists; receiver requires valid schema and signed manifest. Contrast: Different validator-report labels can express identical requirement.

**Output:** `HandoffAudit{shared,mismatches,contract}`. **Exact:** Contract preserves mandatory requirements and unresolved incompatible choices. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-098 — Blocked work frontier

**Typed inputs:** `ProjectBundle = {files:DirectorySnapshot, records:ProjectNote{id,team,date,statement,dependency,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Find useful work respecting hard prerequisites while separating preferences.

**NL decomposition:** Extract tasks and prerequisites. → Classify dependency strength. → Propagate satisfied states. → Return useful unblocked frontier.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Check task against captured prerequisite evidence.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional cross-team plans, decision records, handoffs and completion evidence. Positive witness: Integration waits for approved schema but fixtures need only agreed meanings. Contrast: Preferred joint review is not mandatory approval without explicit rule.

**Output:** `ProgressFrontier{ready,blockers,optional}`. **Exact:** Ready tasks meet hard requirements; preference cannot create false blockers. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-099 — Meeting action ownership

**Typed inputs:** `ProjectBundle = {files:DirectorySnapshot, records:ProjectNote{id,team,date,statement,dependency,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Separate accepted owners, helpers, reviewers and suggestions.

**NL decomposition:** Decompose action speech acts. → Find acceptance and authority. → Separate concrete roles. → Register unresolved ownership.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Interpret assignment using captured acceptance and authority.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional cross-team plans, decision records, handoffs and completion evidence. Positive witness: One offers advice; another accepts implementation and lead confirms. Contrast: Name in a suggested plan is not accepted ownership.

**Output:** `ActionRegister{owners,roles,unassigned}`. **Exact:** Ownership requires acceptance or authorized assignment. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

### sol-100 — Requirement contradiction repair

**Typed inputs:** `ProjectBundle = {files:DirectorySnapshot, records:ProjectNote{id,team,date,statement,dependency,authority}[], context:DomainContext}` and `Inquiry = {question:string, scope:string[], requestedOutput:string}`.

**Purpose:** Reconcile behavior requests by supplied audience, mode and rationale.

**NL decomposition:** Extract scope and rationale. → Align same behaviors. → Partition explicit compatible scopes. → Ask minimal contradiction-resolving questions.

**Runtime scoped NL literal:** `(item:EvidenceUnit, context:StageState) -> RelationDecision` captures `inquiry`, `stageState`, and `domainContext`. 2. Test whether captured audience and mode allow coherent scoped behavior.

**Iteration / call stack:** The immutable snapshot yields a finite frontier; each continuing step removes one item. Uncertainty is an output record, not a retry loop. Measured `iterateOn` drains that frontier. Required genuine nested path: `main -> buildSemanticRelations -> resolveScopedMeaning -> interpretLocalRelation`.

**Sources:** Fictional cross-team plans, decision records, handoffs and completion evidence. Positive witness: Operators need terse alerts; administrators need detailed review explanations. Contrast: Opposite same-action same-mode requirements truly conflict.

**Output:** `RequirementPlan{scopes,conflicts,questions}`. **Exact:** Scope splits need evidence; invented modes cannot hide conflict. Verify paths, spans, typed structure, decreasing work, captures, and nested parent IDs. **Teacher:** contextual entailment and uncertainty, adversarial paraphrases and context reversals, distinct algorithm review, then about 48 live samples. **State:** source candidate; oracle and sampling require explicit evidence.

## Priority fixture cohort

`sol-001` through `sol-010` now each carry 24 concrete interacting evidence records plus a query (250 visible files total). Structured exact oracles are manually derived from those passages and remain outside model-visible files. These ten are prepared for independent source replay and semantic teacher review; the other 90 remain design seeds. Actual nested execution and about 48 live samples are still required. The JSON records the exact oracle derivation and admission state per case.
