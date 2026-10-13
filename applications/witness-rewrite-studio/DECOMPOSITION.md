# Unreliable Witness Rewrite Studio

## The natlang program

The studio rewrites a fictional story after a precise change to one canonical event. It must preserve the director's protected meanings: who can know which facts, which physical objects actually move, why later actions happen, which callbacks remain intelligible, and which recollections remain deliberately unresolved. A plausible new line is insufficient if it creates knowledge from nowhere or resolves a protected ambiguity.

The whole editing algorithm and all semantic policy are natural-language functions. The application decides on an immutable story snapshot, returns a typed patch packet, and can apply that packet to a private folder transaction. Effects are data until the exact commit seam applies the approved packet. A diagnostic fixture does not publish a story.

## Source organization and typed ownership

The implementation has 27 `.nl` modules and 713 lines of numbered NL pseudocode. `types.ts` defines data only. Each stage records its own source-backed contribution before calling a deeper specialist. The root consumes the returned analysis, proposals and audits; it does not redo their judgments from raw text.

| Unit | Source | Owned NL transformation |
| --- | --- | --- |
| Entrypoint | `studio.nl` | Interpret the edit request, orchestrate branches, run measured repair, and return one justified patch packet. |
| Intake | `studio/intake.nl` | Bind visible file identities to one immutable input snapshot and separate task rules from story assertions. |
| Source reading | `readRecords.nl` | Inventory acts, selected takes, memories, notes and protected clauses by their declared ownership. |
| Request meaning | `studio/intake/interpretRequest.nl` | Recover the exact fact delta and the director's hard preservation requirements. |
| World analysis | `world.nl` | Assemble entity, frame, temporal and authority views without merging different kinds of truth. |
| Frame construction | `world/parseFrames.nl` | Interpret selected, remembered, quoted and speculative scene frames. |
| Local frame judgment | `world/parseFrames/interpretFrame.nl` | Form and consume a scoped NL literal that judges one event under its inherited frame and speaker. |
| Entity binding | `world/bindEntities.nl` | Join explicit aliases while keeping duplicate physical objects and namesakes separate. |
| Temporal relation | `world/orderEvents.nl` | Infer only supported event precedences and preserve unresolved branches. |
| Canon authority | `world/evaluateAuthority.nl` | Distinguish world facts, endorsed claims, reports and beliefs by source authority. |
| Knowledge analysis | `epistemics.nl` | Construct character-local information histories and audit dialogue. |
| Acquisition | `epistemics/acquire.nl` | Interpret witnessing and communication events as scoped knowledge transfers. |
| Dialogue audit | `epistemics/checkDialogue.nl` | Match asserted dialogue content to its speaker's actual information path. |
| Licensing judgment | `epistemics/checkDialogue/resolveLicensing.nl` | Invoke a contextual NL literal and attach its decision to the dialogue obligation. |
| Physical continuity | `props.nl` | Fold custody and state transitions while keeping originals, copies and remembered appearances distinct. |
| Motif linkage | `motifs.nl` | Interpret meaningful callbacks and their dramatic function rather than repeated vocabulary. |
| Impact construction | `studio/impact.nl` | Build the complete finite graph of obligations affected by the requested edit. |
| Impact step | `studio/impact/closureStep.nl` | Expand one pending obligation into its already inventoried dependencies and consume a scoped dependency judgment. |
| Candidate generation | `proposals.nl` | Produce alternative repairs with explicit preserved and changed propositions. |
| Candidate replay | `proposals/simulate.nl` | Replay each proposed fact delta through the existing world, knowledge and prop views. |
| Preservation audit | `audit.nl` | Combine independent canon, knowledge, motivation and ambiguity checks. |
| Canon check | `audit/checkCanon.nl` | Judge whether a proposal realizes the requested event change. |
| Belief safety | `audit/checkKnowledge.nl` | Judge whether proposed lines have visible information paths. |
| Ambiguity safety | `audit/checkAmbiguity.nl` | Preserve explicitly protected unresolved recollections and provenance questions. |
| Repair reducer | `studio/repairStep.nl` | Settle one connected affected component in a private folder, consuming generation and audit results. |
| Tradeoff choice | `studio/repairStep/compareCandidates.nl` | Prefer the declared smallest coherent repair while retaining genuine equal alternatives. |
| Packet assembly | `studio/packagePatch.nl` | Return proposed edits, obligation coverage, unresolved issues and a source-witness ledger. |

## Real call graph and different execution paths

- `studio → intake → interpretRequest`: hard requirements and event delta.
- `studio → world → parseFrames → interpretFrame → runtime NL frame literal`: deeply nested ownership and epistemic scope.
- `studio → epistemics → checkDialogue → resolveLicensing → runtime NL knowledge literal`: a distinct information-access judgment.
- `studio → impact → closureStep → runtime NL dependency literal`: graph expansion under a captured event delta.
- `studio → repairStep → proposals → simulate → audit → checkAmbiguity`: counterfactual replay and preservation, using explicit `uses` for shared callable modules.
- `studio → repairStep → compareCandidates`: selection from verified semantic alternatives, followed by packet assembly.

There is no uniform pass-through chain. Frame interpretation, actor knowledge, physical continuity and motifs form parallel semantic branches; they join at impact construction. Candidate replay reuses those views with changed facts. Its audit calls specialists through explicit callable paths. Each inline literal's result is attached to an obligation, used to eliminate a candidate, or included in the final witness ledger.

## Iteration and state

First inventory the complete finite obligation graph from the snapshot. Impact iteration visits one unexpanded graph node at a time with a measure equal to the count of unexpanded inventoried nodes. A node already visited is not appended again. Repair uses a folder reducer over connected affected components; each continuing step settles one component as repaired or explicitly unresolved. Its measure is the remaining component count. Neither uncertainty nor an unavailable repair causes an endless retry.

A candidate lives on a private transaction copy. A semantic rejection discards that candidate. The final pure commit applies only the packet's listed edits after exact base-hash and source-span verification. Timers and external sends are outside this application.

## Exact seams and their justification

- **Source snapshot I/O:** load declared files and hashes exactly; it is an outside-world service, not an interpretation policy.
- **Witness verifier:** check that every cited source path exists and quotation bytes match. It does not decide what a quotation entails.
- **Patch verifier:** check nonoverlapping original spans, base hashes and allowed paths. Meaning preservation remains NL.
- **Private folder transaction:** use existing `Folder.propose`/`apply` and `iterateOn`; add no alternate durability mechanism.
- **Output serialization:** write typed reports and apply a bounded patch packet exactly. Candidate choice and recovery remain NL.

No crisp helper decides source authority, character knowledge, motif identity, dependency relevance, ambiguity preservation, repair acceptability or tradeoffs.

## First concrete fixture and oracle contract

A selected harbor drama contains a lost letter, an original key and a rehearsal duplicate, a quoted recollection, an unseen physical transfer, a misleading confession and a callback whose meaning depends on who received the letter. The director removes one actual delivery while preserving the protagonist's uncertainty and the final reconciliation. More than twenty connected facts contribute to its causal, knowledge or physical obligations; unrelated padding is excluded.

The fixture's 26 source records contain more than twenty connected story, authority, access, timing, custody and dramatic premises; seven records are adversarial proposal recipes. Its input IDs and output field meanings are model-visible. The oracle lives outside callable source and input files. It states required changed facts, allowed alternative repairs, preserved ambiguities, exact witness references and forbidden unsupported transfers. Grading accepts a different valid repair only if it satisfies all declared constraints. Call depth alone gives no stage credit: the live trace must show new contributions, consumed literal results and repair decisions.

Status: decomposition authored before implementation; all 27 modules and the connected fixture are authored. The authored project loads; typed canonical source replay and independent source/oracle review pass. The first live full-project diagnostic is underway, with semantic output and literal-consumption qualification still pending. Model-facing adoption requires about 48 live samples per variant or explicitly recorded debt.

## Reused department visibility

World, epistemics, physical continuity, motifs, candidate replay and audit are package-root callable departments. `studio` and candidate replay declare their shared `uses` explicitly. Their child folders retain specialist ownership and type inheritance. This avoids treating a reused specialist as an unrelated descendant of a still-loading entrypoint. The manifest is host-only under `training/semantic_applications/sol/witness-rewrite-studio`; it is excluded from model-visible source.

The first live diagnostic exposed a positional service-argument error. The exact service now validates shapes and returns a one-sentence typed recovery instruction; the current running snapshot stays unchanged. The corrected declaration/error presentation is unmeasured and recorded in `plans/MODEL_FACING_CHANGES.md`.
