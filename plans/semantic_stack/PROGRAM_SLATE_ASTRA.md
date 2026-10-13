# Astra semantic program slate

Owner request, 2026-10-13: at least 100 distinct, inventive, sizable natlang programs spanning at least 15 domains, with real semantic understanding, structured and directory processing, deep calls, iteration and ad hoc natural-language function literals.

## Delivered design and honest execution status

`astra-programs.json` contains **120 individually authored program designs across 20 domains**, six per domain. Each entry has a concrete objective, distinct semantic operation, source evidence, typed input and output contracts, six named stages, a scoped language literal with numbered instructions, a structured provisional oracle, and a task-specific counterfactual. This is a program design catalog, not a claim of 120 completed live executions.

Eight priority programs have **24 manually authored interacting records each**, with cross-record dependencies and source-grounded expected results. The other 112 have runnable diagnostic seed fixtures and a recorded expansion specification; they remain held for production generation until the expansion is materialized and reviewed. Their seeds do not count as sizable training examples. None of these records is training-admitted or live-teacher-qualified by this document.

The complete machine-readable designs are the primary artifact. The source-pool integrator owns actual `.nl` materialization, scoped access, native runtime replay, teacher sampling, corpus registration and manifests. Astra owns this catalog and its review. Sol authors a separate catalog in `training/semantic_stack/sol_programs.json`; the domain allocation was coordinated to avoid duplicate task families.

## Expansion progress after the first cohort

The catalog now contains **80 substantial authored fixture views** of at least 24 records each. Thirteen have passed independent source/oracle review and exact checks. The others are held pending independent review/replay; 40 designs remain seed-only. Recent programs share coherent domain worlds with `world_family_id`; distinct program views do not count as independent worlds. Every program still requires live executor qualification and training admission. The first-cohort discussion below records its original eight cases.

Four strongly patterned worlds (archaeology-02/03/05 and folklore-04) are explicitly tagged diagnostic-first rather than mass-generation priority. Calendar-year dating permits deposition earlier in the same floor-construction year; object biography evidence now explicitly states manufacture location.

## Natlang decomposition before implementation

A program starts as a typed natural-language function over a folder snapshot. Its decomposition follows the domain algorithm rather than a shared classify-and-filter template.

| Part | Form | Reason |
|---|---|---|
| Entry objective and success contract | Named NL function | States the actual semantic task, uncertainty and required output. |
| Domain planning and reconciliation | Named NL stages | Decides which relations, hypotheses, edits or actions mean what. |
| Local interpretation in a particular case | Scoped `nl` literal | Binds the current item, rule and related evidence; its numbered algorithm is supplied in each entry. |
| Graph traversal, interval arithmetic, set equality, hash identity | Existing exact helpers | These operations have exact meanings and should not consume model judgment. |
| Inventory and file access | Existing folder/service mechanisms | Effects and external access remain traceable. |
| File mutations | Pure patch data followed by the existing bounded transaction | Makes changes reviewable and preserves exact commit behavior. |
| Quality and promotion judgments | Existing natural-language review program | Semantic correctness, ambiguity, novelty and usefulness are policy. |
| Admission receipt and manifest verification | Existing crisp verifiers | Exact artifact identity and recorded gate results must not be guessed. |

`skills/natlang-authoring/SKILL.md` and its language/pattern references informed these boundaries. This catalog introduces no runtime syntax or semantic feature and therefore needs no authoring/integration skill change. The source instructions are new model-facing content: they require roughly 48 live samples per meaningful variant or explicit model-facing debt before adoption. Source inspection and a static replay do not satisfy that requirement.

The six labels in `call_chain` are an intended meaningful decomposition, not a measured callstack. A builder must nest real calls through those responsibilities, including the local literal, and inspect traces. A root function that reads everything and returns an answer while never calling its children fails the depth requirement even if the answer is correct. Neither unused helper files nor long prompts create call-depth credit.

Use finite walks over source items, edges, candidate assignments or intervals. For recursive exact tree helpers, recurse on strict subtrees. For semantic refinement, use the existing `iterateOn` with an explicit decreasing unresolved-work measure or meaningful termination predicate. A natural-language function must not appear in its own caller chain. No arbitrary turn or nesting cap is proposed.

## Twenty domains and 120 distinct tasks

| Domain | Six programs |
|---|---|
| Archives | Expedition chronology; identity authority with cannot-links; purpose-sensitive redaction; provenance-preserving finding aid; document descent with contaminated copies; catalogue amendment merge. |
| Archaeology | Stratigraphic cycle localization; sherd refit assembly; deposition dating with intrusive material; discriminating survey selection; reused-material biography; recovery-method-matched assemblage comparison. |
| Folklore | Causal motif alignment; nested taboo permissions; retelling dependence; accessible participatory adaptation; narrator/character belief tracking; shared narrative spine with alternative endings. |
| Museum curation | Loan/rest/transit schedule; ownership versus exhibition provenance gaps; layered evidence-qualified labels; thematic spatial route; component versus duplicate accession; reversible deinstallation. |
| Ecology | Conflicting signed food-web paths; detection-process-aware occupancy evidence; seasonal functional matching; connected habitat Pareto frontier; temporally constrained introduction routes; rival indicator explanations. |
| Conservation | Material-compatible treatment sequence; condition-report semantic diff; storage assignment with preferences versus requirements; region-specific reversibility boundary; diagnostic monitoring coverage; pending-decision-preserving handoff. |
| Disaster logistics | Observation-versus-forecast route repair; functional relief-kit substitutions; household group placement; independent-origin rumor tracing; receipt-aware delivery handoff; prerequisite restoration schedule. |
| Transit | Accessible deadline-constrained journey repair; landmark-based stop identity; scoped transfer guarantees; effective service calendars and short turns; passenger-consequence holding decisions; wayfinding graph repair. |
| Agriculture | Crop rotation constraints; deadline-aware canal turns; blocked trial reconstruction; seed mixture lineage; partial-information harvest readiness; spatial anomaly hypotheses. |
| Food systems | Transitive ingredient exclusions; targeted batch lineage hold; functional menu substitutions; storage-record interval gaps; perishable recipe resource flow; scope-correct supplier provenance claims. |
| Publishing | Moved-content cross-reference repair; analytical index sense hierarchy; author/copyeditor intent merge; argument-preserving page cuts; work/translation/edition identity; scoped house-style transformations. |
| Localization | Interface action-intent labels; discourse and grammatical reference; independently scoped plural variables; sense-aware terminology consistency; metaphor pragmatic adaptation; regional fallback compatibility. |
| Theatre | Dramatically valid blocking repair; prop custody continuity; cast-and-prerequisite rehearsal schedule; observable-event cue alignment; motivation-preserving scene cuts; understudy doubling with quick changes. |
| Games | Narrative quest locks and unlocks; nested tabletop exceptions; progressive hints; character-local belief simulation; asymmetric cooperative plans; legal rule-interaction counterexamples. |
| Sports tactics | Passing paths under defender commitments; phase-specific role mapping; skill-and-participation roster assignment; causal drill comparison; timed set-play conflicts; conditional scouting transfer. |
| Accessibility | Equivalent interaction paths; semantic reading order; purpose-sensitive image-description facts; exception-preserving plain language; personal sensory route; minimal form-error repair. |
| Civic deliberation | Claim-level argument map; incompatible amendment composition; stakeholder-interest coverage; earmarked project portfolio; procedural decision reconstruction; interest-based compromise. |
| Education | First invalid reasoning transition; evidence-backed prerequisite plan; criterion-specific rubric calibration; minimal counterexamples; feedback dependency graph; skill-covered learning role assignment. |
| Manufacturing quality | Process-history contrast; failure-mode inspection cover; model-and-serial instruction revisions; invariant-preserving rework; requirement-specific deviation evidence; reference-based measurement drift. |
| Scientific reproducibility | Meaning-changing protocol diff; exact artifact provenance; registered exclusion audit; discriminating reproduction ablations; sample-dependent evidence synthesis; design-limited research claims. |

These tasks share useful exact abstractions without making the same judgment. For example, a food composition graph requires interpreting exhaustive component declarations, a quest graph distinguishes mentioning an item from granting it, and a citation graph distinguishes a copied report from an independent observation. Those distinctions must remain in the source and oracle; renaming graph nodes alone earns no distinct-program credit.

## Source choices and provenance

All current records are **original authored fictional worlds**. This avoids pretending that a plausible invented fact is a sourced scientific, medical, legal or operational recommendation. Crop rules, material protocols, game mechanics, survey relationships and institutional rules are stated in each task. The model is judged on using that world, not on whether it recalls external domain facts.

Every fixture provides `source_fixture.files`, records with stable IDs and paths, and a source note. The oracle is a separate field and must stay outside model-visible task files and callable folders. A record may legitimately state facts needed to derive an answer; it must not contain an answer label disguised as a source note. Licenses are recorded facts; no license review hold is proposed. A future external source addition needs its own source identity, immutable bytes, split decision and manifest rather than replacing this authored provenance.

These synthetic seeds establish controllable semantics and exact counterfactuals. They are not a substitute for later independently sourced messy language. Teacher generation can expand prose, conflicting testimony and plausible distractors, but must preserve an explicit world model and a separately reviewed oracle. A generated paraphrase should be rejected if it accidentally changes a quantifier, exception, chronology, referent or certainty level.

## Full-size priority cohort

| ID | 24-record connected case | Independent exact boundary |
|---|---|---|
| `astra-archives-01` | Eighteen-event voyage chronology, second voyage, shared ship name, alias and unbound scrap. | Precedence graph and total order; retain the unresolved voyage assignment. |
| `astra-archaeology-01` | One minimal four-context cycle with a long connected stratigraphic tail. | Physical-relation orientation, exact edge set and minimal cycle membership. |
| `astra-museum_curation-04` | Directed gallery graph with semantic distractors and a requested three-theme order. | Valid route, semantic theme coverage, narrative order and shortest length. |
| `astra-ecology-01` | Several signed food-web branches, conflicting paths and downstream uncertainty. | Sign products and disagreement propagation, without invented strengths. |
| `astra-transit-04` | Fifteen trip instances, calendar rules, extensions, a reroute, cancellations, a withdrawal and tomorrow-only notice. | Effective service by date and destination, including scope of withdrawal. |
| `astra-food_systems-01` | Nested component DAG, mixtures, terminal declarations and propagated unknowns. | Exhaustive transitive composition; known exclusion versus unresolved composition. |
| `astra-games-01` | Eleven-stage unlock chain, two unreachable cycles, misleading mentions and promises. | Reward replay from empty inventory and unreachable strongly connected components. |
| `astra-manufacturing_quality-03` | Three models, overlapping revisions, a partial withdrawal and sixteen boundary work orders. | Inclusive interval partition, explicit supersession and authority. |

For the remaining 112, each `expansion_recipe` binds the program ID, semantic operation, exact counterfactual intervention and 24-record linked-world construction. Its status is `specified_pending_materialization`. A random seed is not a determinism guarantee for a teacher. Reproducibility means freezing the recipe, provider/generator revision, produced files and oracle hashes. Generation-ready status remains false until those artifacts exist and meet the gates.

## Exact checks, teacher judgments and reward

`closed_world_exact` marks programs whose supplied finite rules and world state support an exact reference **after source relations are faithfully interpreted**. It does not certify the model interpretation. `hybrid` marks tasks involving paraphrase, purpose, evidence strength, causal scope, attribution or permissible alternatives requiring independent semantic review. Every entry retains `teacher_review_required: true` until that review is actually recorded.

The catalog expected value is either canonical semantic facts or a representative valid witness. Scheduling, rewriting and planning often admit several answers. Do not reward only one authored wording or one arbitrary schedule. Check constraints and equivalence classes; use exact tie-breaks only when the source contract states them. Byte equality belongs to IDs, hashes and exact arithmetic, not semantic paraphrase.

A useful reward record keeps components separate:

1. **Contract and execution:** valid types, referenced IDs exist, no invalid effects, result returned through the real runtime.
2. **Semantic relations:** evidence-grounded interpretation, correct scope, negation, identity, chronology and uncertainty.
3. **End-to-end objective:** the output actually solves the declared task or returns a proved infeasibility/ambiguity.
4. **Invariants:** conservation of quantities, complete source accounting, admissible paths, satisfied prerequisites, preserved valid content.
5. **Counterfactual behavior:** the meaningful intervention changes the appropriate downstream result; order and opaque-ID permutations do not.
6. **Trajectory usefulness:** real helper consumption, growing and shrinking state, scoped literal calls, meaningful recovery and no fabricated evidence.

Do not let high schema validity compensate for wrong meaning, or let a long trace compensate for a wrong final result. Partial credit should correspond to locally correct relations and preserved obligations, giving the student a useful improvement gradient before complete success. The exact scalar weighting is a shared training-policy decision and should use the existing natural-language policy mechanism, not a catalog-local Python heuristic.

Reviewers should see source evidence and task requirements independently of the proposed answer. Use a second teacher or a separately authored deterministic reference when feasible. Sample teacher disagreement and adjudicate actual ambiguous cases. Split by world construction, entity graph and paraphrase family; a held example cannot be a name-substituted training graph. Keep whole trajectories, with prompts and tool outputs at lower weight as required by the training foundation.

## Formulaic-duplicate and failure risks

- **Domain paint over one selector:** reject programs differing only in nouns, labels or descriptions. Compare operation, relation ontology, control flow, objective, verifier and failure modes.
- **Toy fixtures called sizable:** seed-only status is explicit. Require connected multi-item evidence, actual deep calls and a meaningful iteration trace before full-size credit.
- **Padding as complexity:** duplicate prose and irrelevant files do not count toward 24 records. Each retained record contributes a fact, constraint, scope decision, contradiction or specific counterfactual.
- **Answer leakage:** expected results and hidden-world facts remain outside model-visible sources. Inspect generated task files for answer-key phrases.
- **Exact-answer overreach:** a canonical witness is not the sole valid solution. Source contracts and semantic equivalence govern acceptance.
- **Uncertainty erasure:** explicitly unknown composition, identity, custody or causal strength cannot be silently filled in.
- **Narrative external knowledge:** a familiar domain fact cannot override the supplied fictional protocol.
- **Fake depth:** helper definitions alone earn nothing; require actual nested calls and evidence that their values influence the result.
- **Oracle-generator common bug:** compare source interpretation against an independently written reference and case-level human/teacher review, especially at interval boundaries and negated conditions.
- **Premature adoption:** catalog creation, typed source checks and authored replay are distinct from live sampling, source clearance and training admission.

The correct next step is to materialize and sample the eight substantial fixtures first while expanding and reviewing the other 112 program designs. Failures should trigger a broad implementation dragnet—rendering, scope, captures, masks, source access, exact helpers, output typing and oracle ambiguity—before rejecting the program family.

## Checks recorded for this revision

On 2026-10-13, an independent exact-reference review of the eight full fixtures passed: signed graph propagation, stratigraphic edge orientation and cycle enumeration, shortest gallery path, quest reward replay, material composition closure, effective trip calendars, instruction interval precedence, and the independently transcribed chronology edges. The catalog contains 120 unique IDs and 120 distinct operation names, with six entries in every domain. Operation-name uniqueness is a schema check, not a semantic-novelty proof.

The repository quick check passed: 22 Python ratchet tests and the TypeScript check. No live executor or teacher samples were run by this author. The source integrator independently reviews the full fixtures and owns subsequent replay and live qualification. Two seed ambiguities were corrected during author review: the sensory-route traversal now explicitly overlaps the noisy interval, and roster assignment has a stated tie-break for its otherwise interchangeable slots.

The source-pool integrator independently reviewed all eight 24-record fixtures and their semantic oracles on 2026-10-13, requesting no oracle corrections. That review found a separate builder issue: stage files emitted as siblings do not implement the intended deep callable chain. The integrator owns the nested-folder repair and subsequent exact replay/live diagnostic. Catalog source review does not certify that builder or its eventual executions.
