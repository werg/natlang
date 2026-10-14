# Seed Lineage Laboratory

## Closed fictional task

Interpret a seed bank's pedigree, expression theories, observation protocols and living-stock ledger. Reconstruct the surviving joint biological hypotheses, update their source-declared prior masses, and propose an adaptive experiment policy that identifies the requested expression mechanism. Preserve unresolved genotypes when the goal does not require them. This is a fictional, fully specified inheritance system; no real breeding convention supplies missing rules.

The application is a belief-state experimental designer. It differs from a classification worksheet: an observation belongs to a biological compartment, a stage and an assay regime; a cross has oriented parents; an experiment consumes particular living material; evidence changes joint worlds rather than independent marginal labels. A policy branch owns its remaining material and its posterior. Hypothetical sibling branches cannot spend each other's resources.

Target: 27 connected NL callables, approximately 600–800 substantive source lines, a declaration-only type file, six scoped semantic literal sites, nested evidence directories and an external exact verifier. Source/type replay, independent semantic review and live execution remain separate gates. No qualification yet.

## Typed uncertainty and update landscape

- `SourceClause` and `Witness` retain original IDs, paths and exact text.
- `PedigreeReading` distinguishes physical parentage, accession labels and merely proposed parentage. `GenotypeDomain` retains alternatives with source grounds.
- `ExpressionTheory` owns the interpreted expression rule and the tissue/stage in which it applies. `AssayReading` owns what a reported outcome can distinguish, including a typed unusable status.
- `JointWorld` binds one source theory to a complete accession genotype/cytoplasm assignment and a source-declared integer prior mass. Correlated tickets are never multiplied as independent marginal priors.
- `WorldPrediction` is a typed possible-outcome distribution with source witnesses. `ObservationAssessment` compares an actual source observation with that prediction and has `compatible: boolean | null`; missing interpretation preserves an unresolved branch.
- `Posterior` retains survivors, explicit rejected worlds/reasons and rational normalized mass. A likelihood-zero observation removes a world; a missing observation is not likelihood zero.
- `ExperimentPlan` binds mother/father, biological material, stage, regime, readout and stock costs. `ExperimentPartition` joins each surviving world to its actual predicted observable answer.
- `BeliefNode` owns the current posterior, remaining original experiment slots, exact remaining stock, and the goal classes still possible. Worlds may remain distinct after their goal class agrees.
- `PolicyCandidate` contains the actual experiment, all nonzero-probability outcome branches, their returned child policies, expected rational cost, worst-case cost and a witnessed optimality comparison.
- `LaboratoryCertificate` exposes interpreted evidence, posterior, per-experiment predictions, a chosen policy, alternatives, unresolved issues and source coverage. Dynamic world/node IDs are alpha-renamable; original accession/theory/experiment/clause IDs are exact.

## First world: Lantern accession

All facts are source-visible. At one nuclear locus, alleles are A/a and genotypes AA/Aa/aa. The replica-complete nursery produces exact gamete-combination proportions; this is not a random finite sample. Cytoplasm follows the mother. Three competing source theories define adult lamp expression: D lights when at least one A is present; R lights only for aa; M lights only with amber maternal cytoplasm. Seed-coat color is a different maternal tissue and does not establish adult expression.

Accession U has a verified aa mother, a father whose relevant allele remains unresolved, amber cytoplasm and a bright adult observation under a valid cool assay. Original joint tickets cover all nine theory/genotype pairs with masses D:4/12/2, R:3/4/2, M:2/1/1 in AA/Aa/aa order. Pedigree excludes AA, adult observation excludes D-aa and R-Aa. Four surviving worlds have masses12,2,1,1; the two M worlds differ in U's genotype and must remain separate.

Tester T is aa with clear cytoplasm. Available experiments include S, a forward U-mother cross scored only for uniformity at cost1; C, a reciprocal T-mother cross scored for the full adult bright fraction at cost3; G, a U genotype probe at cost2; F, a maternal seed-coat image at cost1; and W, a warm regime whose loss of assay sensitivity makes it unsuitable. Each has explicit material and original one-use slot identities. No experiment can be repeated under a new generated name.

S predicts mixed for D and uniform for R/M. C predicts half bright for D, all bright for R and all dark for M. G can separate genotypes without necessarily separating mechanisms. F's coat result is constant and cannot distinguish the goal. The requested goal is mechanism D/R/M, not complete genotype. The policy objective is exact expected cost under the declared joint prior, then worst-case cost, then source experiment ID as the visible final tie convention. The intended source-derived best policy is S first, stop on mixed, and C on uniform: expected7/4, worst4. C alone costs3. This paragraph is author review metadata and is excluded from model-visible files.

## Callable graph: owned transformations

Paths below are relative to `designExperiments`; each helper is physically nested beneath its actual caller.

| Module | Owned semantic work and consumed child result |
|---|---|
| `designExperiments` | Compose actual source, posterior, experiment predictions and policy certificate. |
| `readBank` | Folder reducer reads original inventory and assigns source roles. |
| `readBank/readPedigree` | Separate accession identity from genetic parentage; infer allowed assignments. |
| `readBank/readPedigree/judgeParentage` | Scoped literal decides one claimed parentage relation and its uncertainty. |
| `readBank/readTheories` | Preserve competing expression rules, compartments and source prior tickets. |
| `readBank/readAssays` | Interpret stage/regime/readout limitations. |
| `readBank/readAssays/judgeObservationScope` | Scoped literal decides whether one observation measures the claimed biological quantity. |
| `readBank/readStocks` | Join stock identity, consumption, capacity and experiment slot eligibility. |
| `reconstructBelief` | Own finite joint-world construction and sequential observation updates. |
| `reconstructBelief/expandAssignment` | One typed frontier step extends a pedigree-consistent joint assignment. |
| `reconstructBelief/assessObservation` | Compare a world with one actual observation using its returned prediction. |
| `reconstructBelief/assessObservation/predictObservedTrait` | Interpret the theory under the actual accession/stage/regime. |
| `reconstructBelief/updatePosterior` | Consume compatibility/likelihood, retain unknowns and normalize exact mass. |
| `predictExperiments` | Own per-world/per-experiment predictions and valid observable partitions. |
| `predictExperiments/interpretCross` | Scoped literal binds oriented parents, generation and actual material. |
| `predictExperiments/deriveOffspring` | Produce source-permitted nuclear and cytoplasmic offspring classes. |
| `predictExperiments/readOffspring` | Scoped literal maps actual offspring classes through theory and readout. |
| `predictExperiments/partitionAnswers` | Merge observationally identical answers while preserving goal/world membership. |
| `planPolicy` | Build the finite belief/stock subset state space and consume bottom-up policy values. |
| `planPolicy/expandBeliefNode` | Choose an unexpanded original state; create valid experiment/outcome successors. |
| `planPolicy/expandBeliefNode/judgeExperimentUse` | Scoped literal decides material feasibility and goal relevance from typed predictions. |
| `planPolicy/valueBeliefNode` | Choose a dependency-ready state and evaluate all actual child policy values. |
| `planPolicy/valueBeliefNode/compareCandidates` | NL comparison of exact cost vectors, goal completion and source tie rule. |
| `auditCertificate` | Audit posterior, all reachable policy branches and source obligations. |
| `auditCertificate/challengeCompartment` | Scoped literal challenges maternal-tissue and reciprocal-orientation shortcuts. |
| `auditCertificate/checkBranchStock` | Audit isolated stock changes and non-reused original experiment slots. |
| `auditCertificate/explainResidualUncertainty` | Distinguish solved mechanism from intentionally unresolved genotype. |

## Structural iteration

World construction consumes a finite source-derived assignment variable frontier; each step consumes one original joint ticket, joining all its assignments with established pedigree constraints. The observation reducer consumes one original observation identity, retaining all surviving or unresolved worlds.

Planning does not recursively call a named ancestor. It first constructs a finite closure of states identified by surviving source-world content and conditional rational weights, remaining original experiment slots and exact stock. Each transition consumes one original experiment slot; every path is strictly shorter in that set. A worklist may grow, so its length is not used as a decreasing measure. The `iterateOn` finished predicate is that every reachable state in the bounded subset closure is expanded. Each successful expansion marks one previously unexpanded state permanently, and all successors belong to the finite original subset/stock universe.

A second reducer values dependency-ready states from leaves upward. Its measure is the number of unvalued original closure-state IDs. Every step consumes one; a missing dependency or ambiguous source contract yields a typed unresolved terminal state. Candidate expected cost is exact rational arithmetic over returned branch masses and costs. NL owns feasibility, goal equivalence, answer partition and candidate comparison.

## Exact seams and their justification

- Folder reads and immutable source hashing use existing runtime mechanisms.
- Parent-owned witness identity and exact quote membership prevent invented attribution; they do not decide genetic meaning.
- Rational arithmetic, finite set identity, stock subtraction after an NL-selected experiment and generated reference-domain checks are exact mechanics.
- The external verifier independently enumerates this fixture's finite source-defined inheritance possibilities and experiment partitions, checks complete branch coverage and expected/worst cost, and accepts generated-ID renaming. It is excluded from the model-visible application and cannot supply policy choices to NL stages.
- No crisp semantic parser, experiment selector or hypothesis eliminator is installed as an application helper.

## Qualification plan

Author source and connected fixture first, then a full typed canonical certificate and positive/negative verifier controls. Obtain independent Sol reconstruction before source readiness. A small live diagnostic must show actual parentage/scope/cross/readout/use/comparison decisions entering typed state, including the uninformative coat and invalid warm assay. A successful canonical replay only certifies packaging/type compatibility. Model-facing source remains debt until measured; no training admission is implied.
