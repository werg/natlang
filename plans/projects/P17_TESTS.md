# P17 — Living specification and test explorer

Status: finite dependency-plan explorer implemented in `codebases/test_explorer`,
`applications/test_explorer.py`, and `scripts/run_test_explorer.py`. Teacher runs
and multi-target exploration remain unmeasured.

## Natlang prerequisites

C3 provides checked load/run of programmes supplied as data; C4 supplies structured failure/reduction evidence; C2 controls attempted reproduction. C0 handles semantic obligation discovery and bounded test/refinement loops. C6 is optional for independent scenarios. No global testing or debugger tool is needed.

## Programme and typed boundary

`explore.nl(program, contracts, budget) -> Findings` calls `identify_obligations.nl`, `invent_case.nl`, `assess_failure.nl`, `choose_shrink.nl` and `explain_finding.nl`. Exact helpers validate generated input, execute checks and test whether a candidate shrink retains the relevant failure.

Records contain source revision, declared behavior, generated inputs, expected predicates, observed traces and reproducibility evidence. Expected outcomes are not automatically trusted because the same model wrote both the test and the programme interpretation. An uncertain oracle becomes a question, not a confirmed defect.

## Crisp environment

Expose source inspection, child-run invocation, contract evaluation and bounded trace reads. Reuse P05's exact type APIs where helpful. Search source/trace data through eval code, with no ambient access to unrelated runs.

Each generated trial starts from a reset fixture environment. A shrinking step cannot silently retain mutated native state from its predecessor. Child budgets include model and host work. The host returns observable facts; natlang explains whether they conflict with the supplied contract.

## Reduction and stream shape

A bounded Iterate proposes a scenario, executes it, analyses results and chooses another or stops. Minimise a failure with a crisp or mixed worklist constrained by type validity and the failure predicate. When model behavior is stochastic, measure repeat frequency under specified seeds; do not equate one non-reproduction with a fixed bug.

A later UI may consume test-result streams, but the first explorer is finite. Schedule exploration for C5/C6 records exact delivery schedules and only asserts invariants required by that semantics.

## Delivery and checks

1. Target current dependency-plan and saga examples with manually specified contracts.
2. Discover seeded defects and minimise cases. Gate: reported counterexample loads, runs and violates an independently justified check.
3. Add false-oracle and nondeterministic cases. Gate: uncertain findings are labelled and invalid fixtures rejected.
4. Extend to stream schedules and behavioural compatibility between package revisions.

Test type-invalid input generation, unrelated failures during shrinking, changed source, incomplete traces, state leakage and attempts to weaken the contract to make a run pass.

## Trace and training

The first explorer uses an independently written graph oracle. Natlang selects
offered cases and explains observations; the host validates inputs, runs a fresh
target runtime for each case, checks task identity, dependency and completion,
and tries one-task deletions under the same seed and failure code. Reports retain
source revision, input hashes, model seeds, trace digests and shrink attempts.
The oracle does not judge semantic priority among ready tasks. Runtime failure
or invalid output remains unknown. The next gate is a live teacher run over all
four scenario groups with manually reviewed traces, followed by a separate saga
effect and queue contract.

Capture proposal, actual execution, oracle identity and minimisation history. Accepted cases become linked contrast families: correct behavior, execution mistake, legitimate blocker and verified repair. Keep each family in one split. Rejected hypotheses remain useful diagnostic data but are not supervised success targets. This is an early consumer of trace-as-data and can improve the other projects' fixtures without becoming the sole judge of its own outputs.
