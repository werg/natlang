# S05 — Applications with explicit beliefs and investigations

Status: evidence-linked claim graph, exact impact query and natlang investigation source work in Inquiry Lab; warranted belief revision remains unmeasured. Part of [the semantic software plan](../SEMANTIC_SOFTWARE.md).

## Product and semantic ambition

An application should retain what it currently believes, why it believes it,
which assumptions the conclusion depends on and what could change its mind.
When observations change, it should reconsider the affected conclusions and
take useful investigative action. This knowledge must influence execution and
presentation rather than exist only as explanatory text in a side panel.

Example: Fieldnotes concludes that a deployment reduced failures, conditional
on comparable traffic. A new log batch shows a shift in device mix. Natlang
reopens the comparability assumption, develops a cohort analysis, revises the
conclusion and updates the generated comparison UI. An unrelated spelling fix
in a source note should not trigger a complete reinvestigation.

## User experience

The main answer has inspectable “Why?”, “What could change this?” and “Compare
interpretations” affordances. Opening them reveals evidence and executable
analysis. The user can challenge a premise, provide a source, run a proposed
experiment or keep an alternative interpretation visible.

Changes produce an intelligible update: what was observed, which conclusion
changed, which stayed supported and what remains unresolved. Missing evidence
can become an investigation task. Natlang should perform available queries and
experiments before asking the user for information only they can provide.

## Ordinary application types

- `Observation`: source revision, actual value/host reference, observation time,
  relevant event time and collection provenance.
- `Claim`: proposition, scope, status, supporting/opposing references and current
  assessment revision.
- `Assumption`: premise, scope, origin and circumstances under which it is used.
- `EvidenceLink`: relationship and rationale, including whether it was inferred
  semantically or established by an exact computation.
- `Question`: competing possibilities and what evidence would distinguish them.
- `Investigation`: objective, outstanding work, attempted approaches, results and
  continuation/workspace references.

Use statuses such as supported, contested, withdrawn and unresolved with reasons.
Do not manufacture calibrated probabilities from model self-confidence. Numeric
uncertainty is useful when it comes from an explicit statistical calculation
whose assumptions and method are recorded.

These records are the first application's schema. S02 can later revise it. They
are not new core truth values, epistemic types or mandatory metadata on every
natlang value.

## Natlang program and investigation loop

1. `interpret_observation.nl` determines what a new datum bears on, retaining
   provenance and alternative interpretations.
2. `relate_evidence.nl` identifies support, opposition and limitations. Literal
   quotation checking is crisp; judging entailment is semantic.
3. `identify_affected.nl` follows known relationships and searches for missing
   links. An incomplete graph cannot justify claiming nothing else is affected.
4. `reconsider.nl` compares prior reasoning with new observations, updating
   conclusions and assumptions without rewriting historical assessments.
5. `choose_investigation.nl` selects an informative feasible query, computation,
   experiment or user question.
6. `investigate.nl` drives those steps, observes results, revises its approach and
   continues until the objective is met or a concrete missing dependency remains.
7. `present_findings.nl` prepares the answer and, through S03, the interaction
   appropriate to the remaining uncertainty.

Avoid turning this into a host-coded inference engine. The graph supports
retrieval and provenance; natlang determines semantic significance and what to
do next. Long investigations can loop through these functions many times.

## Host work

Start with exact text/metadata search and host-owned collections. Add SQLite
indices if actual data size needs them; vector search is optional later.
Expose record IDs and selective reads instead of serializing the entire graph
into each model turn. Return complete relevant passages when needed, not only
a lossy summary. Persist complete evidence and label previews as previews.

Supply queries, child execution, observations and immutable revisions through
eval. Search ranking does not establish truth. Quoted content remains evidence
data and cannot independently alter the execution authority or instruction set.

Store a durable work agenda and result references. A continuation note helps a
running lambda but must not be the only place that a pending investigation
exists. A fresh invocation should resume from artifacts and actual outcomes.

For streamed input, queue revisioned observation events. Begin with one state
owner. Long-running queries may finish later with the input revision attached;
natlang handles the result as an observation about that snapshot and decides
whether it remains relevant. Do not mutate an active lambda's inputs.

## Delivery sequence

1. Build an evidence-backed notebook answer with linked observations, assumptions
   and actual computations. This is the first consumer of natlang-owned state.
2. Feed contrary evidence and irrelevant evidence in separate scenarios; compare
   the resulting assessment changes and retained support.
3. Add an investigation agenda, query execution and durable resumption after
   browser reload. Prevent completed external work from being blindly replayed.
4. Add competing explanations and a distinguishing experiment. Use S01 to retain
   a useful new method and S03 to create its inspection interface.
5. Apply the same libraries to Signal room: changing incident hypotheses should
   affect selected queries and escalation recommendations. Real escalation sinks
   remain a separate integration with explicit user authorization.

## Tests and teacher tasks

Exact checks: missing/deleted evidence, invalid quotes, snapshot mismatch, broken
links, large evidence retrieval, duplicate observations, stale job completion,
recovery and preservation of complete original data.

Semantic tasks: correlation versus confounding, contradictory witnesses,
updated methods, stale measurements, missing denominators, time-dependent
claims and mutually dependent assumptions. Include irrelevant changes and
strong evidence that really should reverse an earlier conclusion.

Acceptance: evidence changes execution and conclusions appropriately; users can
inspect the support; the system investigates a resolvable uncertainty; it
preserves uncertainty when available observations cannot settle it. A populated
graph plus an unchanged canned answer does not pass.

Evaluate unsupported claims, useful revisions, unnecessary recomputation,
missed implications and quality of chosen experiments. Use independent source
sets and held-out scenarios. Capture teacher corrections and recovery steps,
then test whether the student can maintain the same discipline over long runs.

## Risks and consumers

Risks include elaborate but ungrounded belief graphs, stale dependency links,
self-confirming experiments and treating an earlier generated claim as a new
independent source. Preserve evidence origin and detect repeated provenance;
natlang still needs semantic review of the actual relationship.

Consumers include log investigation, build diagnosis, semantic type checking,
NPC memory, scheduling assumptions, media quality review and evidence-aware
publishing. Extract retrieval/provenance utilities where genuinely shared,
without imposing one domain's ontology on every application.
