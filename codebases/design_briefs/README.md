# One hundred structured natural-language programs

Browse the [complete application index](INDEX.md) for the 100 entry points,
independent cases, and fixed application splits.

Each numbered directory implements one of the supplied application briefs as a
standalone native natlang project. Start at the operations declared in
`application.json`, read the concrete domain contracts in `types.ts`, and follow
the callable folders and explicit imports into the program's own reasoning graph.
The original brief's algorithms are coverage obligations, not a stage sequence. `brief.json` binds the original
brief, algorithms, gates, and operational details. `DECOMPOSITION.md` explains
the semantic responsibilities and exact host boundary.

The programs own their domain decisions, interactive events, durable state and
recovery. Their outputs include revised ledgers, diagnosed learner sessions,
shipment reservations, consent withdrawals and audience knowledge. The
[optional application host](../../applications/design-briefs-host/README.md)
executes those native operations and atomically saves their typed checkpoints.
It does not impose a shared proposal/approval/commit business process. Actual
external effects use configured exact adapters and verified receiver receipts;
a source application is not evidence of a deployed integration.

Independent scenario inputs and oracles, reproducible native IR packaging,
teacher commands, RL preparation, split rules, and validation evidence live in
[`training/semantic_applications/design_briefs`](../../training/semantic_applications/design_briefs/).

All source authors used the repository's
[`natlang-authoring`](../../skills/natlang-authoring/SKILL.md) contracts.
Illustrative generic types from the briefs are expanded into concrete native
aliases where the current loader requires them. No generated assistant answers
or model executions are included in these source packages.
