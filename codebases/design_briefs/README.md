# One hundred structured natural-language programs

Browse the [complete application index](INDEX.md) for the 100 entry points,
independent cases, and fixed application splits.

Each numbered directory implements one of the supplied application briefs as a
standalone native natlang project. Start at `runCase.nl`, read the concrete
domain contracts in `types.ts`, and follow the callable folder into its five
domain stages and their nested procedures. `brief.json` binds the original
brief, algorithms, gates, and operational details. `DECOMPOSITION.md` explains
the semantic responsibilities and exact host boundary.

The programs produce typed, evidence-backed review proposals. They do not imply
that external approvals, database commits, integrations, or user interfaces have
been deployed. Authority and exact external effects belong to independently
implemented host services.

Independent scenario inputs and oracles, reproducible native IR packaging,
teacher commands, RL preparation, split rules, and validation evidence live in
[`training/semantic_applications/design_briefs`](../../training/semantic_applications/design_briefs/).

All source authors used the repository's
[`natlang-authoring`](../../skills/natlang-authoring/SKILL.md) contracts.
Illustrative generic types from the briefs are expanded into concrete native
aliases where the current loader requires them. No generated assistant answers
or model executions are included in these source packages.
