# Task Author (`@natlang/task-author`)

Semantic-FP training task author and qualifier (Tier 0 campaign tool, WS3 T3).

## Architecture

- **`draftTask.nl`**: authors a `TaskSeed` into a complete `AuthoredTask` (manifest, source files, reference answer, negative controls, and cue-word `keywordBaseline`), refining `previous` on repair rounds.
- **`author.ts`**: runs exact structural & held-custody checks (`checkTaskStructure`) and oracle replay checks (`replayProblems`) inside `semanticRepair`, then gates admission on escalated `isSemantic.nl` and `reviewOracle.nl` decisions.
- **`main.ts`**: runnable CLI target (`natlang run applications/task-author -- seed.json`).

## Usage

```bash
natlang run applications/task-author -- seed.json
node --test ts-host/test/tier0-tools.test.mjs
```
