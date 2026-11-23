# Codemod Studio (`@natlang/app-codemod-studio`)

Hybrid crisp + semantic repository migration engine with transactional `proposeAndVerify` repair and verified crisp rule mining.

## Architecture

- **`types.ts`**: `MigrationSpec`, `CrispRule`, `SourceFile`, `FileResult`, `MigrationReport`, `MinedRuleCandidate`.
- **`transformFile.nl`**: migrates one file not covered by crisp rules while resolving verifier feedback.
- **`mineCrispRule.nl`**: synthesizes a reusable crisp rule from verified semantic migrations.
- **`studio.ts`**: applies crisp rules first (0 LLM calls on covered files), runs `proposeAndVerify` over a `Folder` workspace on uncovered files against a verifier (`tsc` + tests), and verifies candidate mined rules before admitting them.

## Verification

```bash
node ts-host/bin/natlang.mjs check applications/codemod-studio
node --test ts-host/test/flagships-tier3.test.mjs
```
