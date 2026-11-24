# Idiom Reviewer (`@natlang/idiom-reviewer`)

Semantic idiom reviewer (Tier 0 campaign tool, WS3 I2) complementing the crisp idiom floor of `natlang check`.

## What It Checks

- **Per-step kind** (`@natlang/std` `stepKind`): classifies every instruction step as `judgment`, `mechanism`, `guard-for-type`, or `redundant` (escalating uncertain calls to `teacher`).
- **Prose interpreters** (`isProseInterpreter.nl`): detects multi-step `.nl` files acting as English algorithms instead of TypeScript orchestration over small typed judgments.
- **Cue-word rules** (`detectCueWords.nl`): detects keyword-matching branches masquerading as semantic judgments.
- **Untyped external input** (`detectUntypedExternal.nl`): flags parameters carrying untrusted external text without `Untrusted<...>`, resolving sibling `types.ts` aliases transitively so records already carrying `Untrusted<...>` fields are accepted cleanly.

## Usage

```bash
natlang run applications/idiom-reviewer -- applications/pr-shepherd
natlang check --idiom applications/pr-shepherd
node --test ts-host/test/tier0-tools.test.mjs
```
