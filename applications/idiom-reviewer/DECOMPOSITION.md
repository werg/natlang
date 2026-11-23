# Idiom Reviewer (`applications/idiom-reviewer/`): Decomposition

Tier 0 flagship tool (WS3 I2, WS6 §2): reviews a natlang program for semantic-FP idioms beyond the exact syntactic
floor in `ts-host/src/compiler/idiom.ts`.

## Policy vs Mechanism
- **Exact mechanism (TypeScript, `review.ts`):**
  - Splits `.nl` bodies into sentence/step groups while ignoring YAML frontmatter and fenced code blocks.
  - Detects exact syntactic clues (literal code tokens, finite return types missing `readout: decision`, sibling
    `types.ts` declarations, and untrusted boundary hints).
  - Fans out concurrently over sentence groups with `semanticMap`, gates uncertain classifications with `escalate`,
    and aggregates structured findings with exact file/group locations.
- **Typed semantic policy (`.nl` functions, `readout: decision`):**
  - `classifyGroup.nl` (`readout: decision`): classifies one sentence group of a `.nl` instruction as
    `judgment` (genuine semantic criteria), `mechanism` (exact control flow, parsing, arithmetic, or bookkeeping that
    belongs in TypeScript), or `prose-interpreter` (step-by-step imperative algorithm written in English).
  - `detectCueWords.nl` (`readout: decision`): judges whether a classification/routing `.nl` relies on brittle keyword
    or cue-word shortcuts instead of meaning.
  - `detectUntypedExternal.nl` (`readout: decision`): judges whether an argument carries external user/web/diff/email
    content that should be typed as `Untrusted<T>`.
