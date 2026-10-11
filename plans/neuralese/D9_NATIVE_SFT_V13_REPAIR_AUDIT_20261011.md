# D9 native SFT v13 repair audit (2026-10-11)

Source: `native-sft-v13-r2`, Pop `data/neuralese/corpora/native-sft-v13-r2/teacher.jsonl`, immutable SHA-256 `a55d1c9cb51693ea8b4ea07eac9dee8226ae5fa84efacd4f08a3fbc971ff7252` (111,301 rows; 105,869 train and 5,432 test). This is a representative-prefix audit, not a full-file conversion or complete incidence count.

## Confirmed repair: renamed code tools

The first 100 rows contain one train row offering the historical `read_function`, `edit_function`, and `diff_functions` tools. Current `ts-host/src/native/runtime.ts` exposes `read_code`, `edit_code`, and `diff_code` instead. The row is otherwise approved by its exact native-runtime oracle and has admitted trace data; this is a tool-name conversion, not evidence for holding the source group.

- Row: `teacher-program:02590d13f1d4c3530aa2:decision:0003`
- Source row SHA-256: `da0eec807c0858efecf211795a014ddead23293ed6edb0c6f1e4137e400f1b52`
- `source_groups`: `inline-curriculum:kqapro:kqapro_question:train_36730:q`, `kqapro:Nottingham`
- The existing `ts-host/scripts/inline-curriculum/migrate-code-tools.mjs` transformed the row in memory. It rewrote the offered schemas, added `provenance.code_tools_migration = "read_code/1"`, updated the bound program IR digest, and left no historical tool names in the transformed row. Transformed row SHA-256: `95437c217449b4c415ef975b20ef8061f5b9af71ccb8cb9ab31bbab16c9a3505`.
- Whole-corpus incidence remains unaudited. Apply this existing semantic-preserving migration to rows containing old names before current-runtime rendering; do not infer a corpus-wide hold from this sample.

## Historical system prompt migration

The existing `ts-host/scripts/inline-curriculum/migrate-system-prompt.mjs` exposes `promptHistory()`, which loads exact historical prompt strings from Git. Its history contained 78 `TOOLS_PROMPT`, 8 `APPROACH_PROMPT`, 8 `FUNCTION_TOOLS_PROMPT`, and 19 `DIRECTORY_REDUCER_PROMPT` variants in this checkout. Among the first 100 source rows, all 100 system prompts differed from the current prompt; 69 matched known exact concatenations, while 31 were rejected by `currentPrompt()` because they had unmatched trailing text. One such tail begins with task-specific instructions about selecting fields from a large input. Dropping those tails would lose program guidance.

DGX owns TS/Python prompt-scope parity. A history-aware normalizer should replace recognized historical generic components and preserve unmatched program addenda verbatim, then apply the shared scope declarations. Python paragraph-prefix heuristics are not migration authority. No full-file prompt rewrite has started; no missing-runtime-API conclusion is drawn from serialized-history compiler diagnostics.

## Remaining source checks

Compiler diagnostics in serialized histories include `console` references and redeclarations across persistent-eval turns. These contexts need native-aware replay or compiler isolation before they can establish a real current-runtime defect. Historical source compiler diagnostics alone are not grounds to exclude otherwise approved rows.
