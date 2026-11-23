# Neuralese soft state and self-improvement (`natlang improve`)

When a program passes large documents, histories, or accumulated state between semantic calls, or improves from its own
recorded runs, natlang exposes both capabilities through the same typed functions and `@natlang/node/std` combinators
you already use.

## `Neuralese<T>` soft values

`Neuralese<T>` is a typed, opaque compressed state that carries meaning between model calls without re-serializing large
text payloads on every hop:

- **Where views come from:** with `carry: 'neuralese'` the combinators write each view with the runtime's builtin
  view operator, the same one that compacts a long argument in a call's opening (faithful without instructions; with
  `viewInstructions`, it keeps what their purpose needs). It needs a runtime with a Neuralese server; elsewhere
  `carry: 'neuralese'` is refused (`neuralese-unsupported-backend`), so leave `carry` out or pass your own `view`.
- **Carrying soft state through combinators:** `semanticReduce`, `semanticMap`, and `fixedPoint` accept
  `{ carry: 'neuralese' }`:
  ```ts
  import { semanticReduce } from '@natlang/node/std';

  const summary = await semanticReduce(sections, initialState, mergeSection, {
    mode: 'tree',
    carry: 'neuralese',
  });
  ```
  When `carry` is omitted (`'text'`, the default), values stay ordinary typed JS values. With `carry: 'neuralese'`,
  intermediate states travel as compressed soft views and are read once at the boundary.
- **Opacity:** never interpolate or inspect a `Neuralese<T>` value directly in code—pass it as a typed argument to the
  consuming `.nl` or `nl` function, or read it explicitly at the end.

## Profiling and improving a program from its own runs

1. **Profile where time, tokens, and uncertainty go:**
   ```bash
   natlang profile ./my-program
   ```
   `natlang profile` aggregates the machine's call store per function (calls, p50 latency, average tokens, compiled
   crisp share, decision confidence p10/p50, escalation rate, failure rate) and runs the `nextOptimization` policy to
   recommend the next concrete step (`readout: decision`, `natlang specialize`, `natlang tune`, or `natlang improve`).
2. **Improve the program from its own runs:**
   ```bash
   natlang improve ./my-program --model MODEL            # propose from the runs, check, adopt
   natlang improve ./my-program --model MODEL --dry-run  # propose and check, install nothing
   ```
   A natural-language improver on MODEL reads the program's recorded calls in the call store: calls that failed, decisions
   `escalate` sent to the fallback (with both answers; a disagreement is a labelled hard case), results that broke an
   `ensures:` postcondition, and calls a tester marked with `natlang traces bad|fix`, plus the support examples the
   program answers wrongly. For at most `--max-functions` (3) of the most troubled functions it diagnoses what the
   troubled calls share, decides whether the instructions are at fault, and rewrites only their instructions; the front
   matter (signature, `protect:`, `ensures:`) stays byte-identical. The gate then builds `support`, `dev`, and
   `confirmation` splits from `examples.json` and trustworthy recorded calls (the fallback's answer stands for an
   escalated decision; a fixed call's expected value is the tester's), runs `natlang check` and the dev and
   confirmation cases on a copy with the change, rejects it if either pass rate drops (or if it edits `types.ts` or a
   `protect:`ed source), and installs an accepted change with a byte-identical rollback record
   (`natlang improve rollback <record>`). The improver's own calls are recorded like any other.
   To check a change you wrote instead: `natlang improve ./my-program --proposal change.json --model MODEL`, where
   `change.json` is `{ "files": { "PATH": "new text" or null } }`. `--dry-run` without `--model` prints the splits.
3. **Adopt a single call's transactional fix:**
   When a call repairs a non-protected helper via `edit_code`, review and install it with:
   ```bash
   natlang improve adopt-call <CALL_ID>
   ```
   Protected verifiers (`ensures:`, `protect:`, `types.ts`) are never editable by the model.
