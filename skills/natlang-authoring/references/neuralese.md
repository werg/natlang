# Neuralese soft state and self-improvement (`natlang improve`)

When a program passes large documents, histories, or accumulated state between semantic calls, or improves from its own
recorded runs, natlang exposes both capabilities through the same typed functions and `@natlang/node/std` combinators
you already use.

## `Neuralese<T>` soft values

`Neuralese<T>` is a typed, opaque compressed state that carries meaning between model calls without re-serializing large
text payloads on every hop:

- **Creating a view:** use `compactValue` or `compactParams` from `@natlang/node/std` (or `view` from
  `natlang:neuralese`) to compact a large value into `Neuralese<T>` on a Neuralese-capable runtime.
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
2. **Improve a program directory transactionally:**
   ```bash
   natlang improve ./my-program
   ```
   `natlang improve DIR` builds `support`, `dev`, and `confirmation` splits from `examples.json` and recorded calls,
   proposes instruction/helper/guidance updates in an isolated snapshot, verifies them with `natlang check` and
   `natlang test`, rejects any proposal whose confirmation pass rate drops, and installs accepted changes with a
   byte-identical rollback record (`natlang improve rollback <record>`).
3. **Adopt a single call's transactional fix:**
   When a call repairs a non-protected helper via `edit_code`, review and install it with:
   ```bash
   natlang improve adopt-call <CALL_ID>
   ```
   Protected verifiers (`ensures:`, `protect:`, `types.ts`) are never editable by the model.
