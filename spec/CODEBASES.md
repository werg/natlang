# Named functions and callable folders

The normative contract is in [`SPEC.md`](SPEC.md). A named natural-language
function lives in `foo.nl`; the items it may call live in its companion folder
`foo/`:

```text
review/
  types.ts          # aliases for review.nl and everything below it
  review.nl
  review/
    assess.nl       # a natural-language child
    summarize.ts    # a TypeScript child
```

A TypeScript child is an ordinary module. A default-exported function makes the
module callable; named exports and values become attributes:

```ts
import { wiki } from 'natlang:services';

export default function summarize(assessments: Assessment[]): Report {
  return { assessments, supported: assessments.filter(a => a.verdict === 'supported').length };
}
```

Children may import each other (`import assess from './assess.nl'`), declared
npm packages, and `natlang:services`; nothing else local. They follow the
finite-iteration and no-recursion rules. The model calls items with ordinary
awaited positional calls, `await assess(observation, criterion)`, and may
inspect and edit them with the function tools; the set of items is fixed during
a call.

Application code uses the same layout for inline `nl` calls: the nearest
`natlang.d/` folder above a file is its callable context.

`codebases/` holds the corpus of named programs used for teacher data; each
directory is a small callable tree with its own `types.ts` and README.
