# Wiki

A local collaborative wiki whose logic is natural language. See `DECOMPOSITION.md` for every part and its decision.

- `merge.nl`: the edit pipeline. Change summary, conflict detection, merge planning, merge, verification against each
  intent, repair with `iterateOn`. Unmergeable edits stay conflicts with all alternatives.
- `maintain.nl`: sections, links (with rename detection and repair updates), and which cell results are still fresh.
- `cells.nl`: which requested cells run, in which dependency batches, and which are refused.
- `index.ts` (`WikiWorkspace`): the mechanism. Transport identity, coverage, block shape, atomic publish, the VM for
  JavaScript cells, natlang cells, stale-on-change.

Two hot paths are pluggable through `WikiSettings`: `changes` and `staleness`, each `"crisp"`, `"nl"` (default) or `"shadow"` (`"natlang"` is accepted for `"nl"`).

```ts
const wiki = new WikiWorkspace(page, { profile, runtime, settings: { changes: 'crisp' } });
const report = await wiki.merge(wiki.snapshot(), updates);   // report.structure, report.repairs
await wiki.schedule([{ block_id: 'demo', input: 'x', origin: 'user' }]);
```

Tests: `ts-host/test/wiki.test.mjs` (scripted models). Check: `natlang check applications/wiki`.
