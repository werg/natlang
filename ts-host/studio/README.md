# Natlang Studio

The [Inquiry Lab](research/README.md) adds a research workspace where natlang
can author executable methods, revise representations, generate interactive
views, reconcile candidate changes and maintain evidence-linked conclusions.
It is linked from the Studio sidebar and uses the same local browser/host
infrastructure. Its current semantic quality gates are documented separately.

Twenty-two interactive applications cover the twenty project families (P08 has
three separate games). This is one shared browser shell, model loader, storage
layer and operation journal, with distinct application surfaces and programmes.

From the repository root:

```bash
npm --prefix ts-host run build
node ts-host/scripts/serve-studio.mjs
```

Open **http://127.0.0.1:8766/ts-host/studio/**. The companion binds to loopback.
Choose an app, open **Interpreter settings**, and load a published local model.
The catalog supplies the context size; the UI exposes context, root seed and
CPU/GPU selection. Loading is explicit and there are no application-added
model turn or episode limits. CPU is initially selected on this workstation.

**Use explicit controls without a model** (or `?fixture`) runs scripted
interpreter turns for individual operations. It does not generate semantic
answers. Multi-step goals such as running a notebook dependency graph require
a model; individual cell controls remain available for wiring tests.

## Who runs the application?

Natlang owns an interaction, including its algorithm, operation order,
interpretation of results, revisions and completion. A submitted event enters
`programs/<app>/reduce.nl`. That programme can call helpers repeatedly before
returning the final state. `choose.nl` interprets a request; `apply.ts` exposes
one domain operation and returns a typed `Step { state, ok, detail }`.
Natlang inspects that Step and decides what to do next. `finish.ts` returns
the chosen final Step's state.

For example, the notebook host executes **one cell**. Natlang walks its
dependency graph, detects a cycle or missing input, runs prerequisites,
inspects outputs, and decides whether to proceed. The test explorer similarly
runs **one case** per host call; natlang iterates the collection and reasons
about failures. A host `compareEverything()` or `fixTheProject()` would obscure
precisely the algorithmic work we want the interpreter to perform and learn.

Crisp operations implement mechanics: exact arithmetic, file/process IO,
SQLite execution, source replacement, citation substring checks, and DOM
projection. Their concrete constraints protect the meaning of those operations:
a trade cannot spend nonexistent money; an edited cell cannot reuse stale
results. Semantic merging, type proposals, NPC dialogue, and spell
interpretation remain model decisions.

`view.nl` produces a heading, summary, panel order and suggested next commands.
The renderer projects that view through domain-specific editors, previews,
arenas, timelines, tables and journals. Text entry, selection and animation do
not require a model turn. The model receives a submitted event instead of a
keystroke transcript. The small core language has not acquired UI primitives.

## Applications

| Application | Project | Interactive workflow |
|---|---|---|
| Cut & light | P01 | Import/sample video, choose a transform, render with FFmpeg, preview and inspect hashes/metadata |
| Confluence | P02 | Edit ancestor and two replicas across ten data families, propose a semantic merge, inspect unresolved issues, adopt and retain history |
| Foundry | P03 | Edit declared source, execute a builtin transformation, inspect output and verified cache receipts |
| Waypoint | P04 | Run Bash or recipes, inspect actual output, cancel owned jobs |
| Type garden | P05 | Edit context, generate evidence-backed type proposals and semantic diagnostics |
| Atelier | P06 | Edit source, run child programs, inspect trace frames, collect behavior scenarios and evaluate them |
| Parcel | P07 | Browse an offline registry, resolve a content-addressed lock and atomically install its bundle |
| Market day | P08 | Trade inventory, adjust offers and inspect a conservation-preserving ledger |
| Sparring grounds | P08 | Resolve simultaneous choices with exact energy/health mechanics and a round journal |
| The lantern inn | P08 | Talk with an NPC, generate replies grounded in inspectable memories and track trust |
| Spellweaver | P09 | Translate an incantation into a typed spell, resolve mana and damage, retain a grimoire |
| Fieldnotes | P10 | Edit natlang/TS/SQLite cells, resolve dependencies, inspect results and invalidate descendants |
| Commonplace | P11 | Edit/render pages, stage collaborator drafts, semantically merge them and run living cells |
| Signal room | P12 | Ingest log windows, cite evidence in incidents, resolve them and record local escalations |
| Data kitchen | P13 | Inspect imported JSON records, propose field mappings, transform and export with missing-field diagnostics |
| Atlas | P14 | Add passages, search, compose claims with exact quotations and inspect provenance |
| Relay | P15 | Execute a simulated saga, inject failures/unknown outcomes, reconcile and compensate in dependency order |
| Possibility lab | P16 | Form a hypothesis, run matched seeded inventory trials and compare recorded rewards |
| Counterexample | P17 | Edit a program/contract, generate cases, execute them and inspect actual counterexamples |
| Folio | P18 | Write/reorder sections, preview a document and export HTML/Markdown |
| Daylight | P19 | Add tasks, place them on an explicit UTC timeline, check conflicts and track completion |
| Patchwork | P20 | Propose exact edits, inspect before/after source and run syntax/behavior checks in isolated candidates |

These are working local application scopes, not a claim that every long-term
product gate in the project plans is finished. In particular, the wiki has no
network replication transport, the simulated workflow has no external provider,
the package registry contains bundled examples, the repository workbench uses
an explicit example check contract, and the scheduler has no calendar-provider
sync. The [frontend ledger](../../plans/projects/FRONTEND_STATUS.md) records
these boundaries and next development work.

## Recovery and authority

- IndexedDB stores completed event states, model/source/seed provenance,
  drafts, reduction traces and individual operation results.
- **History & traces** can inspect a reduction frame or restore a state as a
  new branch. A restoration does not undo host effects. Interrupted interactions
  expose their completed operation states so work can continue from evidence.
- Web Locks give an app one writer across tabs. A stale tab refreshes to the
  latest committed state before accepting a new interaction.
- A completed reduction persists before its view runs. **Refresh view** retries
  presentation without repeating the operation.
- Child programmes execute in dedicated Web Workers. Model turns borrow the
  shared model while the parent interpreter waits. Cancelling terminates the
  child worker; a spinning crisp cell does not freeze the editor.
- Child traces stay in host-owned IndexedDB records. Natlang requests one frame
  at a time rather than carrying a complete trace through its state.
- The companion is an explicitly **trusted local host**, not a sandbox. Bash
  has the host user's authority. A session token and origin/Host checks protect
  its HTTP boundary; they are not filesystem or process isolation.
- Native jobs have event-derived IDs, atomic receipt files and full output logs
  under `runs/studio/`. A restarted or cancelled job with uncertain effects is
  `unknown`, never silently replayed. The displayed output is a tail; logs retain
  the full stream. The Cancel button targets only that job's process group.
- Studio state restoration is not durable continuation of an arbitrary model
  stack. Native interpreter continuation support remains a separate mechanism.
  Here recovery resumes at a recorded application operation boundary.

## Development and checks

Application contracts and exact helpers are in `apps/`. Materialize their
reviewable `.nl` programmes and `.ts` type/helper files after changing a contract:

```bash
node ts-host/scripts/generate-studio-programs.mjs
node --test ts-host/test/studio-applications.test.mjs ts-host/test/studio-companion.test.mjs
NATLANG_CHROMIUM=/path/to/chrome node ts-host/scripts/studio-smoke.mjs
```

The browser smoke uses hardware-GPU-disabled Chromium, visits every application,
performs real controls, checks persistence/drafts/traces, runs notebook cells,
cancels a spinning worker and checks the mobile layout. Screenshots are saved
under `runs/studio-ui-smoke/`. Companion integration checks execute real local
SQLite, FFmpeg, Bash, build/cache, package and repository operations.

Scripted turns establish runtime and UI wiring. They do **not** establish live
model quality, convergence, training-sample admission or GPU stability. Live
teacher/student scenario evaluation remains a separate, reviewable step.

Notebook outputs also stay in host storage. A cell's typed state contains a
native-value ID and a bounded preview; dependent crisp cells receive the full
value. **Download full result** exports that value, and workspace export bundles
the current referenced values/child trace alongside the portable snapshot.
Preview sizes are presentation limits, not execution or dataset limits.
