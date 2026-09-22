import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { NatlangHost, TypeScriptEnvironment, TerminalNatlangApplication, TerminalSessionStore,
  NodeFileTree, runTerminalShell } from '../ts-host/dist/index.js';
import { NotebookWorkspace } from './notebook.mjs';
import { STARTER_NOTEBOOK } from './package_targets/notebook_console.mjs';
import { cliFlag, modelTurnFromCli } from './natlang_cli.mjs';

const source = name => fileURLToPath(new URL(`../codebases/notebook_console/${name}`, import.meta.url));
const empty = () => ({ requests: [], runs: [], status: 'idle' });

export async function runNotebookConsole({ cells = STARTER_NOTEBOOK.cells, tables = STARTER_NOTEBOOK.tables, workspace,
  modelTurn, sessionPath, traceDirectory, input, output, seedRoot = 17,
  root = process.cwd() } = {}) {
  if (!modelTurn || (!workspace && !Array.isArray(cells)))
    throw new Error('notebook console needs a modelTurn driver and a valid workspace or cell collection');
  const notebook = workspace ?? new NotebookWorkspace(cells, tables, { environment: new TypeScriptEnvironment({ mode: 'fresh' }) });
  const host = new NatlangHost({ host: { notebook, drainEvents: () => notebook.drainEvents() }, mode: 'retained' });
  const store = sessionPath ? new TerminalSessionStore(sessionPath) : null;
  const checkpoint = store?.load(empty()) ?? { revision: 0, state: empty(), seen_event_ids: [] };
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: source('reduce.nl'), view: source('view.ts') },
    reducerInputs: () => ({ files: new NodeFileTree(root) }),
    initialState: checkpoint.state, initialRevision: checkpoint.revision,
    seenEventIds: checkpoint.seen_event_ids, modelTurn, traceDirectory, seedRoot,
    onCommit: commit => store?.commit(commit, app.seenEventIds) });
  try {
    await runTerminalShell(app, { input, output,
      event: (value, id) => ({ id, kind: 'request', value }), commands: {
        cells: { description: 'list notebook cells and dependencies', run: () => notebook.catalog()
          .map(cell => `${cell.id} [${cell.engine}] needs: ${cell.needs.join(', ') || 'none'} ${cell.description}`).join('\n') },
        load: { description: 'FILE import or replace cells and tables', run: value => {
          if (!value) throw new Error('provide a notebook JSON path');
          const imported = notebook.importConfig(JSON.parse(readFileSync(resolve(value), 'utf8')));
          return `Loaded ${imported.cells.length} cells and ${imported.tables.length} tables.`;
        } },
      } });
  } finally { host.close(); if (!workspace) notebook.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), path = cliFlag(args, '--notebook');
  const config = path ? JSON.parse(readFileSync(resolve(path), 'utf8')) : STARTER_NOTEBOOK;
  await runNotebookConsole({ cells: config.cells, tables: config.tables ?? {},
    modelTurn: modelTurnFromCli(args),
    root: resolve(cliFlag(args, '--root', '.')),
    sessionPath: resolve(cliFlag(args, '--session', '.natlang/notebook-session.json')),
    traceDirectory: resolve(cliFlag(args, '--traces', '.natlang/notebook-traces')) });
}
