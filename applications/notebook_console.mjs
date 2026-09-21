import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { NatlangHost, TypeScriptEnvironment, TerminalNatlangApplication, TerminalSessionStore,
  runTerminalShell } from '../ts-host/dist/index.js';
import { NotebookWorkspace } from './notebook.mjs';
import { cliFlag, modelTurnFromCli } from './natlang_cli.mjs';

const source = name => fileURLToPath(new URL(`../codebases/notebook_console/${name}`, import.meta.url));
const empty = () => ({ requests: [], runs: [], status: 'idle' });

export async function runNotebookConsole({ cells, tables = {}, workspace,
  modelTurn, sessionPath, traceDirectory, input, output, seedRoot = 17 } = {}) {
  if (!modelTurn || (!workspace && !Array.isArray(cells)))
    throw new Error('notebook console requires a workspace or cells and a modelTurn driver');
  const notebook = workspace ?? new NotebookWorkspace(cells, tables, { environment: new TypeScriptEnvironment({ mode: 'fresh' }) });
  const host = new NatlangHost({ host: { notebook, drainEvents: () => notebook.drainEvents() }, mode: 'retained' });
  const store = sessionPath ? new TerminalSessionStore(sessionPath) : null;
  const checkpoint = store?.load(empty()) ?? { revision: 0, state: empty(), seen_event_ids: [] };
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: source('reduce.nl'), view: source('view.ts') },
    initialState: checkpoint.state, initialRevision: checkpoint.revision,
    seenEventIds: checkpoint.seen_event_ids, modelTurn, traceDirectory, seedRoot,
    onCommit: commit => store?.commit(commit, app.seenEventIds) });
  try {
    await runTerminalShell(app, { input, output,
      event: (value, id) => ({ id, kind: 'request', value }) });
  } finally { host.close(); if (!workspace) notebook.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), path = cliFlag(args, '--notebook');
  if (!path) throw new Error('use --notebook FILE containing {cells,tables}');
  const config = JSON.parse(readFileSync(resolve(path), 'utf8'));
  await runNotebookConsole({ cells: config.cells, tables: config.tables ?? {},
    modelTurn: modelTurnFromCli(args),
    sessionPath: resolve(cliFlag(args, '--session', '.natlang/notebook-session.json')),
    traceDirectory: resolve(cliFlag(args, '--traces', '.natlang/notebook-traces')) });
}
