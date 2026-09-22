import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { NatlangHost, TerminalNatlangApplication, TerminalSessionStore,
  NodeFileTree, renderTerminalView } from '../ts-host/dist/index.js';
import { LogWorkspace } from './log_investigator.mjs';
import { cliFlag, modelTurnFromCli } from './natlang_cli.mjs';

const source = name => fileURLToPath(new URL(`../codebases/log_investigator/${name}`, import.meta.url));
export const emptyIncidentState = () => ({ cursor: -1, observed: 0, alerts: [], unknowns: [], status: 'idle' });

/** Consume a real async log source through the shared terminal lifecycle. */
export async function runLogConsole({ events, modelTurn, sessionPath, traceDirectory,
  logs = new LogWorkspace(), output = process.stdout, seedRoot = 17 } = {}) {
  if (!events || !modelTurn) throw new Error('log console requires events and modelTurn');
  const host = new NatlangHost({ host: { logs, drainEvents: () => logs.drainEvents() }, mode: 'retained' });
  const store = sessionPath ? new TerminalSessionStore(sessionPath) : null;
  const checkpoint = store?.load(emptyIncidentState()) ?? { revision: 0,
    state: emptyIncidentState(), seen_event_ids: [] };
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: source('reduce.nl'), view: source('view.ts') },
    inputs: { files: new NodeFileTree(process.cwd()) },
    initialState: checkpoint.state, initialRevision: checkpoint.revision,
    seenEventIds: checkpoint.seen_event_ids, modelTurn, seedRoot, traceDirectory,
    onCommit: commit => store?.commit(commit, app.seenEventIds) });
  try {
    output.write(renderTerminalView((await app.start()).view, { color: Boolean(output.isTTY), width: output.columns }));
    await app.consume(events, transition => output.write(renderTerminalView(transition.view,
      { color: Boolean(output.isTTY), width: output.columns })));
    return { state: app.state, revision: app.revision };
  } finally { await app.close(); host.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  async function* jsonLines() {
    for await (const line of createInterface({ input: process.stdin })) {
      if (line.trim()) yield JSON.parse(line);
    }
  }
  await runLogConsole({ events: jsonLines(), modelTurn: modelTurnFromCli(args),
    sessionPath: resolve(cliFlag(args, '--session', '.natlang/log-session.json')),
    traceDirectory: resolve(cliFlag(args, '--traces', '.natlang/log-traces')) });
}
