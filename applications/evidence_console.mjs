import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { NatlangHost, TerminalNatlangApplication, TerminalSessionStore,
  runTerminalShell } from '../ts-host/dist/index.js';
import { EvidenceCollection } from './evidence_atlas.mjs';
import { cliFlag, modelTurnFromCli } from './natlang_cli.mjs';

const source = name => fileURLToPath(new URL(`../codebases/evidence_console/${name}`, import.meta.url));
const empty = () => ({ questions: [], answers: [], status: 'idle' });

export async function runEvidenceConsole({ documents, modelTurn, sessionPath,
  traceDirectory, input, output, seedRoot = 17 } = {}) {
  if (!modelTurn || !Array.isArray(documents)) throw new Error('evidence console requires documents and modelTurn');
  const evidence = new EvidenceCollection(documents);
  const host = new NatlangHost({ host: { evidence, drainEvents: () => evidence.drainEvents() }, mode: 'retained' });
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
      event: (value, id) => ({ id, kind: 'question', value }) });
  } finally { host.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), path = cliFlag(args, '--documents');
  if (!path) throw new Error('use --documents FILE containing [{id,text}, ...]');
  const documents = JSON.parse(readFileSync(resolve(path), 'utf8'));
  await runEvidenceConsole({ documents, modelTurn: modelTurnFromCli(args),
    sessionPath: resolve(cliFlag(args, '--session', '.natlang/evidence-session.json')),
    traceDirectory: resolve(cliFlag(args, '--traces', '.natlang/evidence-traces')) });
}
