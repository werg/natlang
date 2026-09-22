import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { NatlangHost, TerminalNatlangApplication, TerminalSessionStore,
  NodeFileTree, runTerminalShell } from '../ts-host/dist/index.js';
import { EvidenceCollection } from './evidence_atlas.mjs';
import { readEvidencePath, STARTER_EVIDENCE } from './package_targets/evidence_console.mjs';
import { cliFlag, modelTurnFromCli } from './natlang_cli.mjs';

const source = name => fileURLToPath(new URL(`../codebases/evidence_console/${name}`, import.meta.url));
const empty = () => ({ questions: [], answers: [], status: 'idle' });

export async function runEvidenceConsole({ documents = STARTER_EVIDENCE, workspace = process.cwd(), modelTurn, sessionPath,
  traceDirectory, input, output, seedRoot = 17 } = {}) {
  if (!modelTurn || !Array.isArray(documents)) throw new Error('evidence console needs a modelTurn driver and a valid document collection');
  const evidence = new EvidenceCollection(documents);
  const host = new NatlangHost({ host: { evidence, drainEvents: () => evidence.drainEvents() }, mode: 'retained' });
  const store = sessionPath ? new TerminalSessionStore(sessionPath) : null;
  const checkpoint = store?.load(empty()) ?? { revision: 0, state: empty(), seen_event_ids: [] };
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: source('reduce.nl'), view: source('view.ts') },
    inputs: { files: new NodeFileTree(workspace) },
    initialState: checkpoint.state, initialRevision: checkpoint.revision,
    seenEventIds: checkpoint.seen_event_ids, modelTurn, traceDirectory, seedRoot,
    onCommit: commit => store?.commit(commit, app.seenEventIds) });
  try {
    await runTerminalShell(app, { input, output,
      event: (value, id) => ({ id, kind: 'question', value }), commands: {
        sources: { description: 'list loaded evidence sources', run: () => evidence.catalog()
          .map(row => `${row.id}  ${row.paragraphs} paragraphs  ${row.characters} characters`).join('\n') },
        load: { description: 'PATH add a file, JSON collection, or directory', run: value => {
          if (!value) throw new Error('provide a path');
          const loaded = readEvidencePath(value, workspace);
          for (const document of loaded) evidence.update(document.id, document.text);
          return `Loaded ${loaded.length} sources: ${loaded.map(row => row.id).join(', ')}`;
        } },
      } });
  } finally { host.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), path = cliFlag(args, '--documents');
  const workspace = process.cwd(), documents = path ? readEvidencePath(path, workspace) : STARTER_EVIDENCE;
  await runEvidenceConsole({ documents, modelTurn: modelTurnFromCli(args),
    workspace,
    sessionPath: resolve(cliFlag(args, '--session', '.natlang/evidence-session.json')),
    traceDirectory: resolve(cliFlag(args, '--traces', '.natlang/evidence-traces')) });
}
