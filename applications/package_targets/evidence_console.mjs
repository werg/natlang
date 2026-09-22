import { readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, extname, join, relative, resolve } from 'node:path';
import { EvidenceCollection } from '../evidence_atlas.mjs';
import { flag, terminalExecutable } from './terminal_helpers.mjs';

export const STARTER_EVIDENCE = [
  { id: 'welcome', text: 'The Evidence Console answers questions from loaded sources and checks every quoted citation against the exact retained text. Use /sources to inspect the collection and /load PATH to add a text, Markdown, JSON, or directory source.' },
  { id: 'workflow', text: 'Natlang decides what to search, which spans to read, how to compose an answer, and which gaps to preserve. Crisp host operations provide exact search results and reject citations that do not match a retained span.' },
  { id: 'example', text: 'A useful first question is: How does the console prevent fabricated citations? The answer should cite the welcome or workflow source and distinguish semantic composition from exact verification.' },
];

function documentId(value) {
  let id = value.replace(extname(value), '').replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
  if (!/^[A-Za-z]/.test(id)) id = `doc-${id || 'source'}`;
  return id;
}

export function readEvidencePath(value, workspace, strict = true) {
  const path = resolve(workspace, value), status = statSync(path);
  if (status.isDirectory()) return readdirSync(path, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))
    .flatMap(entry => entry.isSymbolicLink() ? [] : readEvidencePath(join(value, entry.name), workspace, false));
  if (!status.isFile()) return [];
  const extension = extname(path).toLowerCase();
  if (!['.json', '.md', '.txt', '.log', '.csv', '.ts', '.js', '.mjs', '.py', '.nl', '.yaml', '.yml'].includes(extension)) {
    if (!strict) return [];
    throw new Error(`unsupported evidence file: ${value}`);
  }
  const text = readFileSync(path, 'utf8');
  if (extension === '.json') {
    let parsed;
    try { parsed = JSON.parse(text); }
    catch (error) { if (!strict) return []; throw error; }
    const rows = Array.isArray(parsed) ? parsed : [parsed];
    if (rows.some(row => !row || typeof row.id !== 'string' || typeof row.text !== 'string')) {
      if (!strict) return [];
      throw new Error(`${value} must contain {id,text} or an array of them`);
    }
    return rows;
  }
  return [{ id: documentId(relative(workspace, path) || basename(path)), text }];
}

export function createTarget(context) {
  const path = flag(context.args, '--documents');
  const documents = path ? readEvidencePath(path, context.workspace) : STARTER_EVIDENCE;
  const evidence = new EvidenceCollection(documents);
  return terminalExecutable(context, { hostObject: { evidence, drainEvents: () => evidence.drainEvents() },
    initialState: () => ({ questions: [], answers: [], status: 'idle' }),
    event: (value, id) => ({ id, kind: 'question', value }),
    commands: {
      sources: { description: 'list loaded evidence sources', run: () => evidence.catalog()
        .map(row => `${row.id}  ${row.paragraphs} paragraphs  ${row.characters} characters`).join('\n') },
      load: { description: 'PATH add a file, JSON collection, or directory', run: value => {
        if (!value) throw new Error('provide a path');
        const loaded = readEvidencePath(value, context.workspace);
        for (const document of loaded) evidence.update(document.id, document.text);
        return `Loaded ${loaded.length} source${loaded.length === 1 ? '' : 's'}: ${loaded.map(row => row.id).join(', ')}`;
      } },
      example: { description: 'show a useful first question', run: () =>
        'How does the console prevent fabricated citations?' },
    } });
}
