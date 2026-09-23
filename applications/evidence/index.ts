/**
 * Evidence atlas: versioned local documents split into paragraph spans, token-overlap search with
 * explicit truncation, and exact citation checks. `answer` asks natlang to plan searches, pick hits
 * and compose a claim-by-claim answer; every claim's span, revision and literal quote are verified.
 * A quote proves provenance, not entailment, so the best status is `citation-checked`.
 */
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, extname, join, relative, resolve } from 'node:path';
import planSearch from './plan_search.nl';
import select from './select.nl';
import compose from './compose.nl';
import type { Draft, EvidenceAnswer, EvidenceStatus, Passage, SearchResult } from './types.js';

export type * from './types.js';
export type EvidenceDocument = { id: string, text: string };

const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const tokens = (text: string) => [...new Set(String(text).toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [])];

export class EvidenceCollection {
  readonly docs = new Map<string, { id: string, text: string, revision: string, spans: Passage[] }>();
  private readonly events: Record<string, unknown>[] = [];
  private readonly maxHits: number;

  constructor(documents: EvidenceDocument[], { maxHits = 20 } = {}) {
    this.maxHits = maxHits;
    for (const document of documents) this.update(document.id, document.text);
  }

  update(id: string, text: string): string {
    if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(id) || typeof text !== 'string') throw new Error('invalid document');
    const revision = hash(text), spans: Passage[] = [];
    let cursor = 0, index = 0;
    for (const part of text.split(/\n[ \t]*\n/)) {
      const start = text.indexOf(part, cursor), content = part.trim();
      if (content) spans.push({ id: `${id}#p${index++}`, source_id: id, revision, start, end: start + part.length, text: content });
      cursor = start + part.length;
    }
    this.docs.set(id, { id, text, revision, spans });
    this.events.push({ operation: 'evidence.update', source_id: id, revision, spans: spans.length });
    return revision;
  }

  revision(): string {
    return hash(JSON.stringify([...this.docs.values()].map(row => [row.id, row.revision]).sort()));
  }

  catalog(): { id: string, paragraphs: number, characters: number, revision: string }[] {
    return [...this.docs.values()].map(document => ({ id: document.id, paragraphs: document.spans.length,
      characters: document.text.length, revision: document.revision })).sort((left, right) => left.id.localeCompare(right.id));
  }

  search(queries: string[]): SearchResult {
    if (!Array.isArray(queries) || queries.some(query => typeof query !== 'string')) throw new Error('queries must be text');
    const words = tokens(queries.join(' '));
    const hits = [...this.docs.values()].flatMap(doc => doc.spans.map(span => {
      const body = new Set(tokens(span.text));
      const score = words.reduce((sum, word) => sum + (body.has(word) ? 1 : 0), 0);
      return { id: span.id, source_id: span.source_id, revision: span.revision, preview: span.text.slice(0, 200), score };
    })).filter(row => row.score > 0).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
    const result = { hits: hits.slice(0, this.maxHits), total: hits.length, truncated: hits.length > this.maxHits,
      collection_revision: this.revision() };
    this.events.push({ operation: 'evidence.search', queries, total: hits.length, returned: result.hits.map(hit => hit.id),
      collection_revision: result.collection_revision });
    return result;
  }

  read(ids: string[], collectionRevision: string): Passage[] {
    if (collectionRevision !== this.revision()) throw new Error('collection changed since search');
    if (!Array.isArray(ids) || new Set(ids).size !== ids.length) throw new Error('duplicate span IDs');
    const all = this.spans();
    const spans = ids.map(id => {
      const span = all.get(id);
      if (!span) throw new Error(`unknown span: ${id}`);
      return { ...span };
    });
    this.events.push({ operation: 'evidence.read', ids, collection_revision: collectionRevision });
    return spans;
  }

  verify(passages: Passage[], draft: Draft, collectionRevision: string): EvidenceAnswer {
    if (collectionRevision !== this.revision()) throw new Error('collection changed before verification');
    const byId = new Map(passages.map(span => [span.id, span])), actual = this.spans();
    const bad = draft.claims.filter(claim => {
      const span = byId.get(claim.span_id), source = actual.get(claim.span_id);
      return !span || !source || JSON.stringify(span) !== JSON.stringify(source) || span.revision !== claim.revision ||
        !claim.quote || !span.text.includes(claim.quote);
    }).map(claim => claim.span_id);
    const status: EvidenceStatus = bad.length ? 'invalid-citation' : draft.claims.length ?
      draft.gaps.length ? 'partial' : 'citation-checked' : 'unresolved';
    this.events.push({ operation: 'evidence.verify', status, claims: draft.claims.length, bad, collection_revision: collectionRevision });
    return { status, answer: draft.answer, claims: draft.claims, gaps: draft.gaps, collection_revision: collectionRevision,
      detail: bad.length ? `Invalid citation: ${bad.join(', ')}` : '' };
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }

  private spans(): Map<string, Passage> {
    return new Map([...this.docs.values()].flatMap(doc => doc.spans.map(span => [span.id, span] as const)));
  }
}

/** Search, read exact passages, compose a cited answer, and verify every citation. */
export async function answer(evidence: EvidenceCollection, question: string): Promise<EvidenceAnswer> {
  const found = evidence.search(await planSearch(question));
  const passages = evidence.read([...new Set(await select(question, found))], found.collection_revision);
  const draft = await compose(question, passages, found.truncated);
  return evidence.verify(passages, draft, found.collection_revision);
}

export const STARTER_EVIDENCE: EvidenceDocument[] = [
  { id: 'welcome', text: 'The Evidence Console answers questions from loaded sources and checks every quoted citation against the exact retained text. Use /sources to inspect the collection and /load PATH to add a text, Markdown, JSON, or directory source.' },
  { id: 'workflow', text: 'Natlang decides what to search, which spans to read, how to compose an answer, and which gaps to preserve. Crisp host operations provide exact search results and reject citations that do not match a retained span.' },
  { id: 'example', text: 'A useful first question is: How does the console prevent fabricated citations? The answer should cite the welcome or workflow source and distinguish semantic composition from exact verification.' },
];

const TEXT_EXTENSIONS = ['.json', '.md', '.txt', '.log', '.csv', '.ts', '.js', '.mjs', '.py', '.nl', '.yaml', '.yml'];

function documentId(value: string): string {
  let id = value.replace(extname(value), '').replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
  if (!/^[A-Za-z]/.test(id)) id = `doc-${id || 'source'}`;
  return id;
}

/** Read a text file, a JSON `{id,text}` collection, or a directory of them as evidence documents. */
export function readEvidencePath(value: string, workspace: string, strict = true): EvidenceDocument[] {
  const path = resolve(workspace, value), status = statSync(path);
  if (status.isDirectory()) return readdirSync(path, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))
    .flatMap(entry => entry.isSymbolicLink() ? [] : readEvidencePath(join(value, entry.name), workspace, false));
  if (!status.isFile()) return [];
  const extension = extname(path).toLowerCase();
  if (!TEXT_EXTENSIONS.includes(extension)) {
    if (!strict) return [];
    throw new Error(`unsupported evidence file: ${value}`);
  }
  const text = readFileSync(path, 'utf8');
  if (extension !== '.json') return [{ id: documentId(relative(workspace, path) || basename(path)), text }];
  let parsed: unknown;
  try { parsed = JSON.parse(text); } catch (error) { if (!strict) return []; throw error; }
  const rows = (Array.isArray(parsed) ? parsed : [parsed]) as EvidenceDocument[];
  if (rows.some(row => !row || typeof row.id !== 'string' || typeof row.text !== 'string')) {
    if (!strict) return [];
    throw new Error(`${value} must contain {id,text} or an array of them`);
  }
  return rows;
}
