/** Versioned local document spans and exact citation checks. */
import { createHash } from 'node:crypto';

const hash = text => createHash('sha256').update(text).digest('hex');
const tokens = text => [...new Set(String(text).toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [])];

export class EvidenceCollection {
  constructor(documents, { maxHits = 20 } = {}) {
    this.docs = new Map(); this.events = []; this.maxHits = maxHits;
    for (const document of documents) this.update(document.id, document.text);
  }

  update(id, text) {
    if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(id) || typeof text !== 'string')
      throw new Error('invalid document');
    const revision = hash(text), spans = [];
    let cursor = 0, index = 0;
    for (const part of text.split(/\n[ \t]*\n/)) {
      const start = text.indexOf(part, cursor), content = part.trim();
      if (content) spans.push({ id: `${id}#p${index++}`, source_id: id,
        revision, start, end: start + part.length, text: content });
      cursor = start + part.length;
    }
    this.docs.set(id, { id, text, revision, spans });
    this.events.push({ operation: 'evidence.update', source_id: id, revision,
      spans: spans.length });
    return revision;
  }

  revision() {
    return hash(JSON.stringify([...this.docs.values()].map(row => [row.id, row.revision]).sort()));
  }

  catalog() {
    return [...this.docs.values()].map(document => ({ id: document.id,
      paragraphs: document.spans.length, characters: document.text.length,
      revision: document.revision })).sort((left, right) => left.id.localeCompare(right.id));
  }

  search(queries) {
    if (!Array.isArray(queries) || queries.some(query => typeof query !== 'string'))
      throw new Error('queries must be text');
    const words = tokens(queries.join(' '));
    const hits = [...this.docs.values()].flatMap(doc => doc.spans.map(span => {
      const body = new Set(tokens(span.text));
      const score = words.reduce((sum, word) => sum + (body.has(word) ? 1 : 0), 0);
      return { id: span.id, source_id: span.source_id, revision: span.revision,
        preview: span.text.slice(0, 200), score };
    })).filter(row => row.score > 0)
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
    const result = { hits: hits.slice(0, this.maxHits), total: hits.length,
      truncated: hits.length > this.maxHits, collection_revision: this.revision() };
    this.events.push({ operation: 'evidence.search', queries, total: hits.length,
      returned: result.hits.map(hit => hit.id), collection_revision: result.collection_revision });
    return result;
  }

  read(ids, collectionRevision) {
    if (collectionRevision !== this.revision()) throw new Error('collection changed since search');
    if (!Array.isArray(ids) || new Set(ids).size !== ids.length) throw new Error('duplicate span IDs');
    const all = new Map([...this.docs.values()].flatMap(doc => doc.spans.map(span => [span.id, span])));
    const spans = ids.map(id => {
      const span = all.get(id);
      if (!span) throw new Error(`unknown span: ${id}`);
      return { ...span };
    });
    this.events.push({ operation: 'evidence.read', ids, collection_revision: collectionRevision });
    return spans;
  }

  verify(passages, draft, collectionRevision) {
    if (collectionRevision !== this.revision()) throw new Error('collection changed before verification');
    const byId = new Map(passages.map(span => [span.id, span]));
    const actual = new Map([...this.docs.values()].flatMap(doc => doc.spans.map(span => [span.id, span])));
    const bad = [];
    for (const claim of draft.claims) {
      const span = byId.get(claim.span_id);
      const source = actual.get(claim.span_id);
      if (!span || !source || JSON.stringify(span) !== JSON.stringify(source) ||
          span.revision !== claim.revision || !claim.quote ||
          !span.text.includes(claim.quote)) bad.push(claim.span_id);
    }
    const status = bad.length ? 'invalid-citation' : draft.claims.length ?
      draft.gaps.length ? 'partial' : 'citation-checked' : 'unresolved';
    this.events.push({ operation: 'evidence.verify', status, claims: draft.claims.length,
      bad, collection_revision: collectionRevision });
    return { status, answer: draft.answer, claims: draft.claims, gaps: draft.gaps,
      collection_revision: collectionRevision, detail: bad.length ? `Invalid citation: ${bad.join(', ')}` : '' };
  }

  drainEvents() { return this.events.splice(0); }
}
