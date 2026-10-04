#!/usr/bin/env node
/** Build leakage-safe skill-authoring cases from the prepared SciFact train-only candidate packet. */
import { createHash } from 'node:crypto';
import { readFileSync, mkdirSync, openSync, writeFileSync, closeSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateEpisode } from '../../dist/skills/episode.js';

const sha = value => createHash('sha256').update(value).digest('hex');
const byHash = (seed, id) => sha(`${seed}:${id}`);
const componentGroup = value => String(value).startsWith('scifact:component:') ? String(value) : `scifact:component:${value}`;

function serviceSource(documents) {
  // Only retrieved source content is embedded. Labels/rationales are never passed into this serializer.
  const privateDocs = documents.map(({ doc_id, title, abstract_sentences }) => ({
    id: String(doc_id), title, sentences: abstract_sentences.map(({ sentence_id, text }) => ({ id: String(sentence_id), text })),
  }));
  return `type ResearchHit = { id: string; title: string; kind: 'abstract' };
type ResearchSentence = { id: string; text: string };
const DOCUMENTS: { id: string; title: string; sentences: ResearchSentence[] }[] = ${JSON.stringify(privateDocs)};
export function search(query: string): ResearchHit[] {
  if (typeof query !== 'string' || query.length > 256) throw new Error('query must be a string of at most 256 characters');
  const terms = [...new Set(query.toLowerCase().split(/[^\\p{L}\\p{N}]+/u).filter(x => x.length > 2))];
  return DOCUMENTS.map(doc => ({ doc, score: terms.reduce((n, term) => n + ((doc.title + ' ' + doc.sentences.map(s => s.text).join(' ')).toLowerCase().includes(term) ? 1 : 0), 0) }))
    .filter(row => row.score > 0).sort((a, b) => b.score - a.score || a.doc.id.localeCompare(b.doc.id)).slice(0, 5)
    .map(({ doc }) => ({ id: doc.id, title: doc.title, kind: 'abstract' }));
}
export function read(sourceId: string): string {
  if (typeof sourceId !== 'string' || sourceId.length > 64) throw new Error('invalid source ID');
  const doc = DOCUMENTS.find(row => row.id === sourceId);
  if (!doc) throw new Error('unknown source ID');
  return JSON.stringify({ id: doc.id, title: doc.title, sentences: doc.sentences });
}`;
}

function parseJsonl(text, label) {
  return text.split(/\r?\n/u).filter(Boolean).map((line, i) => {
    try { return JSON.parse(line); } catch (error) { throw new Error(`${label} line ${i + 1}: ${error}`); }
  });
}

function queryComponentIds(components, rows, fraction = 0.2) {
  const totals = Object.fromEntries(['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'].map(label => [label, rows.filter(r => r.host_only_oracle.label === label).length]));
  const targets = Object.fromEntries(Object.entries(totals).map(([label, n]) => [label, Math.max(1, Math.round(n * fraction))]));
  const counts = Object.fromEntries(Object.keys(totals).map(label => [label, 0]));
  const remaining = new Map([...components].map(([id, members]) => [id, Object.fromEntries(Object.keys(totals).map(label => [label, members.filter(r => r.host_only_oracle.label === label).length]))]));
  const chosen = new Set();
  while (Object.keys(targets).some(label => counts[label] < targets[label])) {
    const candidates = [...remaining].filter(([id]) => !chosen.has(id));
    if (!candidates.length) break;
    candidates.sort(([a, ca], [b, cb]) => {
      const gain = c => Object.keys(targets).reduce((sum, label) => sum + Math.min(c[label], Math.max(0, targets[label] - counts[label])), 0);
      return gain(cb) - gain(ca) || byHash('scifact-query-v1', a).localeCompare(byHash('scifact-query-v1', b));
    });
    const [id, additions] = candidates[0];
    chosen.add(id);
    for (const label of Object.keys(counts)) counts[label] += additions[label];
  }
  return { chosen, totals, targets, counts };
}

function verifyCandidate(row, index) {
  const bad = message => { throw new Error(`candidate ${index} (${row?.candidate_id ?? 'unknown'}): ${message}`); };
  if (row?.schema !== 'natlang.scifact-research-candidate/1' || row.collection_status !== 'candidate_not_admitted' || row.role !== 'train') bad('wrong schema/status/role');
  if (typeof row.candidate_id !== 'string' || !row.task || typeof row.task.claim !== 'string' || !row.task.claim.trim() || !Array.isArray(row.task.documents) || !row.task.documents.length) bad('invalid task');
  const oracle = row.host_only_oracle;
  if (!oracle || !['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'].includes(oracle.label) || !Array.isArray(oracle.accepted_evidence_sets)) bad('invalid host oracle');
  const docs = new Map();
  for (const doc of row.task.documents) {
    const id = String(doc.doc_id);
    if (docs.has(id) || !Array.isArray(doc.abstract_sentences) || !doc.abstract_sentences.length) bad('duplicate/empty cited abstract');
    const sentences = new Map();
    for (const s of doc.abstract_sentences) {
      const sid = String(s.sentence_id);
      if (sentences.has(sid) || typeof s.text !== 'string') bad('duplicate or invalid sentence');
      sentences.set(sid, s.text);
    }
    docs.set(id, sentences);
  }
  if (oracle.label === 'NOT_ENOUGH_INFO' && oracle.accepted_evidence_sets.length) bad('NOT_ENOUGH_INFO has annotated positive evidence');
  if (oracle.label !== 'NOT_ENOUGH_INFO' && !oracle.accepted_evidence_sets.length) bad('classified candidate lacks rationale');
  const seen = new Set();
  for (const rationale of oracle.accepted_evidence_sets) {
    const doc = docs.get(String(rationale.doc_id));
    if (!doc || rationale.label !== oracle.label || !Array.isArray(rationale.sentence_ids) || !rationale.sentence_ids.length) bad('rationale label/doc/sentences do not match');
    for (const id of rationale.sentence_ids) if (!doc.has(String(id))) bad('rationale refers to missing sentence');
    const key = JSON.stringify([String(rationale.doc_id), [...new Set(rationale.sentence_ids.map(String))].sort()]);
    if (seen.has(key)) bad('duplicate rationale');
    seen.add(key);
  }
  if (!Array.isArray(row.source_groups) || !row.source_groups.some(x => x.startsWith('scifact:component:')) ||
      row.source_groups.filter(x => x.startsWith('scifact:component:')).length !== 1) bad('must have exactly one connected-component group');
  if (!row.provenance || !row.provenance.claim_license || !row.provenance.abstract_license || !row.provenance.source_line_sha256) bad('missing source/license provenance');
  return row.source_groups.find(x => x.startsWith('scifact:component:'));
}

export function buildScifactEpisode({ candidateText, sourceManifest, candidateSha256, sourceManifestSha256 }) {
  const candidates = parseJsonl(candidateText, 'candidate input');
  if (!sourceManifest || sourceManifest.schema !== 'natlang.scifact-research-preparation/1') throw new Error('invalid SciFact source manifest');
  if (sha(candidateText) !== candidateSha256) throw new Error('candidate input hash mismatch');
  if (sourceManifest.outputs?.['candidates.jsonl']?.sha256 !== candidateSha256 || sourceManifest.counts?.candidate_rows !== candidates.length)
    throw new Error('candidate bytes/count disagree with source manifest');
  const rows = candidates;
  const components = new Map();
  const candidateIds = new Set();
  rows.forEach((row, i) => {
    const component = verifyCandidate(row, i);
    if (candidateIds.has(row.candidate_id)) throw new Error(`duplicate candidate id ${row.candidate_id}`);
    candidateIds.add(row.candidate_id);
    components.set(component, [...(components.get(component) ?? []), row]);
  });
  const partition = queryComponentIds(components, rows);
  if (partition.chosen.size < 2 || partition.chosen.size >= components.size) throw new Error('component partition is too small');
  const supportRows = rows.filter(row => !partition.chosen.has(componentGroup(row.provenance.component_id)));
  // Candidate component_id stores the digest without its public group prefix.
  const queryRows = rows.filter(row => partition.chosen.has(componentGroup(row.provenance.component_id)));
  if (!supportRows.length || !queryRows.length) throw new Error('empty support or query role');
  const supportGroups = new Set(supportRows.map(row => componentGroup(row.provenance.component_id)));
  const queryGroups = new Set(queryRows.map(row => componentGroup(row.provenance.component_id)));
  if ([...supportGroups].some(group => queryGroups.has(group)) || supportGroups.size < 2 || queryGroups.size < 2) throw new Error('component leakage/insufficient groups');
  const caseRow = row => {
    const documents = row.task.documents;
    const docsById = new Map(documents.map(doc => [String(doc.doc_id), doc]));
    const available = documents.map(doc => ({ doc_id: String(doc.doc_id), sentence_ids: doc.abstract_sentences.map(s => String(s.sentence_id)) }));
    const accepted = row.host_only_oracle.accepted_evidence_sets.map(r => [{ doc_id: String(r.doc_id), sentence_ids: r.sentence_ids.map(String) }]);
    const packet = { instruction: row.task.instruction, claim: row.task.claim,
      scope: 'Decide only from these supplied cited abstracts; NOT_ENOUGH_INFO means the supplied abstracts do not provide sufficient evidence for either classification.',
      allowedLabels: ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'],
      catalog: documents.map(doc => ({ id: String(doc.doc_id), title: doc.title, kind: 'abstract' })) };
    const expected = { kind: 'scifact-claim-evidence', label: row.host_only_oracle.label,
      accepted_evidence_sets: accepted, available_documents: available, scope: 'supplied-cited-documents-only' };
    const docs = [...docsById.values()];
    return { item: { id: row.candidate_id, group: componentGroup(row.provenance.component_id), args: [JSON.stringify(packet)], expected,
      services: { research: serviceSource(docs) } }, lineage: { case_id: row.candidate_id, component_id: row.provenance.component_id,
      source_groups: row.source_groups, provenance: row.provenance } };
  };
  const supportEntries = supportRows.map(caseRow), queryEntries = queryRows.map(caseRow);
  const queues = new Map();
  for (const entry of supportEntries) queues.set(entry.lineage.component_id, [...(queues.get(entry.lineage.component_id) ?? []), entry]);
  const componentOrder = [...queues.keys()].sort((a, b) => byHash('scifact-support-order-v2', a).localeCompare(byHash('scifact-support-order-v2', b)));
  for (const queue of queues.values()) queue.sort((a, b) => byHash('scifact-support-row-v2', a.item.id).localeCompare(byHash('scifact-support-row-v2', b.item.id)));
  const interleaved = [];
  let left = supportEntries.length;
  while (left) for (const component of componentOrder) {
    const row = queues.get(component).shift();
    if (row) { interleaved.push(row); left--; }
  }
  const supportShards = [];
  for (let i = 0; i < interleaved.length; i += 6) {
    const shard = interleaved.slice(i, i + 6);
    if (new Set(shard.map(x => x.lineage.component_id)).size < 2) throw new Error(`support shard ${supportShards.length} needs at least two source components`);
    supportShards.push(shard);
  }
  const queryOrder = [...queryEntries].sort((a, b) => byHash('scifact-query-order-v2', a.item.id).localeCompare(byHash('scifact-query-order-v2', b.item.id)));
  const queryShards = Array.from({ length: supportShards.length }, () => []);
  queryOrder.forEach((entry, i) => queryShards[Math.floor(i * supportShards.length / queryOrder.length)].push(entry));
  const targetV2 = { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'natlang.skill-research/1', id: 'scifact-claims-train-v1' },
    files: { 'solve.nl': '---\nargs: { packet: string }\nreturns: string\n---\nYou receive a JSON packet containing one scientific claim, an allowedLabels list, and a catalog of cited abstract IDs and titles. Use research.search(query) to find relevant supplied abstracts and research.read(sourceId) to inspect them. Decide only from the listed abstracts. Return JSON only: {"label":"SUPPORT|CONTRADICT|NOT_ENOUGH_INFO","citations":[{"doc_id":"...","sentence_ids":["..."]}]}. Cite sufficient sentence IDs for SUPPORT or CONTRADICT. For NOT_ENOUGH_INFO, return an empty citations array; this means the supplied abstracts do not provide sufficient evidence, not that no evidence exists elsewhere. Do not infer evidence from a title alone.\n' } };
  const episodesV2 = supportShards.map((supportShard, index) => {
    const queryShard = queryShards[index];
    if (!queryShard?.length || queryShard.length > 3 || supportShard.length > 6) throw new Error(`episode ${index} violates its case budget`);
    const localSupportGroups = new Set(supportShard.map(x => x.item.group));
    const localQueryGroups = new Set(queryShard.map(x => x.item.group));
    if (localSupportGroups.size < 2 || [...localQueryGroups].some(group => localSupportGroups.has(group))) throw new Error(`episode ${index} violates group separation`);
    return { version: 'natlang.skill-episode/1', id: `scifact-cited-evidence-train-v2-${String(index + 1).padStart(3, '0')}`,
      family: 'research:scifact-cited-evidence', split: 'train', source_groups: [...new Set([...localSupportGroups, ...localQueryGroups])].sort(),
      license: 'Claims and evidence annotations: CC-BY-4.0; abstracts: ODC-By-1.0', target: targetV2,
      library: { kind: 'empty', skills: {} }, support: { cases: supportShard.map(x => x.item) }, query: { cases: queryShard.map(x => x.item) },
      operations: ['create', 'revise', 'test'], limits: { maxSteps: 6 },
      provenance: { generator: 'natlang.scifact-skill-episodes/2', metric: { schema: 'natlang.skill-scifact/1', kind: 'scifact-claim-evidence' },
        source_manifest_sha256: sourceManifestSha256, source_candidates_sha256: candidateSha256, source_revision: sourceManifest.source_revision,
        source_archive_sha256: sourceManifest.source_archive_sha256, citation: sourceManifest.dataset_citation,
        schema_reference: sourceManifest.schema_reference, license_reference: sourceManifest.license_reference,
        query_holdout_components: [...localQueryGroups].sort(), support_components: [...localSupportGroups].sort(), episode_index: index,
        label_semantics: 'Exact SciFact labels; NOT_ENOUGH_INFO is scoped to insufficient evidence in the supplied cited abstracts.',
        admission: 'candidate skill episodes only; not admitted or training-ready' } };
  });
  const issuesV2 = episodesV2.flatMap(episode => validateEpisode(episode));
  if (issuesV2.length) throw new Error(`episode validation failed: ${JSON.stringify(issuesV2.slice(0, 5))}`);
  return { episodes: episodesV2, lineage: [...supportEntries, ...queryEntries].map(x => x.lineage), audit: { input_rows: rows.length, components: components.size,
    episodes: episodesV2.length, support_rows: supportEntries.length, query_rows: queryEntries.length, support_components: supportGroups.size, query_components: queryGroups.size,
    support_cases_max: Math.max(...episodesV2.map(e => e.support.cases.length)), query_cases_max: Math.max(...episodesV2.map(e => e.query.cases.length)),
    support_groups_min_per_episode: Math.min(...episodesV2.map(e => new Set(e.support.cases.map(c => c.group)).size)),
    query_appearances: episodesV2.reduce((n, e) => n + e.query.cases.length, 0), query_unique_cases: queryRows.length,
    query_reuse: episodesV2.reduce((n, e) => n + e.query.cases.length, 0) - queryRows.length,
    query_label_counts: Object.fromEntries(['SUPPORT','CONTRADICT','NOT_ENOUGH_INFO'].map(label => [label, queryRows.filter(row => row.host_only_oracle.label === label).length])),
    support_label_counts: Object.fromEntries(['SUPPORT','CONTRADICT','NOT_ENOUGH_INFO'].map(label => [label, supportRows.filter(row => row.host_only_oracle.label === label).length])),
    planned_query_target_counts: partition.targets, selected_query_component_counts: partition.counts,
    model_calls: 0, provider_calls: 0, admitted_training_rows: 0 } };


}

function main(argv) {
  const get = name => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : null; };
  const inputPath = get('--candidates'), manifestPath = get('--source-manifest'), outPath = get('--out');
  if (!inputPath || !manifestPath || !outPath) throw new Error('Usage: build-scifact-episodes.mjs --candidates FILE --source-manifest FILE --out NEW_DIR');
  const candidateText = readFileSync(inputPath, 'utf8'), manifestText = readFileSync(manifestPath, 'utf8');
  const sourceManifest = JSON.parse(manifestText), out = resolve(outPath);
  const built = buildScifactEpisode({ candidateText, sourceManifest, candidateSha256: sha(candidateText), sourceManifestSha256: sha(manifestText) });
  mkdirSync(out, { recursive: false });
  const episodeText = built.episodes.map(episode => JSON.stringify(episode)).join('\n') + '\n';
  const lineageText = built.lineage.map(row => JSON.stringify(row)).join('\n') + '\n';
  const auditText = JSON.stringify({ schema: 'natlang.scifact-skill-episodes/2', ...built.audit }, null, 2) + '\n';
  const outputs = { 'episodes.jsonl': episodeText, 'lineage.jsonl': lineageText, 'audit.json': auditText };
  for (const [name, text] of Object.entries(outputs)) {
    const fd = openSync(join(out, name), 'wx'); try { writeFileSync(fd, text); } finally { closeSync(fd); }
  }
  const manifest = { schema: 'natlang.scifact-skill-episode-build/2', episodes: built.episodes.length, candidate_rows: built.audit.input_rows,
    support_rows: built.audit.support_rows, query_rows: built.audit.query_rows, model_calls: 0, provider_calls: 0,
    admitted_training_rows: 0, source_candidates_sha256: sha(candidateText), source_manifest_sha256: sha(manifestText),
    outputs: Object.fromEntries(Object.entries(outputs).map(([name, text]) => [name, { sha256: sha(text), bytes: Buffer.byteLength(text) }])) };
  const fd = openSync(join(out, 'manifest.json'), 'wx'); try { writeFileSync(fd, JSON.stringify(manifest, null, 2) + '\n'); } finally { closeSync(fd); }
  process.stdout.write(JSON.stringify({ ...manifest, audit: built.audit }) + '\n');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2));
export { serviceSource };
