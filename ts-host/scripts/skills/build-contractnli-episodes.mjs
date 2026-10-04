#!/usr/bin/env node
/** Build bounded ContractNLI research episodes from the official split JSON without importing evaluation labels. */
import { createHash } from 'node:crypto';
import { readFileSync, mkdirSync, openSync, writeFileSync, closeSync } from 'node:fs';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateEpisode } from '../../dist/skills/episode.js';

const sha = value => createHash('sha256').update(value).digest('hex');
const digestJson = value => sha(JSON.stringify(value));
const hashOrder = (seed, value) => sha(`${seed}:${value}`);
const TASK_KIND = 'contract-nli-classification';
const CHOICES = ['Entailment', 'Contradiction', 'NotMentioned'];
const CASE_LIMITS = { support: 6, query: 3 };
const MIN_PARAGRAPH_CODEPOINTS = 100;

function pointSlice(text, start, end) {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) throw new Error('invalid source span offsets');
  const points = Array.from(text);
  if (end > points.length) throw new Error('source span exceeds document text');
  return points.slice(start, end).join('');
}

function normalizeSourceParagraph(text) {
  return text.normalize('NFKC').toLowerCase().replace(/\s+/gu, ' ').trim();
}

function makeComponents(allDocs) {
  const parent = new Map(allDocs.map(row => [`${row.split}:${row.doc.id}`, `${row.split}:${row.doc.id}`]));
  const find = id => { let p = parent.get(id); while (p !== parent.get(p)) p = parent.get(p); let cursor = id; while (parent.get(cursor) !== p) { const next = parent.get(cursor); parent.set(cursor, p); cursor = next; } return p; };
  const union = (a, b) => { const pa = find(a), pb = find(b); if (pa !== pb) parent.set(pa < pb ? pb : pa, pa < pb ? pa : pb); };
  const firstByParagraph = new Map(), paragraphHashes = new Map();
  for (const row of allDocs) {
    const key = `${row.split}:${row.doc.id}`;
    const hashes = new Set();
    row.spans.forEach((text, index) => {
      const normalized = normalizeSourceParagraph(text);
      if (Array.from(normalized).length < MIN_PARAGRAPH_CODEPOINTS) return;
      const hash = sha(normalized); hashes.add(hash);
      if (firstByParagraph.has(hash)) union(key, firstByParagraph.get(hash)); else firstByParagraph.set(hash, key);
    });
    row.paragraphHashes = [...hashes].sort();
    for (const hash of hashes) paragraphHashes.set(hash, (paragraphHashes.get(hash) ?? 0) + 1);
  }
  const components = new Map();
  for (const row of allDocs) {
    const root = find(`${row.split}:${row.doc.id}`);
    components.set(root, [...(components.get(root) ?? []), row]);
  }
  const componentByDocument = new Map();
  for (const [root, members] of components) {
    const digest = sha(members.map(row => `${row.split}:${row.doc.id}`).sort().join('\n')).slice(0, 24);
    for (const row of members) componentByDocument.set(`${row.split}:${row.doc.id}`, `contract-source-component:${digest}`);
  }
  return { components, componentByDocument, repeatedParagraphs: [...paragraphHashes.values()].filter(n => n > 1).length,
    crossRoleComponents: [...components.values()].filter(members => new Set(members.map(row => row.split)).size > 1) };
}

function buildDocumentRow(doc, split, labels, rowInfo, componentId) {
  const annotationSets = doc.annotation_sets;
  if (!Array.isArray(annotationSets) || annotationSets.length === 0) return { held: 'missing_annotation_sets' };
  const hypotheses = Object.keys(labels).sort();
  const annotationMaps = annotationSets.map(set => set?.annotations);
  if (annotationMaps.some(map => !map || typeof map !== 'object')) return { held: 'invalid_annotation_set' };
  if (annotationMaps.some(map => hypotheses.some(id => !Object.hasOwn(map, id)) || Object.keys(map).some(id => !Object.hasOwn(labels, id))))
    return { held: 'annotation_hypothesis_mismatch' };
  const spans = doc.spans;
  if (!Array.isArray(spans)) return { held: 'missing_source_spans' };
  const spanText = spans.map(([start, end]) => pointSlice(doc.text, start, end));
  const expectedHypotheses = {};
  for (const id of hypotheses) {
    const annotations = annotationMaps.map(map => map[id]);
    const choices = annotations.map(annotation => annotation?.choice);
    if (choices.some(choice => !CHOICES.includes(choice)) || new Set(choices).size !== 1) return { held: 'annotation_choice_disagreement' };
    const goldChoice = choices[0];
    const evidenceAlternatives = [];
    for (const annotation of annotations) {
      if (!Array.isArray(annotation.spans)) return { held: 'invalid_annotation_evidence' };
      const ids = annotation.spans.map(index => {
        if (!Number.isInteger(index) || index < 0 || index >= spanText.length) throw new Error(`document ${doc.id} annotation ${id} refers to invalid span ${index}`);
        return String(index);
      });
      if (new Set(ids).size !== ids.length) return { held: 'duplicate_annotation_span' };
      if (goldChoice === 'NotMentioned' && ids.length) return { held: 'neutral_with_evidence' };
      if (goldChoice !== 'NotMentioned' && !ids.length) return { held: 'positive_without_evidence' };
      if (ids.length) evidenceAlternatives.push([...ids].sort((a,b)=>Number(a)-Number(b)));
    }
    const alternatives = [...new Map(evidenceAlternatives.map(row => [JSON.stringify(row), row])).values()];
    expectedHypotheses[id] = { choice: goldChoice, evidence_alternatives: alternatives };
  }
  const packet = { schema: 'natlang.contractnli-task/1', document: { id: String(doc.id), span_ids: spanText.map((_, i) => String(i)),
      span_count: spanText.length, document_type: doc.document_type },
    hypotheses: hypotheses.map(id => ({ id, description: labels[id].short_description ?? '', text: labels[id].hypothesis })),
    allowed_choices: CHOICES,
    instruction: 'For each listed hypothesis, decide whether this contract entails it, contradicts it, or does not mention it. Use research.search(query) to find relevant contract spans and research.read(spanId) to inspect exact text. Return JSON with an annotations object keyed by the exact hypothesis IDs; each value has choice and span_ids. For Entailment and Contradiction, cite sufficient exact source span IDs that support the decision. For NotMentioned, no evidence annotation is defined. Decide only about this supplied contract.' };
  const expected = { kind: TASK_KIND, scope: 'single-supplied-contract', hypotheses: expectedHypotheses,
    available_span_ids: spanText.map((_, i) => String(i)) };
  const service = contractService(spanText);
  return { item: { id: `contractnli:${split}:${doc.id}`, group: componentId, args: [JSON.stringify(packet)], expected,
      services: { research: service } }, lineage: { case_id: `contractnli:${split}:${doc.id}`, document_id: String(doc.id), split,
      component_id: componentId, file_name: doc.file_name, document_type: doc.document_type, source_url: doc.url,
      source_document_sha256: sha(doc.text), source_spans_sha256: digestJson(doc.spans), source_row_sha256: rowInfo.row_sha256,
      source_paragraph_hashes: rowInfo.paragraphHashes, source_revision: rowInfo.source_revision, license: 'CC-BY-4.0',
      dataset_citation: 'Koreeda and Manning, ContractNLI, Findings of EMNLP 2021' } };
}

function contractService(spanText) {
  if (spanText.some(text => Array.from(text).length > 8192)) throw new Error('a ContractNLI span exceeds the 8192-codepoint retrieval bound');
  const spans = spanText.map((text, index) => ({ id: String(index), text }));
  return `type ContractSearchHit = { span_id: string; preview: string };
const SOURCE_SPANS: { id: string; text: string }[] = ${JSON.stringify(spans)};
export function search(query: string): ContractSearchHit[] {
  if (typeof query !== 'string' || query.length > 512) throw new Error('query must be at most 512 characters');
  const terms = [...new Set(query.toLowerCase().split(/[^\\p{L}\\p{N}]+/u).filter(term => term.length > 2))];
  return SOURCE_SPANS.map(span => ({ span, score: terms.reduce((n, term) => n + (span.text.toLowerCase().includes(term) ? 1 : 0), 0) }))
    .filter(row => row.score > 0).sort((a, b) => b.score - a.score || Number(a.span.id) - Number(b.span.id)).slice(0, 8)
    .map(({ span }) => ({ span_id: span.id, preview: Array.from(span.text).slice(0, 220).join('') }));
}
export function read(spanId: string): string {
  if (typeof spanId !== 'string' || !/^(0|[1-9][0-9]*)$/u.test(spanId)) throw new Error('invalid span ID');
  const span = SOURCE_SPANS.find(row => row.id === spanId);
  if (!span) throw new Error('unknown span ID');
  return JSON.stringify({ span_id: span.id, text: span.text });
}`;
}

function queryComponents(groups, targetFractions) {
  const totals = Object.fromEntries(CHOICES.map(choice => [choice, 0]));
  for (const rows of groups.values()) for (const row of rows) for (const outcome of Object.values(row.item.expected.hypotheses)) totals[outcome.choice]++;
  const targets = Object.fromEntries(CHOICES.map(choice => [choice, Math.round(totals[choice] * targetFractions)]));
  const counts = Object.fromEntries(CHOICES.map(choice => [choice, 0]));
  const selected = new Set();
  while (CHOICES.some(choice => counts[choice] < targets[choice])) {
    const candidates = [...groups].filter(([id]) => !selected.has(id)).map(([id, docs]) => {
      const additions = Object.fromEntries(CHOICES.map(choice => [choice, docs.reduce((n, row) => n + Object.values(row.item.expected.hypotheses).filter(x => x.choice === choice).length, 0)]));
      const gain = CHOICES.reduce((n, choice) => n + Math.min(additions[choice], Math.max(0, targets[choice] - counts[choice])), 0);
      return { id, additions, gain };
    }).sort((a,b)=>b.gain-a.gain || hashOrder('contractnli-query-v1',a.id).localeCompare(hashOrder('contractnli-query-v1',b.id)));
    if (!candidates.length) break;
    const best = candidates[0]; selected.add(best.id);
    for (const choice of CHOICES) counts[choice] += best.additions[choice];
  }
  return { selected, totals, targets, counts };
}

function groupAndValidateSplit(split, data, sourceRevision) {
  if (!data || !Array.isArray(data.documents) || !data.labels || typeof data.labels !== 'object') throw new Error(`invalid ContractNLI ${split} dataset`);
  const prepared = data.documents.map(doc => {
    if (!doc || typeof doc.text !== 'string') throw new Error(`invalid ${split} contract row`);
    const rowInfo = { row_sha256: split === 'train' ? digestJson(doc) : digestJson({ id: doc.id, file_name: doc.file_name,
      document_type: doc.document_type, url: doc.url, text: doc.text, spans: doc.spans }), source_revision: sourceRevision };
    let previousEnd = 0;
    const spans = doc.spans?.map(([start, end]) => {
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < previousEnd) throw new Error(`overlapping/unsorted ${split} span offsets in document ${doc.id}`);
      previousEnd = end;
      return pointSlice(doc.text, start, end);
    });
    if (doc.id === undefined || !spans || typeof doc.file_name !== 'string') throw new Error(`invalid ${split} contract row`);
    return { split, doc, spans, rowInfo };
  });
  if (new Set(prepared.map(row => String(row.doc.id))).size !== prepared.length) throw new Error(`duplicate ContractNLI IDs in ${split}`);
  return prepared;
}

export function buildContractNliEpisodes({ train, dev, test, inputHashes, sourceRevision, archiveSha256, sourceManifestSha256 }) {
  const preparedBySplit = { train: groupAndValidateSplit('train', train, sourceRevision), dev: groupAndValidateSplit('dev', dev, sourceRevision),
    test: groupAndValidateSplit('test', test, sourceRevision) };
  const hypothesisLabels = train.labels;
  if (Object.keys(hypothesisLabels).length !== 17 || Object.values(hypothesisLabels).some(x => !x || typeof x.hypothesis !== 'string'))
    throw new Error('expected the official 17 fixed hypotheses');
  for (const [split, source] of [['dev',dev],['test',test]]) if (JSON.stringify(Object.keys(source.labels ?? {}).sort()) !== JSON.stringify(Object.keys(hypothesisLabels).sort()))
    throw new Error(`hypothesis IDs differ in ${split}`);
  const all = [...preparedBySplit.train, ...preparedBySplit.dev, ...preparedBySplit.test];
  const closure = makeComponents(all);
  const componentFor = row => closure.componentByDocument.get(`${row.split}:${row.doc.id}`);
  const byComponent = new Map();
  const held = [];
  for (const row of preparedBySplit.train) {
    const group = componentFor(row);
    const built = buildDocumentRow(row.doc, row.split, hypothesisLabels, row.rowInfo, group);
    if (built.held) held.push({ split: 'train', document_id: String(row.doc.id), reason: built.held, source_row_sha256: row.rowInfo.row_sha256, component_id: group });
    else byComponent.set(group, [...(byComponent.get(group) ?? []), { ...built, labels: Object.values(built.item.expected.hypotheses) }]);
  }
  const protectedComponents = new Set(closure.crossRoleComponents.map(members => closure.componentByDocument.get(`${members[0].split}:${members[0].doc.id}`)));
  for (const [group, docs] of [...byComponent]) if (protectedComponents.has(group)) {
    held.push(...docs.map(row => ({ split: 'train', document_id: row.lineage.document_id, reason: 'shared_source_paragraph_with_original_dev_or_test',
      source_row_sha256: row.lineage.source_row_sha256, component_id: group })));
    byComponent.delete(group);
  }
  if (byComponent.size < 3) throw new Error(`insufficient train-only source groups after protected split/paragraph closure: ${byComponent.size}`);
  const partition = queryComponents(byComponent, 0.2);
  if (partition.selected.size < 2 || partition.selected.size >= byComponent.size) throw new Error('component-based train/query split is too small');
  const supportRows = [], queryRows = [];
  for (const [group, docs] of byComponent) (partition.selected.has(group) ? queryRows : supportRows).push(...docs);
  const supportGroups = new Set(supportRows.map(row => row.item.group)), queryGroups = new Set(queryRows.map(row => row.item.group));
  if ([...supportGroups].some(group => queryGroups.has(group)) || supportGroups.size < 2 || queryGroups.size < 2) throw new Error('contract/source paragraph group leakage');
  const roundRobin = (items, groupOf, seed) => {
    const queues = new Map();
    for (const item of items) queues.set(groupOf(item), [...(queues.get(groupOf(item)) ?? []), item]);
    const keys = [...queues.keys()].sort((a,b)=>hashOrder(seed,a).localeCompare(hashOrder(seed,b)));
    for (const queue of queues.values()) queue.sort((a,b)=>hashOrder(`${seed}:item`,a.item.id).localeCompare(hashOrder(`${seed}:item`,b.item.id)));
    const out = []; let left = items.length;
    while (left) for (const key of keys) { const item=queues.get(key).shift(); if(item){out.push(item);left--;} }
    return out;
  };
  const supportOrder = roundRobin(supportRows, row => row.item.group, 'contractnli-support-order-v1');
  const queryOrder = roundRobin(queryRows, row => row.item.group, 'contractnli-query-order-v1');
  const supportShards=[]; for(let i=0;i<supportOrder.length;i+=CASE_LIMITS.support) supportShards.push(supportOrder.slice(i,i+CASE_LIMITS.support));
  const queryShards=Array.from({length:supportShards.length},()=>[]);
  queryOrder.forEach((row,i)=>queryShards[Math.floor(i*supportShards.length/queryOrder.length)].push(row));
  const target = { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'natlang.contractnli/1', id: 'contractnli-train-v1' },
    files: { 'solve.nl': '---\nargs: { packet: string }\nreturns: string\n---\nReview the supplied non-disclosure agreement against each of its 17 hypotheses. Search for relevant clauses, read the exact span text, and classify each hypothesis as Entailment, Contradiction, or NotMentioned. Return JSON only: {"annotations":{"nda-1":{"choice":"Entailment|Contradiction|NotMentioned","span_ids":["..."]}}}. Use the exact hypothesis IDs. For Entailment or Contradiction, cite sufficient exact source span IDs. NotMentioned has no evidence annotation. Decisions concern this supplied contract only.\n' } };
  const episodes = supportShards.map((support, index) => {
    const query = queryShards[index];
    if (!query?.length || support.length > CASE_LIMITS.support || query.length > CASE_LIMITS.query) throw new Error(`episode ${index} exceeds bounded cases`);
    const sg=new Set(support.map(row=>row.item.group)), qg=new Set(query.map(row=>row.item.group));
    if(sg.size<2 || [...qg].some(group=>sg.has(group))) throw new Error(`episode ${index} leaks source paragraph group`);
    return {version:'natlang.skill-episode/1',id:`contractnli-train-v1-${String(index+1).padStart(3,'0')}`,
      family:'research:contractnli-clause-review',split:'train',source_groups:[...new Set([...sg,...qg])].sort(),
      license:'ContractNLI dataset: CC-BY-4.0; contract text as distributed within this dataset; third-party source URL preserved per document.',target,
      library:{kind:'empty',skills:{}},support:{cases:support.map(row=>row.item)},query:{cases:query.map(row=>row.item)},
      operations:['create','revise','test'],limits:{maxSteps:6},provenance:{generator:'natlang.contractnli-skill-episodes/1',
        metric:{schema:'natlang.skill-contractnli/1',kind:TASK_KIND},source_manifest_sha256:sourceManifestSha256,
        archive_sha256:archiveSha256,source_revision:sourceRevision,input_hashes:inputHashes,
        citation:'Koreeda and Manning, ContractNLI, Findings of EMNLP 2021',
        schema_reference:'https://github.com/stanfordnlp/contract-nli/blob/gh-pages/index.md',license_reference:'https://github.com/stanfordnlp/contract-nli/blob/gh-pages/LICENSE',
        support_components:[...sg].sort(),query_components:[...qg].sort(),label_semantics:'Contract-specific NLI; NotMentioned applies only to the supplied document.',episode_index:index,
        admission:'candidate skill episodes only; not admitted or training-ready'}};
  });
  const issues=episodes.flatMap(episode=>validateEpisode(episode)); if(issues.length) throw new Error(`episode validation failed: ${JSON.stringify(issues.slice(0,5))}`);
  const protectedEvaluation={schema:'natlang.contractnli-protected-splits/1',original_splits:{
    dev:{documents:dev.documents.map(doc=>({document_id:String(doc.id),source_document_sha256:sha(doc.text),component_id:componentFor(preparedBySplit.dev.find(row=>String(row.doc.id)===String(doc.id)))})),input_sha256:inputHashes.dev},
    test:{documents:test.documents.map(doc=>({document_id:String(doc.id),source_document_sha256:sha(doc.text),component_id:componentFor(preparedBySplit.test.find(row=>String(row.doc.id)===String(doc.id)))})),input_sha256:inputHashes.test}},
    evaluation_rows_imported_into_episodes:0,original_role_preserved:true};
  const lineage=[...supportRows,...queryRows].map(row=>row.lineage);
  return {episodes,lineage,held,protectedEvaluation,audit:{train_documents:train.documents.length,dev_documents:dev.documents.length,test_documents:test.documents.length,
    hypotheses:Object.keys(hypothesisLabels).length,train_cases_input:preparedBySplit.train.length,train_cases_held:held.length,
    train_cases_emitted:supportRows.length+queryRows.length,episodes:episodes.length,support_documents:supportRows.length,query_documents:queryRows.length,
    support_groups:supportGroups.size,query_groups:queryGroups.size,min_support_groups_per_episode:Math.min(...episodes.map(ep=>new Set(ep.support.cases.map(x=>x.group)).size)),
    support_cases_max:Math.max(...episodes.map(ep=>ep.support.cases.length)),query_cases_max:Math.max(...episodes.map(ep=>ep.query.cases.length)),
    repeated_long_source_paragraphs:closure.repeatedParagraphs,cross_original_role_components:closure.crossRoleComponents.length,
    model_calls:0,provider_calls:0,admitted_training_rows:0}};
}

function main(argv) {
  const arg = name => { const i=argv.indexOf(name);return i<0?null:argv[i+1]; };
  const sourceDir=resolve(arg('--source-dir')??'vendor/datasets/contract-nli-20261004/data');
  const acquisitionPath=resolve(arg('--acquisition')??'vendor/datasets/contract-nli-20261004/acquisition.json');
  const out=resolve(arg('--out')??'runs/self-improvement-expansion-20261004/contractnli-episodes-v1');
  const acquisitionText=readFileSync(acquisitionPath,'utf8'), acquisition=JSON.parse(acquisitionText);
  const archivePath=resolve(arg('--archive')??join(dirname(acquisitionPath),'contract-nli.zip'));
  const archiveSha=sha(readFileSync(archivePath));
  if(archiveSha!==acquisition.archive_sha256) throw new Error('pinned ContractNLI archive SHA-256 mismatch');
  const names=['train','dev','test']; const inputHashes={}; const splits={};
  for(const split of names){const path=join(sourceDir,`${split}.json`),bytes=readFileSync(path);inputHashes[split]=sha(bytes);splits[split]=JSON.parse(bytes.toString('utf8'));}
  const built=buildContractNliEpisodes({ ...splits,inputHashes,sourceRevision:acquisition.source_revision,
    archiveSha256:archiveSha,sourceManifestSha256:sha(acquisitionText) });
  mkdirSync(out,{recursive:false});
  const outputs={ 'episodes.jsonl':built.episodes.map(x=>JSON.stringify(x)).join('\n')+'\n',
    'lineage.jsonl':built.lineage.map(x=>JSON.stringify(x)).join('\n')+'\n',
    'held.jsonl':built.held.map(x=>JSON.stringify(x)).join('\n')+(built.held.length?'\n':''),
    'protected-evaluation.json':JSON.stringify(built.protectedEvaluation,null,2)+'\n',
    'audit.json':JSON.stringify({schema:'natlang.contractnli-skill-episodes/1',...built.audit},null,2)+'\n' };
  for(const [name,text] of Object.entries(outputs)){const fd=openSync(join(out,name),'wx');try{writeFileSync(fd,text);}finally{closeSync(fd);}}
  const manifest={schema:'natlang.contractnli-skill-episode-build/1',source_revision:acquisition.source_revision,
    archive_sha256:acquisition.archive_sha256,source_manifest_sha256:sha(acquisitionText),input_hashes:inputHashes,
    outputs:Object.fromEntries(Object.entries(outputs).map(([name,text])=>[name,{sha256:sha(text),bytes:Buffer.byteLength(text)}])),
    model_calls:0,provider_calls:0,admitted_training_rows:0,audit:built.audit};
  const fd=openSync(join(out,'manifest.json'),'wx');try{writeFileSync(fd,JSON.stringify(manifest,null,2)+'\n');}finally{closeSync(fd);}
  process.stdout.write(JSON.stringify(manifest)+'\n');
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main(process.argv.slice(2));
