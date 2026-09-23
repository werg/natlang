// KQA Pro adapter: multi-hop questions over a Wikidata-derived knowledge base. Each question's gold program
// is translated into TypeScript over a small paged `kb` API and run against the full knowledge base; a
// question is used only when that run reproduces the dataset's answer. The model gets a module holding the
// part of the knowledge base the run touched, plus decoy attributes and relations, and never the program.
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { fileURLToPath } from 'node:url';
import { curriculumCase, evalCall, literal, returnCall } from './lib.mjs';
import { SOURCES, cachePath } from './acquire.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
const PAGE = 8, MAX_SET = 200, DECOY_ATTRIBUTES = 8, DECOY_RELATIONS = 8;
let state;

function load() {
  if (state) return state;
  const source = SOURCES.kqapro;
  const read = path => {
    try { return JSON.parse(readFileSync(cachePath(CACHE, 'kqapro', source.revision, path), 'utf8')); }
    catch { throw new Error('KQA Pro is not in the dataset cache; run node scripts/inline-curriculum/acquire.mjs --source kqapro'); }
  };
  const kb = read('kb.json');
  const rows = [...read('train.json').map((row, i) => ({ ...row, id: `train:${i}`, split: 'train' })),
    ...read('val.json').map((row, i) => ({ ...row, id: `val:${i}`, split: 'test' }))];
  // Concept ancestry, by name, for FilterConcept.
  const ancestors = new Map();
  const conceptNames = id => {
    if (ancestors.has(id)) return ancestors.get(id);
    ancestors.set(id, []);
    const concept = kb.concepts[id];
    const names = concept ? [concept.name, ...(concept.instanceOf ?? []).flatMap(conceptNames)] : [];
    ancestors.set(id, [...new Set(names)]);
    return ancestors.get(id);
  };
  const byName = new Map();
  for (const [id, entity] of Object.entries(kb.entities)) byName.set(entity.name, [...(byName.get(entity.name) ?? []), id]);
  state = { kb, rows, conceptNames, byName };
  return state;
}

const number = value => String(value);
function formatValue(value) {
  if (value.type === 'quantity') return value.unit === '1' ? number(value.value) : `${number(value.value)} ${value.unit}`;
  return String(value.value);
}
const qualifiers = raw => Object.fromEntries(Object.entries(raw ?? {}).map(([key, values]) => [key, values.map(formatValue)]));

/** The `kb` API over an entity table; `touch` records every entity the code looks at. */
function api(entities, conceptNames, byName, touch = () => {}) {
  const view = id => {
    const entity = entities[id];
    if (!entity) throw new Error(`no entity ${id}`);
    touch(id);
    return entity;
  };
  const relations = id => (view(id).relations ?? []).filter(r => entities[r.object] || r.objectName);
  return {
    find: name => (byName.get(name) ?? []).filter(id => entities[id]),
    entity: id => {
      const entity = view(id);
      return { id, name: entity.name, concepts: entity.concepts ?? (entity.instanceOf ?? []).flatMap(conceptNames),
        attributes: (entity.attributes ?? []).map(a => ({ key: a.key, value: formatValue(a.value), ...(a.value.type === 'quantity' ? { number: a.value.value, unit: a.value.unit } : {}),
          ...(a.value.type === 'year' || a.value.type === 'date' ? { year: Number(String(a.value.value).slice(0, 4)) } : {}), qualifiers: qualifiers(a.qualifiers) })) };
    },
    relation_pages: id => Math.max(1, Math.ceil(relations(id).length / PAGE)),
    relations: (id, n) => relations(id).slice((n - 1) * PAGE, n * PAGE).map(r => ({ predicate: r.predicate, direction: r.direction,
      other: r.object, other_name: entities[r.object]?.name ?? r.objectName, qualifiers: qualifiers(r.qualifiers) })),
  };
}

const COMPARE = { '=': '===', '!=': '!==', '<': '<', '>': '>' };
/** The program as TypeScript over `kb`; undefined when it uses a function this adapter does not support. */
function translate(program) {
  const lines = ['const relationsOf = (id: string) => { const out: { predicate: string, direction: string, other: string }[] = []; const pages = kb.relation_pages(id); for (let n = 1; n <= pages; n++) out.push(...kb.relations(id, n)); return out; };'];
  const at = i => `s${i}`;
  for (const [i, step] of program.entries()) {
    const [a, b] = step.dependencies.map(at), [x, y, z] = step.inputs.map(v => JSON.stringify(v));
    const attrs = `kb.entity(id).attributes.filter(attr => attr.key === ${x})`;
    let code;
    switch (step.function) {
      case 'Find': code = `kb.find(${x})`; break;
      case 'FilterConcept': code = `${a}.filter(id => kb.entity(id).concepts.includes(${x}))`; break;
      case 'Relate': code = `[...new Set(${a}.flatMap(id => relationsOf(id).filter(r => r.predicate === ${x} && r.direction === ${y}).map(r => r.other)))]`; break;
      case 'And': code = `${a}.filter(id => ${b}.includes(id))`; break;
      case 'Or': code = `[...new Set([...${a}, ...${b}])]`; break;
      case 'FilterStr': code = `${a}.filter(id => ${attrs}.some(attr => attr.value === ${y}))`; break;
      case 'FilterNum': {
        const [amount, unit] = step.inputs[1].split(' ');
        if (!COMPARE[step.inputs[2]]) return;
        code = `${a}.filter(id => ${attrs}.some(attr => attr.number !== undefined && attr.unit === ${JSON.stringify(unit ?? '1')} && attr.number ${COMPARE[step.inputs[2]]} ${Number(amount)}))`;
        break;
      }
      case 'FilterYear':
        if (!COMPARE[step.inputs[2]]) return;
        code = `${a}.filter(id => ${attrs}.some(attr => attr.year !== undefined && attr.year ${COMPARE[step.inputs[2]]} ${Number(step.inputs[1])}))`;
        break;
      case 'Count': code = `String(${a}.length)`; break;
      case 'What': case 'QueryName': code = `kb.entity(${a}[0]).name`; break;
      case 'QueryAttr': code = `kb.entity(${a}[0]).attributes.filter(attr => attr.key === ${x}).map(attr => attr.value)`; break;
      case 'VerifyStr': code = `${a}.includes(${x}) ? "yes" : "no"`; break;
      case 'QueryAttrQualifier':
        code = `kb.entity(${a}[0]).attributes.find(attr => attr.key === ${x} && attr.value === ${y})!.qualifiers[${z}][0]`; break;
      case 'QueryRelation': code = `relationsOf(${a}[0]).find(r => r.other === ${b}[0])!.predicate`; break;
      case 'QueryRelationQualifier':
        code = `(relationsOf(${a}[0]) as { predicate: string, other: string, qualifiers: Record<string, string[]> }[]).find(r => r.other === ${b}[0] && r.predicate === ${x})!.qualifiers[${y}][0]`; break;
      case 'SelectBetween': {
        const value = side => `kb.entity(${side}[0]).attributes.find(attr => attr.key === ${x} && attr.number !== undefined)!.number!`;
        code = `(${value(a)} ${step.inputs[1] === 'greater' ? '>' : '<'} ${value(b)} ? kb.entity(${a}[0]) : kb.entity(${b}[0])).name`;
        break;
      }
      default: return;
    }
    lines.push(`const ${at(i)} = ${code};`);
  }
  const last = at(program.length - 1);
  lines.push(`return Array.isArray(${last}) ? ${last}[0] : ${last};`);
  return lines.join('\n');
}

/** Run translated code, transpiled to JavaScript, against an API. */
function run(code, kb) {
  const js = ts.transpileModule(`function question(kb: any) {\n${code}\n}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  return new Function('kb', `${js}\nreturn question(kb);`)(kb);
}

/** The knowledge-base slice a question needs, with capped decoys; undefined when the question is unsuitable. */
function slice(row) {
  const { kb, conceptNames, byName } = load();
  const code = translate(row.program);
  if (!code) return;
  const touched = new Set();
  let answer;
  try { answer = run(code, api(kb.entities, conceptNames, byName, id => touched.add(id))); } catch { return; }
  if (answer === undefined || String(answer) !== row.answer || touched.size > MAX_SET) return;
  const keys = new Set(row.program.flatMap(step => step.inputs));
  const entities = {};
  for (const id of touched) {
    const entity = kb.entities[id];
    const attributes = entity.attributes.filter(a => keys.has(a.key));
    const decoyAttributes = entity.attributes.filter(a => !keys.has(a.key)).slice(0, DECOY_ATTRIBUTES);
    const needed = entity.relations.filter(r => keys.has(r.predicate) || touched.has(r.object));
    const decoys = entity.relations.filter(r => !needed.includes(r) && kb.entities[r.object]).slice(0, DECOY_RELATIONS);
    entities[id] = { name: entity.name, concepts: (entity.instanceOf ?? []).flatMap(conceptNames), attributes: [...attributes, ...decoyAttributes],
      relations: [...needed, ...decoys].map(r => ({ ...r, objectName: kb.entities[r.object]?.name })) };
  }
  // Stubs for related entities the question does not need, so following a decoy relation still works.
  for (const entity of Object.values(entities)) for (const r of entity.relations) if (!entities[r.object] && kb.entities[r.object])
    entities[r.object] = { name: kb.entities[r.object].name, concepts: (kb.entities[r.object].instanceOf ?? []).flatMap(conceptNames), attributes: [], relations: [] };
  // The slice must answer the question on its own.
  const names = new Map();
  for (const [id, entity] of Object.entries(entities)) names.set(entity.name, [...(names.get(entity.name) ?? []), id]);
  let local;
  try { local = run(code, api(entities, () => [], names)); } catch { return; }
  if (String(local) !== row.answer) return;
  return { code, entities };
}

/** KQA Pro: answer a multi-hop question from a paged knowledge-base module. */
export function kqaQuestion(seed, index) {
  const { rows } = load();
  // Try questions spread over the dataset from this index until one is supported.
  for (let offset = 0; offset < 200; offset++) {
    const row = rows[(index * 7919 + offset * 104729) % rows.length];
    const found = slice(row);
    if (!found) continue;
    const first = row.program.find(step => step.function === 'Find')?.inputs[0] ?? 'none';
    const module = `type Attribute = { key: string, value: string, number?: number, unit?: string, year?: number, qualifiers: Record<string, string[]> };
type Relation = { predicate: string, direction: "forward" | "backward", other: string, other_name: string, qualifiers: Record<string, string[]> };
const ENTITIES: Record<string, { name: string, concepts: string[], attributes: any[], relations: any[] }> = ${literal(found.entities)};
const format = (v: any) => v.type === "quantity" ? (v.unit === "1" ? String(v.value) : v.value + " " + v.unit) : String(v.value);
const qualifiers = (raw: any) => Object.fromEntries(Object.entries(raw ?? {}).map(([k, vs]: [string, any]) => [k, vs.map(format)]));
/** The ids of the entities named name. */
export function find(name: string): string[] { return Object.keys(ENTITIES).filter(id => ENTITIES[id].name === name); }
/** An entity: its name, the concepts it is an instance of (with their broader concepts), and its attributes. */
export function entity(id: string): { id: string, name: string, concepts: string[], attributes: Attribute[] } {
  const e = ENTITIES[id];
  if (!e) throw new Error("no entity " + id);
  return { id, name: e.name, concepts: e.concepts, attributes: e.attributes.map((a: any) => ({ key: a.key, value: format(a.value),
    ...(a.value.type === "quantity" ? { number: a.value.value, unit: a.value.unit } : {}),
    ...(a.value.type === "year" || a.value.type === "date" ? { year: Number(String(a.value.value).slice(0, 4)) } : {}), qualifiers: qualifiers(a.qualifiers) })) };
}
/** The number of pages of id's relations (${PAGE} per page). */
export function relation_pages(id: string): number { return Math.max(1, Math.ceil((ENTITIES[id]?.relations.length ?? 0) / ${PAGE})); }
/**
 * Page n (1-based) of id's relations. direction "forward" means id predicate other ("X country Y": Y is X's
 * country); "backward" means other predicate id.
 */
export function relations(id: string, n: number): Relation[] {
  return (ENTITIES[id]?.relations ?? []).slice((n - 1) * ${PAGE}, n * ${PAGE}).map((r: any) => ({ predicate: r.predicate, direction: r.direction,
    other: r.object, other_name: r.objectName, qualifiers: qualifiers(r.qualifiers) }));
}
`;
    return [curriculumCase({ family: 'kqapro_question', shape: row.id.replace(':', '_'), variant: 'q',
      splitGroup: `kqapro:${first}`, split: row.split, slice: 'nested_scoped', domain: 'relational', mode: 'single_call', inline: 'avoid',
      worldSemantics: 'closed_world',
      evidence: { world: [`answer: ${row.answer}`], retrieved: [],
        background: [`source: KQA Pro ${SOURCES.kqapro.revision} ${row.id} (CC BY-SA 4.0)`, `program: ${row.program.map(s => `${s.function}(${s.inputs.join(', ')})`).join(' -> ')}`] },
      minimumSequence: ['find the named entities', 'follow relations and filter by concept and attribute', 'read off the answer'],
      reference: { root: [evalCall(found.code), returnCall(row.answer)] },
      root: { name: 'answer_question', args: { question: 'string' }, returns: 'string',
        instructions: 'Answer question from the knowledge base kb, and only from it. Give the answer exactly as kb states it: an entity\'s name, an attribute value as kb.entity shows it, a count as digits, a relation\'s predicate, or "yes" or "no".' },
      files: { 'answer_question/kb.ts': module },
      inputs: { question: row.question }, expected: row.answer })];
  }
  throw new Error(`KQA Pro: no supported question near index ${index}`);
}
