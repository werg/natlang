#!/usr/bin/env node
/** Materialize independently authored semantic-stack programs without inventing gold. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { nlFile, curriculumCase, evalCall, returnCall } from './lib.mjs';
import { verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const root = resolve(import.meta.dirname, '../../..');
const catalogs = [
  join(root, 'plans/semantic_stack/astra-programs.json'),
  join(root, 'training/semantic_stack/sol_programs.json'),
];
const args = process.argv.slice(2);
const flags = args.slice(2);
const pilotAstra = flags.includes('--pilot-astra');
const stageSpecsMode = flags.includes('--stage-specs');
const groupedAstra = flags.includes('--grouped-astra') || stageSpecsMode;
const reviewAt = flags.indexOf('--review-ids');
const reviewPath = reviewAt >= 0 && flags[reviewAt + 1] ? resolve(flags[reviewAt + 1]) : null;
const variantAt = flags.indexOf('--stage-variant');
const variantPath = variantAt >= 0 && flags[variantAt + 1] ? resolve(flags[variantAt + 1]) : null;
const knownFlags = new Set(['--pilot-astra', '--grouped-astra', '--stage-specs', '--review-ids', '--stage-variant']);
const selected = flags.filter(flag => knownFlags.has(flag));
const expectedFlagCount = selected.length + (reviewPath ? 1 : 0) + (variantPath ? 1 : 0);
if (args[0] !== '--out' || !args[1] || flags.length !== expectedFlagCount ||
    (pilotAstra && (stageSpecsMode || groupedAstra || reviewPath)) ||
    (stageSpecsMode && flags.includes('--grouped-astra')) ||
    (variantPath && !stageSpecsMode) ||
    selected.length !== new Set(selected).size)
  throw Error('usage: node build-semantic-stack-programs.mjs --out FRESH_DIRECTORY [--pilot-astra | --grouped-astra | --stage-specs] [--review-ids IDS_JSON] [--stage-variant VARIANT_JSON]');
const output = resolve(args[1]);
const variantBytes = variantPath ? await readFile(variantPath) : null;
const variant = variantBytes ? JSON.parse(variantBytes) : null;
if (variant && (variant.schema !== 'semantic-stage-variant/1' || typeof variant.program_id !== 'string' ||
    typeof variant.base_catalog_sha256 !== 'string' || typeof variant.base_stage !== 'string'))
  throw Error('invalid stage variant metadata');
const reviewList = reviewPath ? JSON.parse(await readFile(reviewPath, 'utf8')) : [];
if (!Array.isArray(reviewList) || reviewList.some(id => typeof id !== 'string') ||
    new Set(reviewList).size !== reviewList.length)
  throw Error('--review-ids must be a JSON array of distinct program IDs');
const reviewIds = new Set(reviewList);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const safeId = id => typeof id === 'string' && /^[a-z][a-z0-9_-]{4,100}$/.test(id);
const shortText = (value, label, id, minimum = 12) => {
  if (typeof value !== 'string' || value.trim().length < minimum)
    throw Error(`${id}: missing substantive ${label}`);
  return value.trim();
};
const paragraphs = value => Array.isArray(value) ? value.filter(Boolean).map(String) :
  typeof value === 'string' ? [value] : [];
function sourceFixture(entry) {
  const fixture = entry.source_fixture;
  if (!fixture || typeof fixture !== 'object') return null;
  const files = fixture.files ?? fixture.folder_files;
  if (!files || typeof files !== 'object' || Array.isArray(files)) return null;
  const out = {};
  for (const [path, content] of Object.entries(files)) {
    if (!/^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(path) || path.startsWith('.') || path.includes('..'))
      throw Error(`${entry.id}: unsafe fixture path ${path}`);
    if (typeof content !== 'string' || !content.trim()) throw Error(`${entry.id}: empty fixture ${path}`);
    out[path] = content;
  }
  if (Array.isArray(fixture.records)) for (const record of fixture.records) {
    if (typeof record?.path !== 'string' || out[record.path] !== record.text)
      throw Error(`${entry.id}: source_fixture.records differs from files map`);
  }
  return Object.keys(out).length ? out : null;
}
function recordIdentityBindings(entry, fixture) {
  if (entry.typed_inputs) return null;
  const records = entry.source_fixture?.records;
  if (!Array.isArray(records) || records.length !== Object.keys(fixture).length)
    throw Error(`${entry.id}: every Astra source file needs an explicit typed record identity`);
  const ids = new Set(), paths = new Set();
  const bindings = records.map(record => {
    const { id, path, text } = record;
    if (typeof id !== 'string' || !/^[A-Za-z][A-Za-z0-9_-]*$/.test(id) || ids.has(id) || paths.has(path) ||
        fixture[path] !== text) throw Error(`${entry.id}: invalid or duplicate record ID/path binding`);
    ids.add(id); paths.add(path);
    return { id, path };
  });
  return bindings.sort((a, b) => a.path.localeCompare(b.path));
}
function exactExpected(entry) {
  const oracle = entry.oracle;
  if (oracle && typeof oracle === 'object' && Object.hasOwn(oracle, 'expected')) return oracle.expected;
  if (Object.hasOwn(entry, 'expected')) return entry.expected;
  return undefined;
}
function comparisonSpec(entry, fixture, expected) {
  const spec = entry.oracle?.comparison_spec;
  if (!spec) return null;
  if (spec.schema_version !== 'semantic-structured-comparison/1' ||
      !Array.isArray(spec.unordered_paths) || !Array.isArray(spec.prose_paths) ||
      !Array.isArray(spec.ordered_paths)) throw Error(`${entry.id}: invalid comparison spec`);
  const paths = [...spec.unordered_paths, ...spec.ordered_paths, ...spec.prose_paths];
  if (new Set(paths).size !== paths.length)
    throw Error(`${entry.id}: comparison paths have conflicting or duplicate declarations`);
  const parse = pointer => {
    if (typeof pointer !== 'string' || !pointer.startsWith('/') || /~(?![01])/.test(pointer))
      throw Error(`${entry.id}: invalid comparison pointer ${pointer}`);
    return pointer.slice(1).split('/').map(part => part.replace(/~1/g, '/').replace(/~0/g, '~'));
  };
  const valuesAt = (value, path) => path.length === 0 ? [value] :
    path[0] === '*' ? (value && typeof value === 'object' ?
      Object.values(value).flatMap(child => valuesAt(child, path.slice(1))) : []) :
      value && typeof value === 'object' && Object.hasOwn(value, path[0]) ?
        valuesAt(value[path[0]], path.slice(1)) : [];
  for (const [kind, pointers, expectedType] of [
    ['unordered', spec.unordered_paths, 'array'], ['ordered', spec.ordered_paths, 'array'],
    ['prose', spec.prose_paths, 'string'],
  ]) for (const pointer of pointers) {
    const parts = parse(pointer);
    const selected = valuesAt(expected, parts);
    const wildcardIndex = parts.indexOf('*');
    const emptyDeclaredCollection = wildcardIndex >= 0 && !selected.length &&
      valuesAt(expected, parts.slice(0, wildcardIndex)).some(value => value && typeof value === 'object' &&
        Object.keys(value).length === 0);
    if ((!selected.length && !emptyDeclaredCollection) ||
        selected.some(value => expectedType === 'array' ? !Array.isArray(value) : typeof value !== 'string'))
      throw Error(`${entry.id}: ${kind} comparison path ${pointer} does not select ${expectedType} gold`);
  }
  if (spec.source_id_domain?.kind === 'source_paths') {
    const values = spec.source_id_domain.values;
    if (!Array.isArray(values) || new Set(values).size !== values.length ||
        values.some(path => typeof path !== 'string' || !Object.hasOwn(fixture, path)))
      throw Error(`${entry.id}: comparison source ID domain is not bound to visible files`);
  }
  return spec;
}
function returnType(value) {
  if (typeof value === 'string') return 'string';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number' && Number.isFinite(value)) return 'number';
  if (Array.isArray(value) && value.every(item => typeof item === 'string')) return 'string[]';
  if (Array.isArray(value) && value.length) {
    const members = value.map(returnType);
    if (members.every(type => type && type === members[0])) return `${members[0]}[]`;
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const fields = Object.entries(value).map(([name, field]) => [name, returnType(field)]);
    if (fields.length && fields.every(([name, type]) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && type))
      return `{ ${fields.map(([name, type]) => `${name}: ${type}`).join('; ')} }`;
  }
  return null; // A catalog must declare a typed output for compound results.
}
function stages(entry) {
  const functions = Array.isArray(entry.functions) ? entry.functions : [];
  if (functions.length) return functions.map((fn, index) => {
    const name = String(fn.name ?? `stage_${index + 1}`).replace(/[^A-Za-z0-9_]/g, '_');
    const instructions = paragraphs(fn.instructions ?? fn.numbered_instructions ?? fn.purpose).join('\n');
    return { name, instructions, signature: fn.signature };
  }).filter(fn => fn.instructions);
  const chain = Array.isArray(entry.call_chain) ? entry.call_chain.slice(1) : [];
  return (entry.steps ?? []).map((step, index) => ({
    name: String(chain[index] ?? `stage_${index + 1}`).replace(/[^A-Za-z0-9_]/g, '_'),
    instructions: String(step).replace(/^\s*\d+\.\s*/, ''),
  }));
}
function stageBody(step) {
  // Authored catalog steps describe a parent's action: "Call inspect to ...".
  // Repeating that line inside inspect asks the child to invoke itself, which
  // the language correctly rejects. Render the child's own job as an action.
  const escaped = step.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const selfCall = new RegExp(`^Call ${escaped} to (.+)$`, 'i').exec(step.instructions);
  if (selfCall) return selfCall[1][0].toUpperCase() + selfCall[1].slice(1);
  const forEach = new RegExp(`^Call ${escaped} for (.+)$`, 'i').exec(step.instructions);
  if (forEach) return 'For ' + forEach[1];
  const within = new RegExp(`^Within ${escaped},\\s*`, 'i');
  return step.instructions.replace(within, '');
}
function runtimeStageArgs(step, entry, grouped) {
  if (!entry.typed_inputs || !step.signature) return { item: grouped ? 'EvidenceBatch' : 'EvidenceRecord', context: 'EvidenceContext' };
  const signature = /^[^(]+\(([^)]*)\)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)$/.exec(step.signature);
  if (!signature) throw Error(`${entry.id}: invalid named stage signature ${step.signature}`);
  const args = Object.fromEntries(signature[1].split(',').map(part => {
    const match = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(part);
    if (!match) throw Error(`${entry.id}: invalid stage argument ${part}`);
    return [match[1], match[2]];
  }));
  return args;
}
function runtimeTypeFile(entry, grouped) {
  if (!entry.typed_inputs) return `export type EvidenceRecord = { id: string; path: string; text: string };\n` +
    `export type EvidenceFinding = { sourcePath: string; finding: string; evidenceIds: string[]; unresolved: string[] };\n` +
    (grouped ? `export type EvidenceBatch = { folderPath: string; records: EvidenceRecord[] };\n` +
      `export type EvidenceBatchFinding = { folderPath: string; findings: EvidenceFinding[] };\n` : '') +
    `export type EvidenceContext = { objective: string; rule: string; relatedRecords: EvidenceRecord[]; priorFindings: EvidenceFinding[]; stageFindings: EvidenceFinding[] };\n`;
  const declarations = Object.entries(entry.typed_inputs).map(([role, declaration]) => {
    const match = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\{[\s\S]+\})\s*$/.exec(declaration);
    if (!match) throw Error(`${entry.id}: invalid authored ${role} type declaration`);
    return { role, name: match[1], body: match[2] };
  });
  if (new Set(declarations.map(item => item.name)).size !== declarations.length)
    throw Error(`${entry.id}: duplicate authored type alias`);
  for (const role of ['bundle', 'inquiry', 'stage_state', 'record', 'evidence_unit', 'relation_decision', 'work_item'])
    if (!declarations.some(item => item.role === role)) throw Error(`${entry.id}: missing authored ${role} type`);
  // A source fixture is an immutable path-to-text snapshot. DomainContext is
  // the explicit task/rule/source scope available to the authored stages.
  // Every other structure comes from the catalog itself, especially passage
  // witnesses and finite-frontier candidate IDs; never replace it with unknown.
  return `export type DirectorySnapshot = Record<string, string>;\n` +
    `export type DomainContext = { objective: string; semanticRule: string; sourcePaths: string[] };\n` +
    declarations.map(({ name, body }) => `export type ${name} = ${body};\n`).join('');
}
function stageSpecTypeFile(entry, spec) {
  const authored = entry.typed_inputs ? runtimeTypeFile(entry, false) : '';
  const names = new Set([...authored.matchAll(/export type ([A-Za-z_][A-Za-z0-9_]*)\s*=/g)].map(match => match[1]));
  let source = authored;
  for (const [name, body] of Object.entries(spec.types ?? {})) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name) || typeof body !== 'string' || !body.startsWith('{') ||
        names.has(name)) throw Error(`${entry.id}: invalid or duplicate stage type ${name}`);
    names.add(name); source += `export type ${name} = ${body};\n`;
  }
  if (!names.has('FinalSemanticState')) throw Error(`${entry.id}: stage specs need FinalSemanticState`);
  for (const [index, stage] of spec.stages.entries()) {
    if (typeof stage.input_type !== 'string' || !stage.input_type.startsWith('{') ||
        stage.output_type !== 'FinalSemanticState') throw Error(`${entry.id}: invalid stage input/output ${stage.name}`);
    source += `export type StageInput${index + 1} = ${stage.input_type};\n`;
  }
  return source;
}
function checkedStageSpecs(entry, steps) {
  const spec = entry.stage_specs && variant?.program_id === entry.id ?
    structuredClone(entry.stage_specs) : entry.stage_specs;
  if (!spec || spec.schema_version !== 'semantic-stage-contributions/1' ||
      !Array.isArray(spec.stages) || spec.stages.length !== steps.length)
    return null;
  const state = spec.types?.FinalSemanticState ?? '';
  const ownedFields = new Set();
  const noOpField = spec.no_op_field ?? 'stageAbsences';
  if (typeof noOpField !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(noOpField) ||
      !new RegExp(`\\b${noOpField}\\s*:\\s*StageAbsence\\[\\]`).test(state) ||
      typeof spec.types?.StageAbsence !== 'string')
    throw Error(`${entry.id}: stage no-op needs a typed source-grounded absence in FinalSemanticState`);
  if (!entry.typed_inputs && !/^\{[^}]*\bid\s*:\s*string\b/.test(spec.types?.EvidenceRecord ?? ''))
    throw Error(`${entry.id}: typed EvidenceRecord must carry the visible source ID`);
  for (const [index, stage] of spec.stages.entries()) {
    const field = stage.owned_contribution?.field;
    const input = stage.input_type ?? '';
    if (stage.name !== steps[index].name || stage.child !== (steps[index + 1]?.name ?? null) ||
        typeof field !== 'string' || ownedFields.has(field) || !new RegExp(`\\b${field}\\s*:`).test(state) ||
        !new RegExp(`\\b${noOpField}\\s*:\\s*StageAbsence\\[\\]`).test(input) ||
        [...ownedFields].some(previous => !new RegExp(`\\b${previous}\\s*:`).test(input)) ||
        new RegExp(`\\b${field}\\s*:`).test(input) ||
        !Array.isArray(stage.numbered_plan) || stage.numbered_plan.length < 3 ||
        typeof stage.new_semantic_transformation !== 'string')
      throw Error(`${entry.id}: invalid owned stage contribution at ${steps[index].name}`);
    ownedFields.add(field);
  }
  if (variant?.program_id === entry.id) {
    const stage = spec.stages.find(item => item.name === variant.base_stage);
    if (!stage || !stage.scoped_literal_required ||
        !variant.types || !Array.isArray(variant.numbered_plan) ||
        typeof variant.direct_await_idiom !== 'string')
      throw Error(`${entry.id}: stage variant does not match a scoped stage`);
    for (const [name, body] of Object.entries(variant.types)) {
      if (Object.hasOwn(spec.types, name) || typeof body !== 'string' || !body.startsWith('{'))
        throw Error(`${entry.id}: invalid variant type ${name}`);
      spec.types[name] = body;
    }
    stage.numbered_plan = variant.numbered_plan;
    stage.scoped_literal_plan = [variant.literal_unit, variant.capture_policy,
      variant.direct_await_idiom, variant.no_op_criterion, ...variant.invariants];
    stage.no_op_policy = variant.no_op_criterion;
  }
  return spec;
}
function scopedLiteral(entry) {
  const literal = entry.scoped_lambda ?? entry.lambda;
  if (!literal || typeof literal !== 'object') return null;
  const callPath = Array.isArray(entry.call_path) ? entry.call_path : [];
  const literalIndex = callPath.findIndex(name => /runtime literal|scoped literal/i.test(String(name)));
  const placement = literal.placement ?? (literalIndex > 0 ? callPath[literalIndex - 1] : undefined);
  return { placement, signature: literal.signature,
    // The authored plan says "context.related" while the emitted runtime
    // contract names that field relatedRecords. Keep the model-visible plan
    // bound to the actual field instead of teaching it to guess an alias.
    instructions: paragraphs(literal.instructions).join(' ').replace(/\bcontext\.related\b/g, 'context.relatedRecords') };
}
function reference(entry, fixture, typed, expected) {
  const paths = Object.keys(fixture).sort();
  const code = `const paths = ${JSON.stringify(paths)};
const semanticRule = ${JSON.stringify(entry.semantic_rule)};
const objective = ${JSON.stringify(entry.objective)};
type State = { next: number; notes: string[] };
const step = async (state: State): Promise<State> => {
  const currentPath = paths[state.next];
  const source = folder.snapshot().file(currentPath);
  const note = await nl.with<string>({ currentPath, semanticRule, objective })\`Read only the supplied current FileHandle. Return its complete text as a grounded note, preserving its path and all evidence relevant to the captured objective and semanticRule. Do not guess about later files.\`(source);
  return { next: state.next + 1, notes: [...state.notes, note] };
};
const done = await iterateOn(step, { next: 0, notes: [] as string[] })
  .withLimit({ maxSteps: paths.length }).until(state => state.next === paths.length);
return await nl<${typed}>\`Synthesize the typed final result for the captured objective and semanticRule from the complete collected notes. Preserve source identities and apply the program's declared output contract.\`(done.notes);`;
  return {
    root: [evalCall(code), returnCall(expected)],
    children: [
      ...paths.map(path => ({ match: basename(path), expected_reads: [basename(path)],
        calls: [['read_file', { path: basename(path) }], returnCall(fixture[path])] })),
      { match: 'An iterative process has been running for a while.',
        value: { verdict: 'continue', reason: 'The finite current-item frontier shrinks by one on each step.' } },
      { match: 'Synthesize the typed final result', expected_soft_input: 'notes', value: expected },
    ],
  };
}
function render(entry, provider) {
  const id = entry.id;
  if (!safeId(id)) throw Error(`invalid id: ${id}`);
  const domain = shortText(entry.domain, 'domain', id, 3);
  const title = shortText(entry.title, 'title', id);
  const objective = shortText(entry.objective, 'objective', id);
  const rule = shortText(entry.semantic_rule, 'semantic_rule', id);
  const steps = stages(entry);
  if (steps.length < 3) throw Error(`${id}: need at least three distinct stages`);
  const stageSpec = stageSpecsMode ? checkedStageSpecs(entry, steps) : null;
  const fixture = sourceFixture(entry);
  const identities = fixture ? recordIdentityBindings(entry, fixture) : null;
  const expected = exactExpected(entry);
  const comparison = fixture && expected !== undefined ? comparisonSpec(entry, fixture, expected) : null;
  const declaredType = entry.return_type ?? entry.oracle?.return_type;
  if (declaredType !== undefined &&
      (typeof declaredType !== 'string' || !declaredType.trim() || declaredType.length > 1000))
    throw Error(`${id}: invalid declared return_type`);
  const typed = declaredType ?? returnType(expected);
  // Literal keys inside a nested inferred answer map disclose which categories
  // have values (or which hypotheses failed) before the agent reads evidence.
  // Such maps need a generic typed key domain in the source contract.
  if (!entry.typed_inputs && typed && /:\s*\{\s*"/.test(typed))
    throw Error(`${id}: nested literal-key return type may leak gold; declare a generic map`);
  const grouped = groupedAstra && !entry.typed_inputs;
  const literal = scopedLiteral(entry);
  const lambdaInstructions = grouped ? literal?.instructions.replace(/\blocal\b/g, 'each current record') : literal?.instructions;
  const lambda = paragraphs(lambdaInstructions);
  const loop = paragraphs(entry.iteration?.termination ?? entry.iteration_termination ?? entry.scale_plan?.iteration);
  const fullPlan = Array.isArray(entry.functions) && Array.isArray(entry.steps) ?
    entry.steps.map(step => String(step).replace(/^\s*\d+\.\s*/, '')) : [];
  const contract = paragraphs(entry.output_contract);
  const prefix = entry.operation ? `Operation: ${entry.operation}.\n` : '';
  const legacyInstructions = `${prefix}${title}. ${objective}\n\nSemantic rule: ${rule}\n\n` +
    `Carry out these stages over the supplied data, preserving exact source references. ` +
    `Call ${steps[0].name} from solve; each later named stage is a child of the preceding stage and is called from within that active stage. ` +
    (entry.typed_inputs ? `Pass the typed bundle, inquiry, and carried StageState through the chain:\n` : grouped ?
      `Use iterateOn over the sorted source folders. State holds nextFolderIndex and findings. Read the records in each folder and attach their exact id from the source ID binding table below, making EvidenceBatch {folderPath,records:EvidenceRecord[]}. Each step calls ${steps[0].name} with that batch and an EvidenceContext carrying objective, rule, related records from the whole source collection, prior findings, and an initially empty stageFindings array. Every named stage works over the current batch, preserving a separate EvidenceFinding for every current record. Append exactly one final finding for each batch record and advance nextFolderIndex. The unprocessed-folder count decreases by one; finish when nextFolderIndex equals the number of source folders. Then synthesize the declared result from all per-record findings and exact source records. The named stages are:\n` :
      `Use iterateOn over a finite, sorted list of source paths. State holds nextPathIndex and findings. Each step reads that one current file, attaches its exact id from the source ID binding table below, and calls ${steps[0].name} once with item={id,path,text} and an EvidenceContext carrying objective, rule, relevant related records, prior findings, and an initially empty stageFindings array. Append the returned EvidenceFinding and advance nextPathIndex. The unprocessed-path count decreases by one; finish when nextPathIndex equals the number of source paths. Each nested stage receives that same current EvidenceRecord and the updated context. Then synthesize the declared result from all findings and exact source records. The named stages are:\n`) +
    (identities ? `Source ID binding (use these exact identifiers in the declared result):\n${identities.map(({ id, path }) => `${id}: ${path}`).join('\n')}\n\n` : '') +
    steps.map((step, index) => `${index + 1}. ${step.name}: ${step.instructions}`).join('\n') + '\n\n' +
    (fullPlan.length ? `Whole-program control flow:\n${fullPlan.map((step, index) => `${index + 1}. ${step}`).join('\n')}\n\n` : '') +
    (grouped ? `Within each current batch, formulate focused typed natural-language function literals with short numbered instruction plans to interpret the records and their cross-record relations. ` +
      `Return one source-linked finding per record, including unresolved facts; keep each record's sourcePath and evidenceIds distinct. ` :
      `For each current source record, formulate a focused typed natural-language function literal with a short numbered instruction plan. ` +
      `Capture only the current task criterion and pass that item's content and the necessary carried state as arguments. `) +
    `Do not substitute a keyword test for the semantic relation.\n` +
    (lambda.length ? `Scoped child guidance: ${lambda.join(' ')}\n` : '') +
    (loop.length ? `Iteration: ${loop.join(' ')}\n` : '') +
    (contract.length ? `Output contract: ${contract.join(' ')}\n` : '') +
    `Use exact code only for reading, ordering, serialization, and checks; the domain judgments belong in natural-language calls. ` +
    `Return the result specified by the output contract, grounded in the supplied sources.`;
  const stageInstructions = stageSpec ? `${prefix}${title}. ${objective}\n\nSemantic rule: ${rule}\n\n` +
    (identities ? `Source ID binding (attach id and path to every EvidenceRecord, and use these exact IDs in the result):\n${identities.map(({ id, path }) => `${id}: ${path}`).join('\n')}\n\n` : '') +
    (entry.typed_inputs ? `Read the supplied directory snapshot and inquiry into the authored typed bundle. Call ${steps[0].name} with StageInput1 {bundle,inquiry,prior:{${stageSpec.no_op_field ?? 'stageAbsences'}:[]}}. The named stages must each add their own source-backed typed contribution before calling the next stage within the active call. Synthesize the declared answer from the returned FinalSemanticState, not by re-solving from the raw folder.\n` :
      `Use iterateOn over the sorted source folders. Each step builds a typed EvidenceBatch of {id,path,text} records for one current folder and calls ${steps[0].name} with StageInput1 {batch,context,prior:{stageAbsences:[]}}. Append the returned FinalSemanticState to the loop's collected states, then advance the folder frontier; the unprocessed-folder count decreases by one until all folders are consumed. Each named stage adds its owned, source-linked contribution and passes that accumulated state to its child. Synthesize the declared result from all collected states, preserving cross-folder evidence and unresolved claims.\n`) +
    `The named stages are ${steps.map(step => step.name).join(' → ')}. Their typed inputs, owned contribution fields, source witnesses and plans are in the callable .nl files; read those files when called. A stage with no applicable evidence records a source-grounded absence instead of silently forwarding. An ad hoc natural-language literal counts only when invoked and its decision changes the accumulated state.\n` +
    `Output contract: ${contract.join(' ')}\nUse exact code for file reading, binding, finite iteration, coverage checks and serialization. Semantic judgments belong in the named functions and invoked literals. Return the declared typed answer grounded in their FinalSemanticState.` : null;
  const instructions = stageInstructions ?? legacyInstructions;
  const files = { 'solve.nl': nlFile({ args: {}, returns: typed ?? 'string', description: title, kind: 'directory-reducer', instructions }) };
  const typeFile = stageSpec ? stageSpecTypeFile(entry, stageSpec) : runtimeTypeFile(entry, grouped);
  if (typeFile) files['types.ts'] = typeFile;
  let stageFolder = 'solve';
  for (const [index, step] of steps.entries()) {
    const name = step.name;
    const next = steps[index + 1]?.name;
    if (stageSpec) {
      const owned = stageSpec.stages[index];
      stageFolder += `/${name}`;
      const plan = owned.numbered_plan.join('\n');
      const literalPlan = owned.scoped_literal_required ?
        `\nFormulate and invoke a scoped typed natural-language literal; pass its returned decision into ${owned.owned_contribution.field}.\n${paragraphs(owned.scoped_literal_plan).join('\n')}` : '';
      files[`${stageFolder}.nl`] = nlFile({ args: { input: `StageInput${index + 1}` },
        returns: 'FinalSemanticState', description: `${title}: ${name}`,
        instructions: `${plan}\nOwned new semantic contribution: ${owned.owned_contribution.field}: ${owned.owned_contribution.type}.\n` +
          `Transformation: ${owned.new_semantic_transformation}\n` +
          `Evidence: ${owned.source_evidence_obligation}\nInvariant: ${owned.invariant}\n` +
          `No applicable evidence: ${owned.no_op_policy}${literalPlan}\n` +
          (next ? `Pass the accumulated typed contributions to child ${next} within this active call and return its complete FinalSemanticState.\n` :
            `Return the complete FinalSemanticState with every earlier contribution preserved.\n`) });
      continue;
    }
    const scoped = literal?.placement === step.name && literal.instructions
      ? grouped ? `\nAt this stage, formulate ad hoc typed natural-language literals for the current batch. Use this short plan: ${lambdaInstructions} Pass the batch's records and relevant context as arguments; keep individual source IDs in each finding.` :
        `\nAt this stage, formulate an ad hoc typed natural-language literal ${literal.signature ?? ''} for the current item. Use this short plan: ${literal.instructions} Pass the current item and relevant context as arguments; keep source IDs in the finding.` : '';
    const delegate = next ? entry.typed_inputs ?
      `\nAfter making this stage's finding, call the child ${next} within this active call with bundle, inquiry, and the carried StageState. Return its StageState with this stage's source-linked evidence preserved.` :
      grouped ? `\nAfter making this stage's per-record findings, call the child ${next} within this active call. Pass the same EvidenceBatch and an EvidenceContext whose stageFindings includes this stage's findings. Return the child's EvidenceBatchFinding, with one finding for each batch record and all relevant source identities preserved.` :
      `\nAfter making this stage's finding, call the child ${next} within this active call. Pass the same item and an EvidenceContext whose stageFindings includes this stage's EvidenceFinding; return the child's EvidenceFinding with all relevant source identities preserved.` : '';
    stageFolder += `/${name}`;
    const instructionBody = `${stageBody(step)}${scoped}${delegate}\nApply the semantic rule to the supplied ${entry.typed_inputs ? 'bundle, inquiry, and prior state' : grouped ? 'current EvidenceBatch and EvidenceContext' : 'current item EvidenceRecord and EvidenceContext'}. Return a grounded ${entry.typed_inputs ? 'StageState with preserved source identities and carried work' : grouped ? 'EvidenceBatchFinding with folderPath=item.folderPath and exactly one EvidenceFinding per item.records path, each citing evidenceIds and unresolved claims' : 'EvidenceFinding with sourcePath=item.path, a concise finding, cited evidenceIds, and unresolved claims'}.`;
    const selfName = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (new RegExp(`\\b(?:call|invoke|delegate(?:\\s+to)?)\\s+${selfName}\\b`, 'i').test(instructionBody))
      throw Error(`${id}: ${name} instructions direct a self-call`);
    files[`${stageFolder}.nl`] = nlFile({ args: runtimeStageArgs(step, entry, grouped),
      returns: entry.typed_inputs ? 'StageState' : grouped ? 'EvidenceBatchFinding' : 'EvidenceFinding',
      description: `${title}: ${step.name}`, instructions: instructionBody });
  }
  const filePaths = fixture ? Object.keys(fixture) : [];
  const hasScopedCollection = filePaths.length >= 3 &&
    new Set(filePaths.map(path => basename(path))).size === filePaths.length;
  const generationReady = entry.generation_ready === true || entry.admission?.generation_ready === true;
  if (reviewIds.has(id) && (!hasScopedCollection || !typed || filePaths.length < 12))
    throw Error(`${id}: review override requires a typed, expanded source with at least 12 files`);
  const candidate = hasScopedCollection && typed && comparison && (!stageSpecsMode || stageSpec) &&
    (pilotAstra || generationReady || reviewIds.has(id));
  const admission = candidate ? (reviewIds.has(id) ? 'diagnostic_expanded_source_review_only' : pilotAstra ? 'diagnostic_seed_only' :
    'runnable_candidate_pending_reference_and_semantic_review') :
    !comparison ? 'held_missing_declared_answer_comparison' :
    stageSpecsMode && !stageSpec ? 'held_missing_owned_stage_contract' :
    hasScopedCollection && typed ? 'held_design_only_pending_expanded_fixture_review' :
      'held_missing_scoped_collection_or_exact_typed_oracle';
  // A catalog may ask several independent questions of one authored world.
  // Keep that whole evidence world in one split, even when program IDs differ.
  const worldKey = entry.source?.world_group ?? entry.source_fixture?.world_family_id ??
    entry.source?.split_unit ?? id;
  if (!safeId(String(worldKey).replace(/[:]/g, '-')))
    throw Error(`${id}: invalid source world identity`);
  const sourceGroup = `semantic-stack:${provider}:${worldKey}`;
  const programSourceId = `semantic-stack:${provider}:${id}`;
  const caseRow = candidate ? curriculumCase({
    family: 'semantic_stack_authored', shape: id, variant: 'source-v1', splitGroup: sourceGroup,
    slice: 'iterate', domain: 'other', mode: 'single_call', inline: 'required', iterate: 'required',
    evidence: { world: [title, rule], retrieved: Object.keys(fixture), background: [] },
    minimumSequence: [
      grouped ? 'read the task contract and inspect the current source folder as typed records' :
        'read the task contract and inspect the current item only',
      'formulate scoped natural-language literal instructions for the current semantic judgment',
      grouped ? 'carry one typed finding per source path through the named stages and folder iteration' :
        'carry typed child findings through the program-specific stages and structural iteration',
      'synthesize the declared output from grounded findings',
    ],
    reference: reference(entry, fixture, typed, expected),
    root: { name: 'solve', args: {}, returns: typed, kind: 'directory-reducer', instructions },
    files: Object.fromEntries(Object.entries(files).filter(([path]) => path !== 'solve.nl')),
    inputs: {}, expected, folderFiles: fixture,
    expectedFiles: { ...fixture, ...(entry.oracle?.expected_files ?? {}) },
  }) : null;
  if (caseRow) {
    caseRow.semantics.oracle = { level: 'normalized', normalization: 'semantic-structured',
      comparison_spec: comparison,
      context: { objective, semantic_rule: rule, source_files: fixture } };
    caseRow.id = `semantic-stack:${provider}:${id}`;
    caseRow.source_ids = [programSourceId];
    caseRow.source_groups = [sourceGroup];
    caseRow.source_revisions = ['semantic-stack-authored-v1'];
    if (variant?.program_id === id) caseRow.source_revisions.push(`stage-variant:${sha(variantBytes)}`);
    caseRow.generation.collection = 'teacher';
    caseRow.generation.source_domain = domain;
    caseRow.generation.source_quality = 'Authored program and fixture. Exact source/reference and independent semantic review required before training admission.';
    caseRow.collection_guidance = { training_admission: false,
      review_scope: 'source-reference-replay-and-sampled-action-semantic-review' };
  }
  return { id, domain, provider, title, files, fixture, expected, typed, sourceGroup, admission, caseRow };
}

await mkdir(output, { recursive: false });
const rendered = [], ids = new Set(), domainsByProvider = new Map();
const signaturesByProvider = new Map();
for (const [index, path] of (pilotAstra ? catalogs.slice(0, 1) : catalogs).entries()) {
  const provider = index ? 'sol' : 'astra';
  const raw = await readFile(path);
  if (variant?.program_id?.startsWith(`${provider === 'astra' ? 'astra' : 'sol'}-`) &&
      sha(raw) !== variant.base_catalog_sha256)
    throw Error(`${variant.program_id}: stage variant base catalog SHA differs from current source`);
  const parsed = JSON.parse(raw);
  const entries = Array.isArray(parsed) ? parsed : parsed.programs;
  if (!Array.isArray(entries) || entries.length < (pilotAstra ? 1 : 100))
    throw Error(`${provider}: catalog needs at least ${pilotAstra ? 1 : 100} programs`);
  domainsByProvider.set(provider, new Set());
  signaturesByProvider.set(provider, new Set());
  for (const entry of entries) {
    const signature = JSON.stringify([entry.operation, entry.objective, entry.semantic_rule,
      entry.steps ?? entry.functions]);
    if (signaturesByProvider.get(provider).has(signature))
      throw Error(`${provider}: duplicate semantic program body: ${entry.id}`);
    signaturesByProvider.get(provider).add(signature);
    const program = render(entry, provider);
    if (ids.has(program.id)) throw Error(`duplicate program id: ${program.id}`);
    ids.add(program.id); domainsByProvider.get(provider).add(program.domain);
    const directory = join(output, 'programs', provider, program.id);
    await mkdir(directory, { recursive: true });
    for (const [file, body] of Object.entries(program.files)) {
      const target = join(directory, file);
      await mkdir(resolve(target, '..'), { recursive: true });
      await writeFile(target, body);
    }
    if (program.fixture) {
      for (const [file, body] of Object.entries(program.fixture)) {
        const target = join(directory, 'data', file);
        await mkdir(resolve(target, '..'), { recursive: true });
        await writeFile(target, body);
      }
    }
    rendered.push(program);
  }
  if (domainsByProvider.get(provider).size < (pilotAstra ? 1 : 15))
    throw Error(`${provider}: fewer than 15 distinct domains`);
  await writeFile(join(output, `${provider}-catalog.sha256`), `${sha(raw)}  ${path}\n`);
}
const candidateRows = rendered.filter(item => item.caseRow).map(item => item.caseRow);
for (const id of reviewIds) if (!rendered.some(item => item.id === id && item.caseRow))
  throw Error(`${id}: requested review ID is absent or not a runnable source`);
const verification = candidateRows.length ? await verifyCases(candidateRows, TOOLS_PROMPT) : [];
const verified = new Map(verification.map(item => [item.id, item]));
const caseRows = candidateRows.filter(row => verified.get(row.id)?.ok);
const sourceText = caseRows.map(row => JSON.stringify(row)).join('\n') + (caseRows.length ? '\n' : '');
await writeFile(join(output, 'source-candidates.ir.jsonl'), sourceText);
const review = {
  schema: 'natlang.semantic-stack-source-review/1',
  partial_pilot: pilotAstra,
  presentation: stageSpecsMode ? 'owned-typed-stage-contributions' : groupedAstra ?
    'typed-coherent-folder-batches' : 'typed-individual-records',
  ...(variant ? { stage_variant: { id: variant.id, program_id: variant.program_id,
    file: variantPath, sha256: sha(variantBytes), base_catalog_sha256: variant.base_catalog_sha256 } } : {}),
  review_override_ids: [...reviewIds],
  programs: rendered.length, runnable_candidates: candidateRows.length,
  reference_verified_candidates: caseRows.length,
  held: rendered.length - caseRows.length,
  domains: Object.fromEntries([...domainsByProvider].map(([provider, domains]) => [provider, [...domains].sort()])),
  source_sha256: sha(sourceText), teacher_observations: 0, training_admission: false,
  cases: rendered.map(({ id, provider, domain, admission, sourceGroup, caseRow }) => {
    const proof = caseRow ? verified.get(caseRow.id) : undefined;
    return { id, provider, domain, source_group: sourceGroup,
      status: proof ? (proof.ok ? admission : 'held_reference_replay_failed') : admission,
      ...(proof ? { reference_ok: proof.ok, reference_problems: proof.problems,
        opening_sha256: proof.opening_sha256 } : {}) };
  }),
};
await writeFile(join(output, 'source-review.json'), JSON.stringify(review, null, 2) + '\n');
console.log(JSON.stringify({ output, programs: rendered.length, candidates: caseRows.length,
  held: review.held, reference_failures: candidateRows.length - caseRows.length,
  source_sha256: review.source_sha256 }));
