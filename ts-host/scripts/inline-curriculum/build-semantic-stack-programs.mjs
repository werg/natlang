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
const pilotAstra = args[2] === '--pilot-astra';
if ((args.length !== 2 && !(args.length === 3 && pilotAstra)) || args[0] !== '--out' || !args[1])
  throw Error('usage: node build-semantic-stack-programs.mjs --out FRESH_DIRECTORY [--pilot-astra]');
const output = resolve(args[1]);
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
function exactExpected(entry) {
  const oracle = entry.oracle;
  if (oracle && typeof oracle === 'object' && Object.hasOwn(oracle, 'expected')) return oracle.expected;
  if (Object.hasOwn(entry, 'expected')) return entry.expected;
  return undefined;
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
function runtimeStageArgs(step, entry) {
  if (!entry.typed_inputs || !step.signature) return { item: 'string', context: 'string' };
  const signature = /^[^(]+\(([^)]*)\)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)$/.exec(step.signature);
  if (!signature) throw Error(`${entry.id}: invalid named stage signature ${step.signature}`);
  const args = Object.fromEntries(signature[1].split(',').map(part => {
    const match = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(part);
    if (!match) throw Error(`${entry.id}: invalid stage argument ${part}`);
    return [match[1], match[2]];
  }));
  return args;
}
function runtimeTypeFile(entry) {
  if (!entry.typed_inputs) return null;
  const bundle = /^([A-Za-z_][A-Za-z0-9_]*)\s*=/.exec(entry.typed_inputs.bundle ?? '')?.[1];
  if (!bundle) throw Error(`${entry.id}: missing typed bundle alias`);
  // The authored stage signatures retain their actual named types. The source
  // fixture provides concrete data; these common structural aliases avoid
  // pretending an external directory service is implemented in generated TS.
  return `export type ${bundle} = { files: unknown; records: unknown[]; context: unknown };\n` +
    `export type Inquiry = { question: string; scope: string[]; requestedOutput: string };\n` +
    `export type EvidenceUnit = { id: string; path: string; text: string };\n` +
    `export type RelationDecision = { status: string; evidenceIds: string[]; detail: string };\n` +
    `export type WorkItem = { id: string; kind: string };\n` +
    `export type StageState = { records: EvidenceUnit[]; relations: RelationDecision[]; open: WorkItem[]; done: string[] };\n`;
}
function scopedLiteral(entry) {
  const literal = entry.scoped_lambda ?? entry.lambda;
  if (!literal || typeof literal !== 'object') return null;
  const callPath = Array.isArray(entry.call_path) ? entry.call_path : [];
  const literalIndex = callPath.findIndex(name => /runtime literal|scoped literal/i.test(String(name)));
  const placement = literal.placement ?? (literalIndex > 0 ? callPath[literalIndex - 1] : undefined);
  return { placement, signature: literal.signature,
    instructions: paragraphs(literal.instructions).join(' ') };
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
  const fixture = sourceFixture(entry);
  const expected = exactExpected(entry);
  const declaredType = entry.return_type ?? entry.oracle?.return_type;
  if (declaredType !== undefined &&
      (typeof declaredType !== 'string' || !declaredType.trim() || declaredType.length > 1000))
    throw Error(`${id}: invalid declared return_type`);
  const typed = declaredType ?? returnType(expected);
  const literal = scopedLiteral(entry);
  const lambda = paragraphs(literal?.instructions);
  const loop = paragraphs(entry.iteration?.termination ?? entry.iteration_termination ?? entry.scale_plan?.iteration);
  const fullPlan = Array.isArray(entry.functions) && Array.isArray(entry.steps) ?
    entry.steps.map(step => String(step).replace(/^\s*\d+\.\s*/, '')) : [];
  const contract = paragraphs(entry.output_contract);
  const prefix = entry.operation ? `Operation: ${entry.operation}.\n` : '';
  const instructions = `${prefix}${title}. ${objective}\n\nSemantic rule: ${rule}\n\n` +
    `Carry out these stages over the supplied data, preserving exact source references. ` +
    `Call ${steps[0].name} from solve; each later named stage is a child of the preceding stage and is called from within that active stage. ` +
    (entry.typed_inputs ? `Pass the typed bundle, inquiry, and carried StageState through the chain:\n` :
      `Pass the current scoped item and context through the chain:\n`) +
    steps.map((step, index) => `${index + 1}. ${step.name}: ${step.instructions}`).join('\n') + '\n\n' +
    (fullPlan.length ? `Whole-program control flow:\n${fullPlan.map((step, index) => `${index + 1}. ${step}`).join('\n')}\n\n` : '') +
    `For each current item, formulate a focused typed natural-language function literal with a short numbered instruction plan. ` +
    `Capture only the current task criterion and pass that item's content and the necessary carried state as arguments. ` +
    `Do not substitute a keyword test for the semantic relation.\n` +
    (lambda.length ? `Scoped child guidance: ${lambda.join(' ')}\n` : '') +
    (loop.length ? `Iteration: ${loop.join(' ')}\n` : '') +
    (contract.length ? `Output contract: ${contract.join(' ')}\n` : '') +
    `Use exact code only for reading, ordering, serialization, and checks; the domain judgments belong in natural-language calls. ` +
    `Return the result specified by the output contract, grounded in the supplied sources.`;
  const files = { 'solve.nl': nlFile({ args: {}, returns: typed ?? 'string', description: title, kind: 'directory-reducer', instructions }) };
  const typeFile = runtimeTypeFile(entry);
  if (typeFile) files['types.ts'] = typeFile;
  let stageFolder = 'solve';
  for (const [index, step] of steps.entries()) {
    const name = step.name;
    const next = steps[index + 1]?.name;
    const scoped = literal?.placement === step.name && literal.instructions
      ? `\nAt this stage, formulate an ad hoc typed natural-language literal ${literal.signature ?? ''} for the current item. Use this short plan: ${literal.instructions} Pass the current item and relevant context as arguments; keep source IDs in the finding.` : '';
    const delegate = next ? `\nAfter making this stage's finding, call the child ${next} within this active call. Pass the finding and the current source-linked state to that child; return its result together with this stage's evidence.` : '';
    stageFolder += `/${name}`;
    const instructionBody = `${stageBody(step)}${scoped}${delegate}\nApply the semantic rule to the supplied ${entry.typed_inputs ? 'bundle, inquiry, and prior state' : 'current item and context'}. Return a grounded ${entry.typed_inputs ? 'StageState with preserved source identities and carried work' : 'finding with source identity'}.`;
    const selfName = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (new RegExp(`\\b(?:call|invoke|delegate(?:\\s+to)?)\\s+${selfName}\\b`, 'i').test(instructionBody))
      throw Error(`${id}: ${name} instructions direct a self-call`);
    files[`${stageFolder}.nl`] = nlFile({ args: runtimeStageArgs(step, entry),
      returns: entry.typed_inputs ? 'StageState' : 'string',
      description: `${title}: ${step.name}`, instructions: instructionBody });
  }
  const filePaths = fixture ? Object.keys(fixture) : [];
  const hasScopedCollection = filePaths.length >= 3 &&
    new Set(filePaths.map(path => basename(path))).size === filePaths.length;
  const generationReady = entry.generation_ready === true || entry.admission?.generation_ready === true;
  const candidate = hasScopedCollection && typed && (pilotAstra || generationReady);
  const admission = candidate ? (pilotAstra ? 'diagnostic_seed_only' :
    'runnable_candidate_pending_reference_and_semantic_review') :
    hasScopedCollection && typed ? 'held_design_only_pending_expanded_fixture_review' :
      'held_missing_scoped_collection_or_exact_typed_oracle';
  // A catalog may ask several independent questions of one authored world.
  // Keep that whole evidence world in one split, even when program IDs differ.
  const worldKey = entry.source?.world_group ?? entry.source?.split_unit ?? id;
  if (!safeId(String(worldKey).replace(/[:]/g, '-')))
    throw Error(`${id}: invalid source world identity`);
  const sourceGroup = `semantic-stack:${provider}:${worldKey}`;
  const programSourceId = `semantic-stack:${provider}:${id}`;
  const caseRow = candidate ? curriculumCase({
    family: 'semantic_stack_authored', shape: id, variant: 'source-v1', splitGroup: sourceGroup,
    slice: 'iterate', domain: 'other', mode: 'single_call', inline: 'required', iterate: 'required',
    evidence: { world: [title, rule], retrieved: Object.keys(fixture), background: [] },
    minimumSequence: [
      'read the task contract and inspect the current item only',
      'formulate scoped natural-language literal instructions for the current semantic judgment',
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
    caseRow.id = `semantic-stack:${provider}:${id}`;
    caseRow.source_ids = [programSourceId];
    caseRow.source_groups = [sourceGroup];
    caseRow.source_revisions = ['semantic-stack-authored-v1'];
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
const verification = candidateRows.length ? await verifyCases(candidateRows, TOOLS_PROMPT) : [];
const verified = new Map(verification.map(item => [item.id, item]));
const caseRows = candidateRows.filter(row => verified.get(row.id)?.ok);
const sourceText = caseRows.map(row => JSON.stringify(row)).join('\n') + (caseRows.length ? '\n' : '');
await writeFile(join(output, 'source-candidates.ir.jsonl'), sourceText);
const review = {
  schema: 'natlang.semantic-stack-source-review/1',
  partial_pilot: pilotAstra,
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
