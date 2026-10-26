/** Exact source packaging for the authored design-brief programs. No model calls. */
import { createHash } from 'node:crypto';
import { lstat, readFile, readdir } from 'node:fs/promises';
import { basename, isAbsolute, join, relative, resolve } from 'node:path';
import { programDefinition, programNode } from '../../dist/teacher/program.js';
import { definitionNode } from '../../dist/runtime/kernel.js';
import { TypeEnv, fitsType, formatType, parseType } from '../../dist/native/types.js';
import { coerce, problems } from '../../dist/native/values.js';
import { generationHoldReason, quarantineReason } from '../../dist/teacher/curriculum-policy.js';

export const REPO = resolve(import.meta.dirname, '../../..');
export const SOURCE_ROOT = 'codebases/design_briefs';
export const FIXTURE_ROOT = 'training/semantic_applications/design_briefs';
export const RELEASE = 'design-briefs-100-native-20261025-v1';
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
export const json = value => JSON.stringify(value, null, 2) + '\n';
export const jsonl = rows => rows.map(row => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : '');

export function safeRelative(value) {
  if (typeof value !== 'string' || !value || isAbsolute(value) || value.includes('\\') ||
      value.split('/').some(part => !part || part === '.' || part === '..'))
    throw new Error(`unsafe relative path: ${value}`);
  return value;
}

/** The split is fixed for an entire application, including every later variant. */
export function applicationSplit(id) {
  const number = Number(id);
  if (!Number.isInteger(number) || number < 1 || number > 100)
    throw new Error(`invalid application ID: ${id}`);
  const slot = (number - 1) % 5;
  return slot < 3 ? 'train' : slot === 3 ? 'dev' : 'test';
}

export async function confinedFile(root, name) {
  safeRelative(name);
  let current = resolve(root);
  for (const component of name.split('/')) {
    current = join(current, component);
    if ((await lstat(current)).isSymbolicLink()) throw new Error(`symlink in source path: ${name}`);
  }
  if (!(await lstat(current)).isFile()) throw new Error(`not a regular file: ${name}`);
  return current;
}

export async function sourceFiles(root) {
  const files = {};
  async function walk(folder, prefix = '') {
    for (const entry of (await readdir(folder, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith('.') || ['node_modules', 'dist'].includes(entry.name)) continue;
      const name = prefix + entry.name;
      if (entry.isSymbolicLink()) throw new Error(`symlink in source project: ${name}`);
      if (entry.isDirectory()) await walk(join(folder, entry.name), name + '/');
      else if (entry.isFile() && /\.(nl|ts)$/.test(name) && !name.endsWith('.d.nl.ts'))
        files[name] = await readFile(join(folder, entry.name), 'utf8');
    }
  }
  await walk(root);
  return files;
}

export function sourceDigest(files) {
  const hash = createHash('sha256');
  for (const name of Object.keys(files).sort()) hash.update(name).update('\0').update(files[name]).update('\0');
  return hash.digest('hex');
}

export function oracleLevel(spec) {
  const level = typeof spec === 'string' ? spec : spec?.level;
  if (!['exact', 'normalized', 'judged'].includes(level))
    throw new Error('fixture oracle must use the existing exact, normalized, or judged OracleSpec');
  return level;
}

/** Preserve the six concrete domain records from each brief, including enums and exact value types. */
export function checkDomainContracts(files, contracts) {
  const node = programNode({ semantics: { root: 'runCase.nl', files } });
  const scalarNames = Object.fromEntries(['ID', 'Ref', 'SpanRef', 'LocalDate', 'DateTime'].map(name => [name, parseType('string')]));
  const env = new TypeEnv(node.types, new TypeEnv(scalarNames));
  let fields = 0;
  for (const [name, required] of Object.entries(contracts)) {
    const record = env.resolve(parseType(name));
    if (record.kind !== 'record') throw new Error(`${name}: expected a concrete domain record`);
    for (const [fieldName, expected] of Object.entries(required)) {
      const field = record.fields.find(item => item.name === fieldName);
      const expectedType = parseType(expected);
      if (!field || field.optional || !fitsType(field.type, expectedType, env) || !fitsType(expectedType, field.type, env))
        throw new Error(`${name}.${fieldName}: domain contract must remain ${expected}`);
      fields++;
    }
  }
  for (const [name, expected] of Object.entries({
    Decimal: 'string', Ratio: '{ numerator: string; denominator: string }',
    Quantity: '{ value: Decimal; unit: string }', Money: '{ minorUnits: string; currency: string; scale: number }',
  })) {
    const actualType = env.resolve(parseType(name)), expectedType = parseType(expected);
    if (!fitsType(actualType, expectedType, env) || !fitsType(expectedType, actualType, env))
      throw new Error(`${name}: exact platform value shape changed`);
  }
  return { records: Object.keys(contracts).length, fields };
}

/** A prose reference cannot make an out-of-scope sibling into an actual callable. */
export function checkNativeCalls(files) {
  const paths = Object.keys(files).filter(path => path.endsWith('.nl'));
  const names = new Set(paths.map(path => basename(path, '.nl')));
  const sharedTypes = files['runCase.nl'] ? programNode({ semantics: { root: 'runCase.nl', files } }).types : {};
  let references = 0;
  for (const path of paths) {
    const definition = programDefinition({ semantics: { root: path, files } });
    const localTypes = definitionNode(definition, definition.params.map(() => undefined)).types;
    for (const [name, type] of Object.entries(sharedTypes)) {
      if (/^(Draft|StageResult)/.test(name) && localTypes[name] && JSON.stringify(type) !== JSON.stringify(localTypes[name]))
        throw new Error(`${path}: ${name} shadows the shared draft/stage contract`);
    }
    const visible = new Set(Object.keys(definition.codebase ?? {}));
    for (const name of names) {
      const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const call = new RegExp(`\\b${escaped}\\s*\\(|\\b(?:call|calling|invoke|invoking)\\s+${escaped}\\b`, 'i');
      if (call.test(definition.body)) {
        if (!visible.has(name)) throw new Error(`${path}: referenced helper ${name} is outside its native callable scope`);
        references++;
      }
    }
  }
  return { functions: paths.length, visible_references: references };
}

/** Uses the real native type system; structural checks never grade semantic meaning. */
export function checkFixture(files, fixture) {
  if (typeof fixture.id !== 'string' || !fixture.id || typeof fixture.scenario !== 'string' || !fixture.scenario.trim())
    throw new Error('fixture needs an ID and a substantive scenario');
  if (typeof fixture.split_group !== 'string' || !fixture.split_group.trim())
    throw new Error(`${fixture.id}: missing world/split group`);
  safeRelative(fixture.root);
  if (!fixture.root.endsWith('.nl') || !Object.hasOwn(files, fixture.root))
    throw new Error(`${fixture.id}: root is not a packaged .nl function`);
  if (!fixture.inputs || typeof fixture.inputs !== 'object' || Array.isArray(fixture.inputs))
    throw new Error(`${fixture.id}: inputs must be named parameters`);
  if (!Object.hasOwn(fixture, 'expected')) throw new Error(`${fixture.id}: missing independent expected result`);
  const level = oracleLevel(fixture.oracle);
  if (level === 'judged' && (typeof fixture.oracle.rubric !== 'string' || fixture.oracle.rubric.trim().length < 50))
    throw new Error(`${fixture.id}: judged workflow requires an explicit substantive rubric`);
  const record = { semantics: { root: fixture.root, files, inputs: fixture.inputs, expected: fixture.expected } };
  const node = programNode(record);
  const env = new TypeEnv(node.types);
  for (const parameter of node.type.params.fields) {
    const value = coerce(fixture.inputs[parameter.name], parameter.type, env, `input.${parameter.name}`);
    const missing = problems(value, parameter.type, env, `input.${parameter.name}`).holes;
    if (missing.length) throw new Error(`${fixture.id}: incomplete input: ${JSON.stringify(missing)}`);
  }
  if (level !== 'judged') {
    const value = coerce(fixture.expected, node.type.returns, env, 'expected');
    const missing = problems(value, node.type.returns, env, 'expected').holes;
    if (missing.length) throw new Error(`${fixture.id}: incomplete expected value: ${JSON.stringify(missing)}`);
  }
  return { level, returns: formatType(node.type.returns) };
}

/** Bind workflow coverage to the original brief's five use cases, not to stage names. */
export function checkOperationalCoverage(app, fixtures, original) {
  if (!original || original.source_sha256 !== app.source_sha256 ||
      !Array.isArray(original.details) || original.details.length !== 5)
    throw new Error('missing original operational-detail contract or source binding');
  if (JSON.stringify(app.operational_details) !== JSON.stringify(original.details))
    throw new Error('operational details must retain the five original brief use cases verbatim');
  const workflows = fixtures.filter(fixture => fixture.root === 'runCase.nl');
  if (workflows.length !== 5 || new Set(workflows.map(fixture => fixture.operational_detail_id)).size !== 5)
    throw new Error('five workflows must cover five distinct original operational details');
  for (const fixture of workflows) {
    const id = fixture.operational_detail_id;
    if (!Number.isInteger(id) || id < 1 || id > 5 || fixture.brief_detail !== original.details[id - 1])
      throw new Error(`${fixture.id}: workflow does not bind its original operational detail`);
    if (oracleLevel(fixture.oracle) !== 'judged')
      throw new Error(`${fixture.id}: operational workflows require independent review`);
    const inspectObservations = value => {
      if (!value || typeof value !== 'object') return;
      for (const [key, item] of Object.entries(value)) {
        if (key === 'text' && typeof item === 'string' && item.includes(fixture.brief_detail))
          throw new Error(`${fixture.id}: an operational requirement was copied into observed evidence`);
        if (item && typeof item === 'object') inspectObservations(item);
      }
    };
    inspectObservations(fixture.inputs);
    for (const boundary of ['view', 'durable_state', 'effects']) {
      const assertion = fixture.boundary_assertions?.[boundary];
      if (typeof assertion !== 'string' || assertion.trim().length < 35)
        throw new Error(`${fixture.id}: missing substantive ${boundary} proposal assertion`);
    }
  }
  if (new Set(workflows.map(fixture => JSON.stringify(fixture.inputs))).size !== 5)
    throw new Error('operational workflows must supply distinct observable input worlds');
  return { original_details: original.details.length, mapped_workflows: workflows.length };
}

export function makeRecord(app, files, fixture, fixtureHash) {
  const { level } = checkFixture(files, fixture);
  const applicationId = String(app.application_id).padStart(3, '0');
  const group = `design-briefs-v1:application-${applicationId}`;
  const split = applicationSplit(applicationId);
  const exact = level === 'exact';
  const expected = level === 'judged' ? {
    requirements: fixture.expected,
    observations: fixture.inputs,
    domain_gates: app.stages.map(stage => ({ stage: stage.name, gate: stage.gate, algorithm: stage.algorithm })),
    operational_detail: { id: fixture.operational_detail_id, requirement: fixture.brief_detail },
    boundary_assertions: fixture.boundary_assertions,
  } : fixture.expected;
  const row = {
    version: 'natlang.program/2',
    id: `design-briefs-v1:${applicationId}:${fixture.id}`,
    kind: 'lambda_graph', family: `design_brief_${applicationId}`, split,
    source: 'user-design-brief-native-program',
    source_ids: [app.source_relative_path], source_groups: [group],
    source_revisions: [`sha256:${app.source_sha256}`, `sha256:${sourceDigest(files)}`],
    license: 'user-supplied-design; repository-authored-implementation',
    gold_sources: [exact ? 'independently-authored-finite-gate-fixture' : 'independent-workflow-review-rubric'],
    generation: {
      generator: 'natlang.design-brief-freezer/1', release: RELEASE,
      authorship: 'Luna engineering agents; source and synthetic fixtures only',
      provider_calls: 0, teacher_trajectory: false,
      brief_sha256: app.source_sha256, fixture_sha256: fixtureHash,
      operational_detail_id: fixture.operational_detail_id,
    },
    curriculum: {
      split_group: group, world_group: fixture.split_group,
      rl: { level: exact ? 0 : 3, family: `design_brief_${applicationId}`,
        qualification: 'source-only-live-pending' },
    },
    collection_guidance: {
      training_admission: false, teacher: 'self-hosted-only',
      review_required: !exact, actual_model_measurement: 'pending',
      reward: exact ? 'exact-task-outcome' : 'independent-semantic-review-required',
      graph: 'retain-whole-actual-execution; no mandatory Neuralese traffic',
    },
    semantics: { root: fixture.root, files, inputs: fixture.inputs, expected, oracle: fixture.oracle },
  };
  const held = quarantineReason(row) ?? generationHoldReason(row);
  if (held) throw new Error(`${row.id}: current repository source policy: ${held}`);
  return row;
}

export async function readApplication(repo, name, sourceIndex, domainContracts, operationalContracts) {
  safeRelative(name);
  if (!/^\d{3}-[a-z0-9-]+$/.test(name)) throw new Error(`invalid application directory: ${name}`);
  const appRoot = join(repo, SOURCE_ROOT, name);
  const metadataPath = await confinedFile(appRoot, 'brief.json');
  const metadataBytes = await readFile(metadataPath);
  const app = JSON.parse(metadataBytes);
  const id = String(app.application_id).padStart(3, '0');
  const original = sourceIndex.applications.find(item => Number(item.id) === Number(id));
  if (!original || !name.startsWith(id + '-') || app.source_sha256 !== original.sha256 ||
      app.source_relative_path !== original.path || Number(app.domain_id) !== Number(original.domain_id))
    throw new Error(`${name}: source brief provenance differs from the original archive manifest`);
  if (!Array.isArray(app.stages) || app.stages.length !== 5 ||
      new Set(app.stages.map(stage => stage.name)).size !== 5 ||
      app.stages.some(stage => !stage.algorithm?.trim() || !stage.gate?.trim() || !stage.output_type?.trim()))
    throw new Error(`${name}: five distinct domain algorithms and gates required`);
  const files = await sourceFiles(appRoot);
  const nlFiles = Object.keys(files).filter(path => path.endsWith('.nl'));
  if (!files['runCase.nl'] || nlFiles.length < 36)
    throw new Error(`${name}: incomplete root and semantic decomposition`);
  const contracts = domainContracts ?? JSON.parse(await readFile(join(repo, FIXTURE_ROOT, 'domain-contracts.json'), 'utf8'));
  if (!contracts.applications[id]) throw new Error(`${name}: missing original domain contracts`);
  const contractCheck = checkDomainContracts(files, contracts.applications[id]);
  const callCheck = checkNativeCalls(files);
  await confinedFile(appRoot, 'DECOMPOSITION.md');
  for (const stage of app.stages) {
    const stem = `runCase/${stage.name.replace(/\.nl$/, '')}`;
    for (const path of [stem + '.nl', ...['prepare', 'decide', 'construct', 'audit'].map(child => `${stem}/${child}.nl`)])
      if (!files[path]) throw new Error(`${name}: missing domain module ${path}`);
  }
  const fixturePath = await confinedFile(join(repo, FIXTURE_ROOT, name), 'cases.json');
  const fixtureBytes = await readFile(fixturePath);
  const cases = JSON.parse(fixtureBytes);
  if (!Array.isArray(cases) || cases.length !== 8 || new Set(cases.map(item => item.id)).size !== cases.length)
    throw new Error(`${name}: three counterfactual gate cases and five operational workflows are required`);
  const operational = operationalContracts ?? JSON.parse(await readFile(join(repo, FIXTURE_ROOT, 'operational-contracts.json'), 'utf8'));
  const operationalCoverage = checkOperationalCoverage(app, cases, operational.applications[id]);
  const records = cases.map(fixture => makeRecord(app, files, fixture, sha256(fixtureBytes)));
  const gateCases = records.filter(row => oracleLevel(row.semantics.oracle) === 'exact');
  const workflows = records.filter(row => row.semantics.root === 'runCase.nl');
  if (gateCases.length !== 3 || workflows.length !== 5)
    throw new Error(`${name}: missing gate or full-workflow coverage`);
  for (const verdict of ['pass', 'fail', 'needs-review'])
    if (!gateCases.some(row => row.semantics.expected === verdict))
      throw new Error(`${name}: no exact ${verdict} counterfactual`);
  for (const row of gateCases.filter(item => item.semantics.expected === 'pass')) {
    const stage = app.stages.find(item => row.semantics.root.startsWith(`runCase/${item.name.replace(/\.nl$/, '')}/`));
    const required = contracts.applications[id][stage?.output_type];
    const draft = row.semantics.inputs.draft;
    const values = draft?.values ?? draft?.value ?? draft;
    if (!required || !values || Object.keys(required).some(field => !Object.hasOwn(values, field)))
      throw new Error(`${name}: a positive domain gate fixture must provide all fields of its output record`);
  }
  const visibleGateInputs = gateCases.map(row => JSON.stringify(row.semantics.inputs));
  if (new Set(visibleGateInputs).size !== gateCases.length)
    throw new Error(`${name}: identical visible gate inputs have multiple answer records`);
  return { name, app, files, records, metadata_sha256: sha256(metadataBytes), fixture_sha256: sha256(fixtureBytes),
    source_sha256: sourceDigest(files), native_functions: nlFiles.length, domain_contracts: contractCheck,
    callable_scope: callCheck, operational_coverage: operationalCoverage };
}

export async function verifyPacket(directory) {
  const root = resolve(directory);
  const manifest = JSON.parse(await readFile(await confinedFile(root, 'manifest.json'), 'utf8'));
  if (manifest.schema !== 'natlang.design-brief-packet/1' || !Array.isArray(manifest.files))
    throw new Error('unknown design-brief packet manifest');
  if (new Set(manifest.files.map(item => item.path)).size !== manifest.files.length)
    throw new Error('duplicate packet manifest member');
  for (const member of manifest.files) {
    const bytes = await readFile(await confinedFile(root, member.path));
    if (bytes.length !== member.bytes || sha256(bytes) !== member.sha256)
      throw new Error(`packet content changed: ${member.path}`);
  }
  return manifest;
}
