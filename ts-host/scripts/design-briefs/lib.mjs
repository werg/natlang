/** Exact source packaging for the authored design-brief programs. No model calls. */
import { createHash } from 'node:crypto';
import { lstat, readFile, readdir } from 'node:fs/promises';
import { basename, isAbsolute, join, relative, resolve, posix } from 'node:path';
import ts from 'typescript';
import { programDefinition, programNode } from '../../dist/teacher/program.js';
import { definitionNode } from '../../dist/runtime/kernel.js';
import { parseModule, PATH_ONLY } from '../../dist/runtime/loader.js';
import { TypeEnv, fitsType, formatType, parseType } from '../../dist/native/types.js';
import { coerce, problems } from '../../dist/native/values.js';
import { generationHoldReason, quarantineReason } from '../../dist/teacher/curriculum-policy.js';

export const REPO = resolve(import.meta.dirname, '../../..');
export const SOURCE_ROOT = 'codebases/design_briefs';
export const FIXTURE_ROOT = 'training/semantic_applications/design_briefs';
export const RELEASE = 'design-briefs-100-distinct-applications-20261026-v3';
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
  let manifest;
  try { manifest = JSON.parse(await readFile(await confinedFile(root, 'application.json'), 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (manifest?.source_files !== undefined) {
    const paths = manifest.source_files;
    if (!Array.isArray(paths) || !paths.length || new Set(paths).size !== paths.length)
      throw new Error('source_files must be a nonempty inventory of distinct native source paths');
    for (const name of [...paths].sort()) {
      safeRelative(name);
      if (!(name === 'natlang.json' || (/\.(nl|ts)$/.test(name) && !name.endsWith('.d.nl.ts'))))
        throw new Error(`not a native source path: ${name}`);
      files[name] = await readFile(await confinedFile(root, name), 'utf8');
    }
    return files;
  }
  async function walk(folder, prefix = '') {
    for (const entry of (await readdir(folder, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith('.') || ['node_modules', 'dist'].includes(entry.name)) continue;
      const name = prefix + entry.name;
      if (entry.isSymbolicLink()) throw new Error(`symlink in source project: ${name}`);
      if (entry.isDirectory()) await walk(join(folder, entry.name), name + '/');
      else if (entry.isFile() && ((/\.(nl|ts)$/.test(name) && !name.endsWith('.d.nl.ts')) || name === 'natlang.json'))
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
export function checkDomainContracts(files, contracts, root = 'runCase.nl') {
  const node = programNode({ semantics: { root, files } });
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
export function checkNativeCalls(files, root = 'runCase.nl') {
  checkRelativeModuleImports(files);
  const paths = Object.keys(files).filter(path => path.endsWith('.nl'));
  const names = new Set(paths.map(path => basename(path, '.nl')));
  const sharedTypes = files[root] ? programNode({ semantics: { root, files } }).types : {};
  const modules = Object.entries(files).filter(([path]) => path.endsWith('.ts') && !path.endsWith('types.ts'))
    .map(([path, text]) => parseModule(path, text, sharedTypes, PATH_ONLY));
  let references = 0;
  for (const path of paths) {
    const definition = programDefinition({ semantics: { root: path, files } });
    const formalNames = declaredInlineNames(files[path]) ?? definition.params.map(parameter => parameter.name);
    if (new Set(formalNames).size !== formalNames.length)
      throw new Error(`${path}: duplicate native parameter names`);
    const localTypes = definitionNode(definition, definition.params.map(() => undefined)).types;
    for (const [name, type] of Object.entries(sharedTypes)) {
      if (/^(Draft|StageResult)/.test(name) && localTypes[name] && JSON.stringify(type) !== JSON.stringify(localTypes[name]))
        throw new Error(`${path}: ${name} shadows the shared draft/stage contract`);
    }
    const visible = new Map(nativeBindings(definition.codebase));
    checkModuleReferences(definition, modules, path);
    for (const name of new Set([...names, ...visible.keys()].filter(name => !name.includes('.') || visible.get(name)?.kind === 'natlang'))) {
      const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const call = new RegExp(`(?<![\\w$.])${escaped}\\s*\\(|\\b(?:call|calling|invoke|invoking)\\s+${escaped}(?![\\w$.])`, 'i');
      if (call.test(definition.body)) {
        if (!visible.has(name)) throw new Error(`${path}: referenced helper ${name} is outside its native callable scope`);
        if (new RegExp(`[A-Za-z0-9_-]+/${escaped}\\s*\\(`).test(definition.body))
          throw new Error(`${path}: invoke native helper ${name} by its callable name, not a source-file path`);
        const child = visible.get(name);
        if (child.kind === 'namespace')
          throw new Error(`${path}: ${name} is a namespace, not a native callable; qualify its child or move the function into the intended companion folder`);
        if (child.kind === 'natlang') {
          const parameters = Object.keys(child.args ?? {}), minimum = parameters.filter(name => !name.endsWith('?')).length;
          for (const match of definition.body.matchAll(new RegExp(`(?<![\\w$.])${escaped}\\s*\\(`, 'g'))) {
            const args = literalCallArguments(definition.body, match.index + match[0].length);
            if (args && !args.some(arg => arg.includes('...')) && (args.length < minimum || args.length > parameters.length))
              throw new Error(`${path}: ${name} declares ${minimum === parameters.length ? minimum : `${minimum}–${parameters.length}`} positional arguments, but this explicit call supplies ${args.length}`);
          }
        }
        references++;
      }
    }
  }
  return { functions: paths.length, visible_references: references };
}

// A callable folder may contain a namespace as well as a directly callable child.
// Preserve qualified names instead of treating a directory as a function.
function nativeBindings(codebase = {}, prefix = '') {
  return Object.entries(codebase).flatMap(([name, child]) => {
    const spelling = prefix + name;
    return [[spelling, child], ...(child.kind === 'namespace'
      ? nativeBindings(child.codebase, spelling + '.') : [])];
  });
}

// Type-only imports disappear at runtime, but their source paths must still resolve
// in the exact package an author compiles. A moved helper cannot retain a stale path.
function checkRelativeModuleImports(files) {
  for (const [path, text] of Object.entries(files).filter(([path]) => path.endsWith('.ts'))) {
    const source = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    for (const statement of source.statements) {
      if (!ts.isImportDeclaration(statement) && !ts.isExportDeclaration(statement)) continue;
      const specifier = statement.moduleSpecifier;
      if (!specifier || !ts.isStringLiteral(specifier) || !specifier.text.startsWith('.')) continue;
      const target = posix.normalize(posix.join(posix.dirname(path), specifier.text));
      const candidates = [target, target.replace(/\.js$/, '.ts'), `${target}.ts`, `${target}.nl`, `${target}/index.ts`];
      if (!candidates.some(candidate => Object.hasOwn(files, candidate)))
        throw new Error(`${path}: relative import ${specifier.text} is absent from the packaged source; repair the moved helper's import or source inventory`);
    }
  }
}

// The loader exposes a default function as its module name and named exports as
// module.member. A function mentioned in prose is not automatically imported.
function checkModuleReferences(definition, modules, path) {
  const escape = name => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const checked = new Set();
  function inspect(spelling, spec) {
    if (checked.has(spelling)) return;
    checked.add(spelling);
    const pattern = new RegExp(`(?<![\\w$.])${escape(spelling)}\\s*\\(`, 'g');
    for (const match of definition.body.matchAll(pattern)) {
      if (!spec || spec.kind !== 'function')
        throw new Error(`${path}: TypeScript helper ${spelling} is not a visible callable; use its companion-folder default function or visible module.export name`);
      const parameters = Object.keys(spec.args ?? {}), minimum = parameters.filter(name => !name.endsWith('?')).length;
      const args = literalCallArguments(definition.body, match.index + match[0].length);
      if (args && !args.some(arg => arg.includes('...')) && (args.length < minimum || args.length > parameters.length))
        throw new Error(`${path}: ${spelling} declares ${minimum === parameters.length ? minimum : `${minimum}–${parameters.length}`} positional arguments, but this explicit call supplies ${args.length}`);
    }
  }
  for (const module of modules) {
    const visible = definition.codebase?.[module.name];
    if (visible?.kind !== 'natlang') inspect(module.name, visible?.kind === 'module' ? visible.exports.default : null);
    for (const [name, spec] of Object.entries(module.exports)) {
      if (name === 'default' || spec.kind !== 'function') continue;
      inspect(`${module.name}.${name}`, visible?.kind === 'module' ? visible.exports[name] : null);
      const direct = definition.codebase?.[name];
      if (direct?.kind !== 'natlang') inspect(name, direct?.kind === 'module' ? direct.exports.default : null);
    }
  }
}

// Native metadata can normalize an inline argument map before exposing params.
// Inspect its top-level keys first so a duplicate cannot silently replace a type.
function declaredInlineNames(source) {
  const header = source.split(/^---\s*$/m)[1] ?? '';
  const clause = /^args\s*:\s*\{(.*)\}\s*$/m.exec(header)?.[1];
  if (clause === undefined) return null;
  const fields = []; let depth = 0, quote = null, escape = false, start = 0;
  for (let i = 0; i <= clause.length; i++) {
    const c = clause[i];
    if (quote) {
      if (escape) escape = false;
      else if (c === '\\') escape = true;
      else if (c === quote) quote = null;
      continue;
    }
    if (c === '"' || c === "'") { quote = c; continue; }
    if (c && '([{<'.includes(c)) depth++;
    else if (c && ')]}>'.includes(c) && depth > 0) depth--;
    if (i === clause.length || (depth === 0 && (c === ',' || c === ';'))) {
      const match = /^\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z_$][\w$]*\??))\s*:/.exec(clause.slice(start, i));
      if (match) fields.push((match[1] ?? match[2] ?? match[3]).replace(/\?$/, ''));
      start = i + 1;
    }
  }
  return fields;
}

// Only inspect explicitly parenthesized calls. Natural-language references remain potential graph evidence.
function literalCallArguments(body, start) {
  const values = []; let depth = 0, quote = null, escaped = false, from = start;
  for (let i = start; i < body.length; i++) {
    const char = body[i];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === quote) quote = null;
      continue;
    }
    if (char === '"' || char === "'" || char === '`') { quote = char; continue; }
    if ('([{'.includes(char)) depth++;
    else if (')]}' .includes(char)) {
      if (char === ')' && depth === 0) {
        const last = body.slice(from, i).trim(); if (last || values.length) values.push(last);
        return values;
      }
      if (--depth < 0) return null;
    } else if (char === ',' && depth === 0) { values.push(body.slice(from, i).trim()); from = i + 1; }
  }
  return null;
}

/** Check source interfaces without imposing a shared domain API, workflow, or node count. */
export function checkApplicationInterfaces(files, manifest, metadata) {
  if (!manifest || manifest.schema !== 'natlang.application/1' ||
      String(manifest.id).padStart(3, '0') !== String(metadata.application_id).padStart(3, '0') ||
      manifest.title !== metadata.title)
    throw new Error('application interface identity differs from its original brief');
  safeRelative(manifest.entrypoint);
  if (!manifest.entrypoint.endsWith('.nl') || !files[manifest.entrypoint])
    throw new Error('application entrypoint must be an existing native function');
  if (!Array.isArray(manifest.interfaces) || !manifest.interfaces.length)
    throw new Error('application must declare its own callable interfaces');
  const names = new Set();
  const interfaces = manifest.interfaces.map(item => {
    if (!item || typeof item.name !== 'string' || !/^[A-Za-z][A-Za-z0-9_-]*$/.test(item.name) ||
        names.has(item.name) || typeof item.purpose !== 'string' || !item.purpose.trim())
      throw new Error('application interfaces require distinct names and an explicit purpose');
    names.add(item.name);
    safeRelative(item.path);
    if (!item.path.endsWith('.nl') || !files[item.path]) throw new Error(`missing native interface ${item.path}`);
    const definition = programDefinition({ semantics: { root: item.path, files } });
    if (item.stateParameter !== undefined && !definition.params.some(param => param.name === item.stateParameter))
      throw new Error(`${item.name}: stateParameter is not a declared native parameter`);
    if (['__proto__', 'constructor', 'prototype'].includes(item.stateParameter))
      throw new Error(`${item.name}: stateParameter must be a safe property name`);
    for (const key of ['stateInputPath', 'stateResultPath', 'effectsPath']) if (item[key] !== undefined &&
        (!Array.isArray(item[key]) || item[key].some(part => typeof part !== 'string' || !part ||
          ['__proto__', 'constructor', 'prototype'].includes(part))))
      throw new Error(`${item.name}: ${key} must be an explicit safe property path`);
    const node = programNode({ semantics: { root: item.path, files } });
    const env = new TypeEnv(node.types);
    const typeAtPath = (type, path, input = false) => {
      if (!path.length) return type;
      const resolved = env.resolve(type);
      if (resolved.kind === 'union') return { kind: 'union', members: resolved.members.map(member => typeAtPath(member, path, input)) };
      const field = resolved.kind === 'record' && resolved.fields.find(field => field.name === path[0]);
      if (!field || (!input && field.optional)) throw new Error(`${item.name}: ${input ? 'state input' : 'result'} path ${path.join('.')} must exist in its declared type`);
      return typeAtPath(field.type, path.slice(1), input);
    };
    if (item.stateResultPath !== undefined || item.stateParameter !== undefined) {
      const state = typeAtPath(node.type.returns, item.stateResultPath ?? []);
      const parameter = node.type.params.fields.find(field => field.name === item.stateParameter);
      const inputType = parameter && typeAtPath(parameter.type, item.stateInputPath ?? [], true);
      if (inputType && !fitsType(state, inputType, env))
        throw new Error(`${item.name}: saved result state does not fit stateParameter ${item.stateParameter}`);
    }
    if (item.stateInputPath !== undefined && item.stateParameter === undefined)
      throw new Error(`${item.name}: stateInputPath requires a stateParameter`);
    if (item.effectsPath !== undefined) typeAtPath(node.type.returns, item.effectsPath);
    if (item.stateKey !== undefined && (typeof item.stateKey !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,100}$/.test(item.stateKey)))
      throw new Error(`${item.name}: stateKey must be an explicit safe record identifier`);
    return { name: item.name, path: item.path, parameters: definition.params.map(param => param.name),
      returns: formatType(node.type.returns) };
  });
  if (!manifest.interfaces.some(item => item.path === manifest.entrypoint))
    throw new Error('the application entrypoint must be exposed by its own interface');
  return { entrypoint: manifest.entrypoint, interfaces,
    evidence: 'native source contracts; no observed execution or semantic quality claim' };
}

/** Potential calls are evidence of source structure, never a substitute for captured execution. */
export function inspectPotentialCalls(files, entrypoints) {
  const paths = Object.keys(files).filter(path => path.endsWith('.nl'));
  const edges = [];
  for (const path of paths) {
    const definition = programDefinition({ semantics: { root: path, files } });
    for (const [name, child] of nativeBindings(definition.codebase)) {
      // Native loader identity covers both companion-folder children and explicit package `uses:` edges.
      const target = child.source;
      if (typeof target !== 'string' || !paths.includes(target)) continue;
      const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      // NatLang may say "using interpretWitness" or name several calls in one sentence.
      // A visible callable's name in instructions is a potential edge, not proof that it was executed.
      if (new RegExp(`\\b${escaped}\\b`, 'i').test(definition.body))
        edges.push({ caller: path, callee: target });
    }
  }
  const reached = new Set(entrypoints), queue = [...entrypoints];
  for (let i = 0; i < queue.length; i++) for (const edge of edges.filter(edge => edge.caller === queue[i]))
    if (!reached.has(edge.callee)) { reached.add(edge.callee); queue.push(edge.callee); }
  return { evidence: 'statically referenced visible callable names; not observed model calls',
    entrypoints, edges, unreferenced: paths.filter(path => !reached.has(path)) };
}

/** Ownership paths bind every original requirement to actual source, without prescribing its decomposition. */
export function checkSourceCoverage(files, manifest, metadata) {
  const coverage = manifest.coverage;
  if (!coverage || !Array.isArray(coverage.domainGates) || !Array.isArray(coverage.operationalRules))
    throw new Error('application must identify the source owners of its original domain gates and operational rules');
  const owners = (rows, key, expected) => {
    if (rows.length !== expected.length || new Set(rows.map(row => row[key])).size !== expected.length ||
        expected.some(value => !rows.some(row => row[key] === value)))
      throw new Error(`source coverage must retain every original ${key}`);
    for (const row of rows) {
      if (!Array.isArray(row.paths) || !row.paths.length || new Set(row.paths).size !== row.paths.length)
        throw new Error(`${key} ${row[key]} requires explicit distinct source owner paths`);
      for (const path of row.paths) {
        safeRelative(path);
        if (!path.endsWith('.nl') || !files[path]) throw new Error(`${key} ${row[key]} names a missing NatLang owner: ${path}`);
      }
    }
  };
  owners(coverage.domainGates, 'stage', metadata.stages.map(stage => stage.name));
  owners(coverage.operationalRules, 'id', metadata.operational_details.map((_, index) => index + 1));
  return { domain_gates: coverage.domainGates.length, operational_rules: coverage.operationalRules.length,
    evidence: 'explicit source ownership; semantic correctness requires independent execution review' };
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
    if (parameter.optional && !Object.hasOwn(fixture.inputs, parameter.name)) continue;
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
  const workflows = fixtures.filter(fixture => oracleLevel(fixture.oracle) === 'judged');
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
  const group = `design-briefs:application-${applicationId}`;
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
    id: `design-briefs-v3:${applicationId}:${fixture.id}`,
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
  const interfaceBytes = await readFile(await confinedFile(appRoot, 'application.json'));
  const manifest = JSON.parse(interfaceBytes);
  if (!Array.isArray(manifest.source_files)) throw new Error(`${name}: missing explicit native source inventory`);
  const interfaceCheck = checkApplicationInterfaces(files, manifest, app);
  const sourceCoverage = checkSourceCoverage(files, manifest, app);
  const contracts = domainContracts ?? JSON.parse(await readFile(join(repo, FIXTURE_ROOT, 'domain-contracts.json'), 'utf8'));
  if (!contracts.applications[id]) throw new Error(`${name}: missing original domain contracts`);
  const contractCheck = checkDomainContracts(files, contracts.applications[id], manifest.entrypoint);
  const callCheck = checkNativeCalls(files, manifest.entrypoint);
  await confinedFile(appRoot, 'DECOMPOSITION.md');
  const fixturePath = await confinedFile(join(repo, FIXTURE_ROOT, name), 'cases.json');
  const fixtureBytes = await readFile(fixturePath);
  const cases = JSON.parse(fixtureBytes);
  if (!Array.isArray(cases) || cases.length !== 8 || new Set(cases.map(item => item.id)).size !== cases.length)
    throw new Error(`${name}: three counterfactual gate cases and five operational workflows are required`);
  const operational = operationalContracts ?? JSON.parse(await readFile(join(repo, FIXTURE_ROOT, 'operational-contracts.json'), 'utf8'));
  const operationalCoverage = checkOperationalCoverage(app, cases, operational.applications[id]);
  const records = cases.map(fixture => {
    const row = makeRecord(app, files, fixture, sha256(fixtureBytes));
    row.source_ids.push(`${SOURCE_ROOT}/${name}/application.json`);
    row.source_revisions.push(`sha256:${sha256(interfaceBytes)}`);
    row.generation.interface_manifest_sha256 = sha256(interfaceBytes);
    return row;
  });
  const gateCases = records.filter(row => oracleLevel(row.semantics.oracle) === 'exact');
  const workflows = records.filter(row => oracleLevel(row.semantics.oracle) === 'judged');
  if (gateCases.length !== 3 || workflows.length !== 5)
    throw new Error(`${name}: missing gate or full-workflow coverage`);
  for (const verdict of ['pass', 'fail', 'needs-review'])
    if (!gateCases.some(row => row.semantics.expected === verdict))
      throw new Error(`${name}: no exact ${verdict} counterfactual`);
  for (const row of gateCases.filter(item => item.semantics.expected === 'pass')) {
    const stage = app.stages[0];
    const required = contracts.applications[id][stage?.output_type];
    const draft = row.semantics.inputs.draft;
    const values = draft?.values ?? draft?.value ?? draft;
    if (!required || !values || Object.keys(required).some(field => !Object.hasOwn(values, field)))
      throw new Error(`${name}: a positive domain gate fixture must provide all fields of its output record`);
  }
  const visibleGateInputs = gateCases.map(row => JSON.stringify(row.semantics.inputs));
  if (new Set(visibleGateInputs).size !== gateCases.length)
    throw new Error(`${name}: identical visible gate inputs have multiple answer records`);
  const potentialCalls = inspectPotentialCalls(files, [...new Set([
    ...manifest.interfaces.map(item => item.path), ...cases.map(item => item.root),
  ])]);
  return { name, app, files, manifest, records, interfaces: interfaceCheck,
    interface_sha256: sha256(interfaceBytes), potential_calls: potentialCalls,
    metadata_sha256: sha256(metadataBytes), fixture_sha256: sha256(fixtureBytes),
    source_sha256: sourceDigest(files), native_functions: nlFiles.length, domain_contracts: contractCheck,
    callable_scope: callCheck, operational_coverage: operationalCoverage, source_coverage: sourceCoverage };
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
