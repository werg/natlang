/** Native interface loading and exact type checks. Domain algorithms stay in each application's sources. */
import { readFile, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createNatlangRuntime, loadVirtualNatlang, openAICompatibleModelTurn } from '../../ts-host/dist/index.js';
import { programDefinition, programNode } from '../../ts-host/dist/teacher/program.js';
import { TypeEnv, formatType } from '../../ts-host/dist/native/types.js';
import { coerce, problems } from '../../ts-host/dist/native/values.js';
import { checkApplicationInterfaces, confinedFile, sha256, sourceDigest, sourceFiles } from '../../ts-host/scripts/design-briefs/lib.mjs';

export class ApplicationError extends Error {
  constructor(code, detail = '') { super(code); this.name = 'ApplicationError'; this.code = code; this.detail = detail; }
}
const bad = (code, detail) => { throw new ApplicationError(code, detail); };

export async function loadApplications(root = resolve(import.meta.dirname, '../../codebases/design_briefs')) {
  const catalog = new Map();
  for (const entry of (await readdir(root, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory() || !/^\d{3}-/.test(entry.name)) continue;
    const directory = join(root, entry.name);
    const interfaceBytes = await readFile(await confinedFile(directory, 'application.json'));
    const manifest = JSON.parse(interfaceBytes);
    if (!Array.isArray(manifest.source_files)) bad('source_inventory_required');
    const brief = JSON.parse(await readFile(await confinedFile(directory, 'brief.json'), 'utf8'));
    const files = await sourceFiles(directory);
    const checked = checkApplicationInterfaces(files, manifest, brief);
    const id = String(manifest.id).padStart(3, '0');
    if (catalog.has(id)) bad('duplicate_application');
    const interfaces = new Map(manifest.interfaces.map(item => {
      const record = { semantics: { root: item.path, files } };
      const node = programNode(record), definition = programDefinition(record);
      return [item.name, { ...item, definition, node, env: new TypeEnv(node.types) }];
    }));
    catalog.set(id, { id, manifest, files, interfaces, sourceHash: sourceDigest(files), interfaceHash: sha256(interfaceBytes), checked });
  }
  return catalog;
}

export function describeApplication(app) {
  return { id: app.id, title: app.manifest.title, entrypoint: app.manifest.entrypoint,
    sourceHash: app.sourceHash, interfaceHash: app.interfaceHash, persistence: app.manifest.persistence ?? null,
    interfaces: [...app.interfaces.values()].map(item => ({ name: item.name, path: item.path, purpose: item.purpose,
      parameters: item.node.type.params.fields.map(field => ({ name: field.name, type: formatType(field.type), optional: !!field.optional })),
      returns: formatType(item.node.type.returns), types: item.node.typesSrc,
      stateParameter: item.stateParameter ?? null, stateResultPath: item.stateResultPath ?? null })) };
}

export function validateArguments(iface, inputs) {
  if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) bad('invalid_arguments');
  const known = new Set(iface.definition.params.map(p => p.name));
  for (const name of Object.keys(inputs)) if (!known.has(name)) bad('unknown_argument', name);
  for (const field of iface.node.type.params.fields) {
    if (field.optional && !Object.hasOwn(inputs, field.name)) continue;
    try {
      const value = coerce(inputs[field.name], field.type, iface.env, `input.${field.name}`);
      const issues = problems(value, field.type, iface.env, `input.${field.name}`).holes;
      if (issues.length) bad('invalid_arguments', JSON.stringify(issues));
    } catch (e) { bad('invalid_arguments', String(e.message)); }
  }
}

export function validateResult(iface, result) {
  try {
    const value = coerce(result, iface.node.type.returns, iface.env, 'result');
    const issues = problems(value, iface.node.type.returns, iface.env, 'result').holes;
    if (issues.length) bad('invalid_result', JSON.stringify(issues));
  } catch (e) { bad('invalid_result', String(e.message)); }
  // Refuse non-JSON capabilities at the persistence boundary. Native values need their own declared adapter.
  if (JSON.stringify(result) === undefined) bad('unpersistable_result');
}

/** Real runtime path; injectable ModelTurn is used by engineering tests, never as a production answer table. */
export function nativeExecutor({ model, driver, trace, limits } = {}) {
  if (!driver && (!model?.endpoint || !model?.model)) bad('model_configuration_required');
  const runtime = createNatlangRuntime({ model: driver ?? openAICompatibleModelTurn(model),
    network: false, codeEdits: 'deny', calls: false, adaptation: null,
    ...(trace ? { trace } : {}), ...(limits ? { limits } : {}) });
  return async ({ app, iface, inputs, signal }) => {
    validateArguments(iface, inputs);
    const callable = loadVirtualNatlang(app.files, iface.path);
    const result = await runtime.run(() => callable(...iface.definition.params.map(param => inputs[param.name])), { signal });
    validateResult(iface, result);
    return result;
  };
}
