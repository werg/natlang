#!/usr/bin/env node
/**
 * One-time upgrade of program IR files from natlang.program/1 (a `$lambda` root with definition-style
 * children) to natlang.program/2 (a project: root `.nl` path and files). Scripted references written
 * for the retired action surface (`operations`, `source_lines`, `leaf_oracles`) are dropped.
 *   node scripts/migrate-program-ir.mjs FILE.jsonl [...]
 */
import { readFile, rename, writeFile } from 'node:fs/promises';
import { PROGRAM_VERSION, definitionProject, lambdaSignature, programDefinition } from '../dist/teacher/program.js';

/** Legacy type spellings, in the current TypeScript-style syntax. */
function modernType(text) {
  let type = String(text).replace(/\bBool\b/g, 'boolean').replace(/\bNum\b/g, 'number').replace(/\bText\b/g, 'string')
    .replace(/\bDict<([^<>]+)>/g, 'Record<string, $1>');
  const lambda = /^Lambda<\s*\{([\s\S]*)\}\s*,\s*([\s\S]+)>$/.exec(type.trim());
  if (lambda) type = `(${lambda[1].trim()}) => ${lambda[2].trim()}`;
  return type;
}
const modernTypes = types => types && Object.fromEntries(Object.entries(types).map(([name, text]) => [name, modernType(text)]));
function modernSpec(spec) {
  const out = { ...spec, returns: modernType(spec.returns),
    args: Object.fromEntries(Object.entries(spec.args ?? {}).map(([name, type]) => [name, modernType(type)])) };
  if (spec.code !== undefined) out.code = spec.code.replace(/\bargs\.(\w+)/g, '$1');
  if (spec.types) out.types = modernTypes(spec.types);
  if (spec.codebase) out.codebase = Object.fromEntries(Object.entries(spec.codebase).map(([name, child]) => [name, modernSpec(child)]));
  delete out.effects; delete out.engine;
  return out;
}

export function upgradeRecord(record) {
  if (record.version === PROGRAM_VERSION) return record;
  if (record.version !== 'natlang.program/1') throw new Error(`${record.id}: unsupported version ${record.version}`);
  const lambda = record.semantics?.root?.$lambda;
  if (!lambda) throw new Error(`${record.id}: only lambda roots can be upgraded`);
  const name = lambda.function ?? 'main';
  const project = definitionProject(name, { ...lambdaSignature(modernType(lambda.type)), instructions: lambda.instructions,
    ...(lambda.types ? { types: modernTypes(lambda.types) } : {}), ...(lambda.subtype ? { kind: lambda.subtype } : {}),
    ...(lambda.codebase ? { codebase: Object.fromEntries(Object.entries(lambda.codebase).map(([child, spec]) => [child, modernSpec(spec)])) } : {}) });
  const { root, operations, source_lines, leaf_oracles, ...semantics } = record.semantics;
  const upgraded = { ...record, version: PROGRAM_VERSION, semantics: { ...project, ...semantics } };
  programDefinition(upgraded);
  return upgraded;
}

for (const path of process.argv.slice(2)) {
  const lines = (await readFile(path, 'utf8')).split('\n').filter(line => line.trim());
  const rows = lines.map(line => JSON.stringify(upgradeRecord(JSON.parse(line))));
  await writeFile(`${path}.upgrading`, rows.join('\n') + '\n');
  await rename(`${path}.upgrading`, path);
  console.log(`${rows.length} records -> ${PROGRAM_VERSION}: ${path}`);
}
