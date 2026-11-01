import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const input = JSON.parse(readFileSync(0, 'utf8'));
if (!input || typeof input !== 'object' || !Array.isArray(input.roots) || !input.roots.length)
  throw new Error('expected one or more root source/input binding groups');
const dist = resolve(input.runtime_dist ?? new URL('../../ts-host/dist', import.meta.url).pathname);
const { prepareProgramNode } = await import(pathToFileURL(resolve(dist, 'teacher/program.js')).href);
const checked = [];
for (const group of input.roots) {
  if (!group || typeof group.root !== 'string' || !group.files || typeof group.files !== 'object' || !Array.isArray(group.worlds))
    throw new Error('invalid root binding group');
  for (const world of group.worlds) {
    if (!world || typeof world.id !== 'string' || !world.inputs || typeof world.inputs !== 'object' || Array.isArray(world.inputs))
      throw new Error('invalid root input binding row');
    const record = { version: 'natlang.program/2', kind: 'lambda_source', id: world.id,
      semantics: { root: group.root, files: group.files, inputs: world.inputs, expected: null } };
    try {
      await prepareProgramNode(record);
    } catch (error) {
      throw new TypeError(`${world.id}: collector root argument binding rejected: ${error instanceof Error ? error.message : String(error)}`);
    }
    checked.push({ id: world.id, root: group.root, input_names: Object.keys(world.inputs).sort(),
      binding: 'teacher.prepareProgramNode/collector-root-argument-binding' });
  }
}
process.stdout.write(JSON.stringify({schema:'natlang.root-input-binding-preflight/1',runtime_dist:dist,
  loaded_by:'teacher.prepareProgramNode',checked}));
