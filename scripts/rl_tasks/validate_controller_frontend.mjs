import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const input = JSON.parse(readFileSync(0, 'utf8'));
if (!input || typeof input !== 'object' || !input.files || !input.root || !input.controller)
  throw new Error('expected package files, root function, and collector controller declaration');
const repoDist = resolve(dirname(fileURLToPath(import.meta.url)), '../../ts-host/dist');
const tsHostDist = input.runtime_dist ? resolve(input.runtime_dist) : repoDist;
const { nodePlatform } = await import(pathToFileURL(resolve(tsHostDist, 'runtime/node.js')).href);
const { loadVirtualNatlang } = await import(pathToFileURL(resolve(tsHostDist, 'runtime/virtual-project.js')).href);
const mounted = loadVirtualNatlang(input.files, input.root, '/project', nodePlatform());
const controller = input.controller;
const module = mounted[controller.module];
if (!module || typeof module !== 'object' || typeof module[controller.export] !== 'function')
  throw new Error('Natlang loader did not attach the declared controller export');
if (!mounted[controller.operations] || !['object', 'function'].includes(typeof mounted[controller.operations]))
  throw new Error('Natlang loader did not attach the declared operation subtree');
const tsHost = tsHostDist;
const frontendFiles = ['runtime/loader.js', 'runtime/virtual-project.js', 'compiler/project.js'];
const digest = createHash('sha256');
for (const path of frontendFiles) {
  digest.update(path).update('\0').update(readFileSync(resolve(tsHost, path))).update('\0');
}
process.stdout.write(JSON.stringify({
  schema: 'natlang.application-controller-frontend-check/1',
  root: input.root,
  module: controller.module,
  export: controller.export,
  operations: controller.operations,
  loaded_by: 'natlang.runtime.loadVirtualNatlang',
  runtime_dist: tsHostDist,
  node_version: process.version,
  frontend_files: Object.fromEntries(frontendFiles.map(path => [path,
    createHash('sha256').update(readFileSync(resolve(tsHost, path))).digest('hex')])),
  frontend_sha256: digest.digest('hex'),
  accepted: true,
}));
