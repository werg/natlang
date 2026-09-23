#!/usr/bin/env node
/** Build the repository applications against this checkout's runtime (`applications/dist`). */
import { fileURLToPath } from 'node:url';
import { buildProject, formatDiagnostics } from '../dist/index.js';

const runtime = new URL('../dist/index.js', import.meta.url);
const result = buildProject({ project: fileURLToPath(new URL('../../applications', import.meta.url)),
  runtimeModule: { url: runtime.href, path: fileURLToPath(runtime),
    types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] } });
if (!result.ok) { console.error(formatDiagnostics(result.diagnostics)); process.exit(1); }
console.log(`built ${Object.keys(result.outputs).length} application files`);
