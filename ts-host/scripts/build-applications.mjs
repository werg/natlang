#!/usr/bin/env node
/** Build the repository applications against this checkout's runtime (`applications/dist`). */
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildProject, formatDiagnostics } from '../dist/index.js';

// An application with its own dependencies (package.json and lockfile beside its sources) needs them installed: the
// applications build as one project, so one missing install fails every application.
const applications = fileURLToPath(new URL('../../applications', import.meta.url));
const missing = readdirSync(applications, { withFileTypes: true })
  .filter(entry => entry.isDirectory() && existsSync(join(applications, entry.name, 'package-lock.json')) &&
    !existsSync(join(applications, entry.name, 'node_modules')))
  .map(entry => join('applications', entry.name));
if (missing.length) {
  console.error(`These applications' dependencies are not installed: ${missing.join(', ')}. Install them from their ` +
    `lockfiles first: ${missing.map(dir => `npm --prefix ${dir} ci`).join('; ')} (scripts/setup_dev.sh does this).`);
  process.exit(1);
}

const runtime = new URL('../dist/index.js', import.meta.url);
const result = buildProject({ project: fileURLToPath(new URL('../../applications', import.meta.url)),
  runtimeModule: { url: runtime.href, path: fileURLToPath(runtime),
    types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] } });
if (!result.ok) { console.error(formatDiagnostics(result.diagnostics)); process.exit(1); }
console.log(`built ${Object.keys(result.outputs).length} application files`);
