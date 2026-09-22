#!/usr/bin/env node
import { access } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const collector = join(here, '..', 'dist', 'teacher', 'cli.js');
try { await access(collector); }
catch {
  console.error('The Node teacher collector has not been built at ' +
    `${relative(process.cwd(), collector)}. Run npm --workspace @natlang/typescript-host run build:node first.`);
  process.exit(2);
}
const child = spawn(process.execPath, [collector, ...process.argv.slice(2)], {
  cwd: process.cwd(), env: process.env, stdio: 'inherit',
});
child.on('error', error => { console.error(`Unable to start the Node teacher collector: ${error.message}`); process.exitCode = 1; });
child.on('exit', (code, signal) => { if (signal) process.kill(process.pid, signal); else process.exitCode = code ?? 1; });
