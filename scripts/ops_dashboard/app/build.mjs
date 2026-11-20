// Build the publishable page graph: the dashboard UI (ui/), the owning session Worker and the pairing Worker, bundled with
// esbuild, plus public configuration. Output: NATLANG_OPS_BUILD or <private dir>/../page (outside the repository).
// Publish with `npx tendril --session ./session.mjs --mode own publish <out> --operation ops-release-N` (README).
import { build } from 'esbuild';
import { copyFile, mkdir, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { project, readPrivate, privateDirectory } from './local.mjs';

const root = fileURLToPath(new URL('.', import.meta.url));
const output = new URL(process.env.NATLANG_OPS_BUILD ? `file://${process.env.NATLANG_OPS_BUILD.replace(/\/?$/, '/')}` : '../page/', privateDirectory);
const { handle } = await project();
const descriptor = await readPrivate('descriptor.json');
const initialized = await readPrivate('initialized.json');
// This dedicated output is the entire publishable graph. Never publish the private directory: it holds the handle,
// signer and bootstrap key.
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
const publicConfiguration = { applicationId: handle.application.id, page_url: handle.page_url,
  serviceOrigin: new URL(handle.api).origin, pageId: handle.id, nativeDeviceId: initialized.deviceId, descriptor };
await writeFile(new URL('configuration.json', output), JSON.stringify(publicConfiguration, null, 2) + '\n');
for (const f of ['index.html', 'style.css']) await copyFile(new URL(`./ui/${f}`, import.meta.url), new URL(f, output));
// Build stamp: shown in the page footer and announced by the participant's ticks, so an open tab that is still on an
// older publication (Tendriller pins a tab, reload included) can move itself to the selected release.
const { execSync } = await import('node:child_process');
const stamp = { build: new Date().toISOString().replace(/\.\d+Z$/, 'Z'), commit: execSync('git rev-parse --short=10 HEAD', { cwd: root }).toString().trim() };
await writeFile(new URL('build.json', output), JSON.stringify(stamp) + '\n');
await build({ define: { __OPS_BUILD__: JSON.stringify(stamp) }, absWorkingDir: root, entryPoints: { page: 'ui/page.mjs', worker: 'worker.mjs', 'pairing-worker': 'pairing-worker.mjs' },
  outdir: fileURLToPath(output), bundle: true, splitting: true, format: 'esm', platform: 'browser',
  target: 'es2022', outExtension: { '.js': '.mjs' }, chunkNames: 'chunks/[name]-[hash]', logLevel: 'warning' });
console.log(JSON.stringify({ built: fileURLToPath(output) }));
