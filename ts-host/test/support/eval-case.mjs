/**
 * Run model-written eval code once as a named natural-language function, in its own process.
 * `node eval-case.mjs '<json>'` with { code, returns?, settleMs?, files? } prints one JSON line:
 * { ok } or { error }, the service ticks counted when the call settled and `settleMs` later, and the elapsed time.
 * Tests run it with a timeout, so code that would hang the process fails its test instead of the suite.
 */
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { createNatlangRuntime, loadNatlang } from '../../dist/index.js';
import { scriptedModel } from './natlang.mjs';

const spec = JSON.parse(process.argv[2]);
const root = mkdtempSync(join(tmpdir(), 'natlang-eval-case-'));
writeFileSync(join(root, 'probe.nl'), `---\nargs: {}\nreturns: ${spec.returns ?? 'string'}\n---\nRun the probe.\n`);
for (const [path, text] of Object.entries(spec.files ?? {})) {
  mkdirSync(dirname(join(root, path)), { recursive: true });
  writeFileSync(join(root, path), text);
}
let ticks = 0;
const counter = { tick: () => { ticks++; return ticks; } };
const model = scriptedModel(() => spec.code);
const events = [];
const runtime = createNatlangRuntime({ model: model.driver, services: { counter },
  serviceDeclarations: { counter: 'export function tick(): number;' },
  trace: trace => events.push(...trace.events.filter(event => typeof event.kind === 'string').map(event => event.kind)) });
const started = Date.now();
let outcome;
try { outcome = { ok: await runtime.run(() => loadNatlang(join(root, 'probe.nl'))()) }; }
catch (error) { outcome = { error: String(error?.message ?? error) }; }
const elapsedMs = Date.now() - started, ticksAtReturn = ticks;
await new Promise(resolve => setTimeout(resolve, spec.settleMs ?? 0));
process.stdout.write(JSON.stringify({ ...outcome, ticksAtReturn, ticksAfter: ticks, elapsedMs }) + '\n');
process.exit(0);
