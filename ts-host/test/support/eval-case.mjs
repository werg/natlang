/**
 * Run model-written eval code once as a named natural-language function, in its own process.
 * `node eval-case.mjs '<json>'` with { code, timeoutMs?, children?, returns?, settleMs?, files? } prints one JSON line:
 * { ok } or { error }, the service ticks counted when the call settled and `settleMs` later, and the elapsed time.
 * `children` maps text in a child call's instructions to the eval code that answers it (the first match wins).
 * Tests run it with a timeout, so code that would hang the process fails its test instead of the suite.
 */
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { createNatlangRuntime, loadNatlang } from '../../dist/index.js';
import { modelTurnsSoFar } from '../../dist/native/agent.js';

const spec = JSON.parse(process.argv[2]);
const root = mkdtempSync(join(tmpdir(), 'natlang-eval-case-'));
writeFileSync(join(root, 'probe.nl'), `---\nargs: {}\nreturns: ${spec.returns ?? 'string'}\n---\nRun the probe.\n`);
for (const [path, text] of Object.entries(spec.files ?? {})) {
  mkdirSync(dirname(join(root, path)), { recursive: true });
  writeFileSync(join(root, path), text);
}
let ticks = 0;
const counter = { tick: () => { ticks++; return ticks; } };
/** Each call's first turn runs its eval; then it finishes, or fails with the eval's error. */
const driver = async ({ messages }) => {
  const opening = String(messages[1].content);
  const child = Object.entries(spec.children ?? {}).find(([text]) => opening.includes(text));
  if (modelTurnsSoFar(messages) === 0)
    return { calls: [['eval', { code: child ? child[1] : spec.code, ...(!child && spec.timeoutMs ? { timeout_ms: spec.timeoutMs } : {}) }]] };
  const last = String(messages.at(-1).content);
  if (/^(?:rejected|error)|\nerror|Nothing else from this eval was kept/.test(last))
    return { calls: [['return_result', { status: 'failed', reason: `Scripted eval failed: ${last.slice(0, 400)}` }]] };
  return { text: 'done' };
};
/** A host stream: services and packages may hand eval code async iterables, which `for await` consumes. */
const feed = { items: n => (async function* () { for (let i = 1; i <= n; i++) { await new Promise(r => setTimeout(r, 1)); yield i; } })() };
const runtime = createNatlangRuntime({ model: driver, services: { counter, feed },
  serviceDeclarations: { counter: 'export function tick(): number;', feed: 'export function items(n: number): AsyncIterable<number>;' } });
const started = Date.now();
let outcome;
try { outcome = { ok: await runtime.run(() => loadNatlang(join(root, 'probe.nl'))()) }; }
catch (error) { outcome = { error: String(error?.message ?? error) }; }
const elapsedMs = Date.now() - started, ticksAtReturn = ticks;
await new Promise(resolve => setTimeout(resolve, spec.settleMs ?? 0));
process.stdout.write(JSON.stringify({ ...outcome, ticksAtReturn, ticksAfter: ticks, elapsedMs }) + '\n');
process.exit(0);
