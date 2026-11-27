// natlang ops, the agent program's side of the page (`tendril agent run` opens it through `openPage`; README "Operate").
// The starter's native page (admission of the owner's browsers, publication control) plus four duties of its own:
//   1. live data: serveDataFeeds answers the page's feed pulls (the exporter snapshot, pushed on change; on-demand training
//      data) by asking the Python data service (data_service.py, one child process, JSON lines);
//   2. edits: applies each retained `ops/edit` to the errata / exclusion / annotation layers (data_service `edits.apply`) and
//      records the outcome with `recordApplication` (operation id derived from the edit, so a restart never duplicates);
//   3. tick: one small `ops/tick` signal every 10 s (build stamp for pinned tabs, the supervising agent's inbox heartbeat);
//   4. telemetry: what each connected page sent, received and drew (`ops/t`), logged here and kept in <dashboard>/page-telemetry.jsonl.
// The inbox (questions from the page) is NOT here: it is its own worker, inbox-worker.mjs, with its own lifetime and presence.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFile, readFile, stat } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openNativePage } from '../../shared/native-page.mjs';
import { serveDataFeeds } from '../../shared/helpers/data-feed.mjs';
import { definition, imports, topics } from './domain.mjs';
import { FEEDS, TOPICS, pack } from './feeds.mjs';

export { definition, imports, topics };
const PY = process.env.NATLANG_OPS_PYTHON || '/home/werg/natlang/.venv-neuralese/bin/python';
const SERVICE = process.env.NATLANG_OPS_SERVICE || fileURLToPath(new URL('../../../data_service.py', import.meta.url));   // the override is for tests
const DIST = new URL('./dist/', import.meta.url);
/** The data queries the page may pull (data_catalog.py, data_items.py); test/feeds.test.mjs checks the list against the service. */
export const DATA_QUERIES = Object.freeze(['data.catalog', 'data.dataset', 'data.registry', 'data.search', 'data.items', 'data.item', 'data.chain', 'data.tokens', 'data.task_refs']);

/** The Python data service as a JSON-lines RPC child. Restarted on exit. */
export function dataService({ signal, log = () => {}, python = PY, script = SERVICE, args = ['--serve'] }) {
  let child, n = 0;
  const waiting = new Map();
  const start = () => {
    child = spawn(python, ['-u', script, ...args], { stdio: ['pipe', 'pipe', 'inherit'],
      env: { ...process.env, PYTHONPATH: process.env.PYTHONPATH || '/home/werg/natlang/training/neuralese:/home/werg/natlang', OMP_NUM_THREADS: '1' } });
    const rl = createInterface({ input: child.stdout });
    rl.on('line', line => {
      let m; try { m = JSON.parse(line); } catch { return; }
      const w = waiting.get(m.id); if (!w) return;
      waiting.delete(m.id); clearTimeout(w.timer);
      m.ok ? w.resolve(m.value) : w.reject(Object.assign(Error(m.error || 'data service error'), { code: m.code }));
    });
    child.on('exit', code => {
      for (const w of waiting.values()) { clearTimeout(w.timer); w.reject(Error(`data service exited (${code})`)); }
      waiting.clear();
      if (!signal.aborted) { log({ data_service: 'restart', code }); setTimeout(start, 2000); }
    });
  };
  start();
  signal.addEventListener('abort', () => child?.kill('SIGTERM'), { once: true });
  return (q, input, timeoutMs = 28000) => new Promise((resolve, reject) => {
    const id = ++n;
    const timer = setTimeout(() => { waiting.delete(id); reject(Error(`${q} took longer than ${timeoutMs / 1000} s`)); }, timeoutMs);
    waiting.set(id, { resolve, reject, timer });
    child.stdin.write(JSON.stringify({ id, q, input }) + '\n');
  });
}

/** The feeds the page reads: one pushed (the snapshot) and one on-demand feed per data query. */
export function feedSpecs(rpc) {
  const specs = { [FEEDS.snapshot]: { pollMs: 10_000, maxAgeMs: 5_000, ttlMs: 30_000, fetch: async () => pack(await rpc('ops.snapshot', null)) } };
  for (const q of DATA_QUERIES) specs[q] = { maxAgeMs: 0, fetch: async query => pack(await rpc(q, query ?? {})) };
  return specs;
}

/** The recordApplication operation id of one edit: bounded however long the edit's own operation id is. */
export const applyId = edit => 'apply-' + createHash('sha256').update(edit.operationId).digest('hex').slice(0, 40);

/** Apply every edit that has no application record by this agent yet. */
export async function reconcileEdits(session, rpc, { log = () => {} } = {}) {
  const { value } = await session.query('edits', null);
  const mine = session.identity.publicKey;
  const done = new Set(value.applications.filter(a => a.author === mine).map(a => a.edit));
  for (const edit of value.edits) {
    if (done.has(edit.operationId)) continue;
    const operationId = applyId(edit);
    if (await session.operations.get(operationId)) continue;                    // already submitted (maybe not yet applied)
    let result;
    try { result = await rpc('edits.apply', edit); }
    catch (e) { result = { status: 'failed', detail: e.message.slice(0, 1900), files: '' }; }
    const op = await session.command('recordApplication', { edit: edit.operationId, status: result.status,
      detail: String(result.detail || '').slice(0, 2000), files: String(result.files || '').slice(0, 2000) }, { operationId });
    const applied = await op.materialized({ signal: AbortSignal.timeout(30000) }).catch(e => ({ outcome: e.message }));
    log({ edit: edit.operationId.slice(-24), action: edit.action, target: edit.target, status: result.status, outcome: applied.outcome });
  }
}

async function pageBuild() { try { return JSON.parse(await readFile(new URL('build.json', DIST), 'utf8')).build; } catch { return null; } }
async function heartbeatMs(file) { if (!file) return null; try { return Math.round((await stat(file)).mtimeMs); } catch { return null; } }

async function ticks(session, { signal, heartbeat }) {
  while (!signal.aborted) {
    if ((session.status?.connectedPeers ?? 0) > 0) {
      await session.signals.send(TOPICS.tick, { ts: Date.now(), page_build: await pageBuild(), listener: await heartbeatMs(heartbeat) },
        { ttlMs: 30_000, signal }).catch(() => {});
    }
    await new Promise(r => { const t = setTimeout(r, 10_000); signal.addEventListener('abort', () => { clearTimeout(t); r(); }, { once: true }); });
  }
}

async function serveTelemetry(session, { signal, file, log }) {
  for await (const sig of session.signals.subscribe({ topics: [TOPICS.telemetry], mode: 'all', signal })) {
    const rec = { t: new Date().toISOString(), from: sig.authorId?.slice(0, 12), ...(sig.payload || {}) };
    log({ page: { from: rec.from, seq: rec.seq, version: rec.version, build: rec.build?.build, visible: rec.visible, clock_skew_ms: rec.clock_skew_ms, counts: rec.counts, events: (rec.events || []).slice(0, 12) } });
    if (file) await appendFile(file, JSON.stringify(rec) + '\n').catch(() => {});
  }
}

/** `tendril agent run` calls this once the page is initialized; the returned owner is the page's session owner. */
export async function openPage(context) {
  const owner = await openNativePage(context, { definition, imports, topics });
  const { session } = owner, stop = new AbortController();
  const signal = AbortSignal.any([stop.signal, session.signal, context.signal].filter(Boolean));
  const log = obj => context.report(obj);
  const rpc = dataService({ signal, log });
  const paths = await rpc('ops.paths', null, 20000).catch(() => ({}));
  const feeds = serveDataFeeds(session, feedSpecs(rpc), { signal });
  let busy = false, again = false;
  const kick = async () => {
    if (busy) { again = true; return; }
    busy = true;
    try { do { again = false; await reconcileEdits(session, rpc, { log }); } while (again); }
    catch (e) { log({ reconcile_error: String(e.message).slice(0, 300) }); }
    finally { busy = false; }
  };
  const jobs = [
    feeds.done,
    ticks(session, { signal, heartbeat: paths.telemetry ? join(dirname(paths.telemetry), 'inbox.listener') : null }),
    serveTelemetry(session, { signal, file: paths.telemetry, log }),
    (async () => { await kick(); for await (const change of session.changes({ signal })) { void change; await kick(); } })(),
  ];
  for (const job of jobs) job.catch(e => { if (!signal.aborted) log({ duty_failed: String(e?.message ?? e).slice(0, 300) }); });
  const close = owner.close.bind(owner);
  return Object.assign(owner, { async close() { stop.abort(); feeds.close(); await Promise.allSettled(jobs); await close(); } });
}
