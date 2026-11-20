// The natlang ops native participant (run on the DGX as a memory-ledger unit; see ../README.md).
// It owns the application's native device and is the coordinator that applies browser writes. On top of the starter's
// session it runs four duties:
//   1. reads: answers `ops/q` signals by asking the Python data service (data_service.py, one child process, JSON lines);
//   2. ticks: announces fresh exporter state on `ops/tick` every 10 s;
//   3. edits: applies each retained `edit` to the errata / exclusion / annotation layers (data_service `edits.apply`) and
//      records the outcome with `recordApplication` (operation id `apply-<edit operation id>`, so a restart never duplicates);
//   4. inbox: copies questions asked from the page into <data>/ops/dashboard/inbox.jsonl for the supervising agent.
// Browser pairing: enrollment stays open while this runs; every browser is approved once by the owner on tendriller.dev.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { openSession } from './session.mjs';
import { readFile, appendFile } from 'node:fs/promises';
import { privateDirectory } from './local.mjs';
import { startPairingWindow } from './pairing-native.mjs';
import { TOPICS, encodeReply, validId, editRef, MAX_REPLY_CHUNKS, REPLY_TTL_MS } from './ops-protocol.mjs';

const log = obj => console.log(JSON.stringify({ t: new Date().toISOString(), ...obj }));
const PY = process.env.NATLANG_OPS_PYTHON || '/home/werg/natlang/.venv-neuralese/bin/python';
const SERVICE = fileURLToPath(new URL('../data_service.py', import.meta.url));

/** The Python data service as a JSON-lines RPC child. Restarted on exit. */
function dataService({ signal }) {
  let child, rl, n = 0;
  const waiting = new Map();
  const start = () => {
    child = spawn(PY, ['-u', SERVICE, '--serve'], { stdio: ['pipe', 'pipe', 'inherit'],
      env: { ...process.env, PYTHONPATH: '/home/werg/natlang/training/neuralese:/home/werg/natlang', OMP_NUM_THREADS: '1' } });
    rl = createInterface({ input: child.stdout });
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
  return (q, input, timeoutMs = 60000) => new Promise((resolve, reject) => {
    const id = ++n;
    const timer = setTimeout(() => { waiting.delete(id); reject(Error(`${q} took longer than ${timeoutMs / 1000} s`)); }, timeoutMs);
    waiting.set(id, { resolve, reject, timer });
    child.stdin.write(JSON.stringify({ id, q, input }) + '\n');
  });
}

/** Send one signal, waiting out transport backpressure: a relay route stays busy while its previous frame is in flight,
 * and Chronolog refuses (SIGNAL_BACKPRESSURE) rather than queueing, so consecutive reply chunks are paced here. */
async function sendPaced(session, topic, payload, opts) {
  for (let k = 0, wait = 25; ; k++, wait = Math.min(wait * 2, 1000)) {
    try { return await session.signals.send(topic, payload, opts); }
    catch (e) { if (e.code !== 'SIGNAL_BACKPRESSURE' && !/BACKPRESSURE/.test(e.message) || k >= 40) throw e; await new Promise(r => setTimeout(r, wait)); }
  }
}

async function serveQueries(session, rpc, signal) {
  const inflight = { n: 0 };
  for await (const sig of session.signals.subscribe({ topics: [TOPICS.query], mode: 'all', signal })) {
    const { id, q, input } = sig.payload || {};
    if (!validId(id) || typeof q !== 'string' || q.length > 64) continue;
    if (q.startsWith('edits.') || q.startsWith('inbox.')) continue;          // native-only operations
    if (inflight.n >= 6) { void sendPaced(session, TOPICS.reply, { id, error: 'The data service is busy; retry in a moment.', code: 'busy' }, { ttlMs: REPLY_TTL_MS }).catch(() => {}); continue; }
    inflight.n++;
    (async () => {
      const t0 = Date.now();
      try {
        const parts = await encodeReply(await rpc(q, input ?? null));
        if (parts.length > MAX_REPLY_CHUNKS) throw Object.assign(Error(`Answer too large (${parts.length} chunks); ask for a smaller page.`), { code: 'too_large' });
        for (let i = 0; i < parts.length; i++) await sendPaced(session, TOPICS.reply, { id, i, n: parts.length, z: parts[i] }, { ttlMs: REPLY_TTL_MS });
        log({ query: q, ms: Date.now() - t0, chunks: parts.length, from: sig.authorId?.slice(0, 12) });
      } catch (e) {
        log({ query: q, error: e.message });
        await sendPaced(session, TOPICS.reply, { id, error: e.message.slice(0, 2000), code: e.code || 'error' }, { ttlMs: REPLY_TTL_MS }).catch(() => {});
      } finally { inflight.n--; }
    })();
  }
}

// The build of the page as last built for publication: ticks announce it so pinned tabs move to the selected release.
const PAGE_DIR = process.env.NATLANG_OPS_BUILD ? pathToFileURL(process.env.NATLANG_OPS_BUILD.replace(/\/?$/, '/')) : new URL('../page/', privateDirectory);
async function pageBuild() { try { return JSON.parse(await readFile(new URL('build.json', PAGE_DIR), 'utf8')).build; } catch { return null; } }

/** Page telemetry: each connected page reports what it sent, received and drew (`ops/t`); logged here and kept in
 * <dashboard>/page-telemetry.jsonl for the operators. */
async function serveTelemetry(session, rpc, signal) {
  const file = (await rpc('ops.paths', null).catch(() => null))?.telemetry;
  for await (const sig of session.signals.subscribe({ topics: [TOPICS.telemetry], mode: 'all', signal })) {
    const rec = { t: new Date().toISOString(), from: sig.authorId?.slice(0, 12), ...(sig.payload || {}) };
    log({ page: { from: rec.from, seq: rec.seq, version: rec.version, build: rec.build?.build, visible: rec.visible, counts: rec.counts, events: (rec.events || []).slice(0, 12) } });
    if (file) await appendFile(file, JSON.stringify(rec) + '\n').catch(() => {});
  }
}

async function ticks(session, rpc, signal) {
  let last = null;          // log a tick failure when it changes (no reader connected is the normal idle state)
  while (!signal.aborted) {
    let err = null;
    try { await sendPaced(session, TOPICS.tick, { ...await rpc('ops.tick', null, 15000), page_build: await pageBuild() }, { ttlMs: REPLY_TTL_MS }); }
    catch (e) { err = e.code || e.message; }
    if (err !== last) { log({ tick: err || 'delivered' }); last = err; }
    await new Promise(r => setTimeout(r, 10000));
  }
}

/** Apply every edit that has no native application record yet; ask/answer pairs go to the inbox file. */
async function reconcile(session, rpc, nativeId) {
  const { value } = await session.query('edits', null);
  const done = new Set(value.applications.filter(a => a.author === nativeId).map(a => a.edit));
  for (const edit of value.edits) {
    const ref = editRef(edit.operationId);
    if (done.has(ref)) continue;
    const opId = `apply-${ref.slice(-120)}`;
    if (await session.operations.get(opId)) continue;                         // already submitted (maybe not yet applied)
    let result;
    try { result = await rpc('edits.apply', edit); }
    catch (e) { result = { status: 'failed', detail: e.message.slice(0, 1900), files: '' }; }
    const op = await session.command('recordApplication', { edit: ref, status: result.status,
      detail: String(result.detail || '').slice(0, 2000), files: String(result.files || '').slice(0, 2000) }, { operationId: opId });
    const applied = await op.applied({ signal: AbortSignal.timeout(30000) }).catch(e => ({ outcome: e.message }));
    log({ edit: edit.operationId, action: edit.action, target: edit.target, status: result.status, outcome: applied.outcome });
  }
  const inbox = await session.query('inbox', null);
  await rpc('inbox.sync', inbox.value);
}

export async function runOpsPeer({ signal }) {
  const owned = await openSession({ signal });
  const { session } = owned, nativeId = session.deviceId;
  const rpc = dataService({ signal });
  log({ running: true, deviceId: nativeId, ownerInstanceId: owned.supervisor.ownerInstanceId });
  session.participation.onStatus(s => log({ connection: s.phase, error: s.error }));
  // Browser enrollment stays open for the participant's lifetime; host notices wake it when a browser asks or the owner
  // approves on tendriller.dev. A transition in flight keeps its durable pause and is reconciled by the next pass.
  let lastPairing = '';
  const pairingWindow = startPairingWindow({ owned, signal, onStatus: status => {
    if (!status?.state || status.state === 'idle') return;
    const k = JSON.stringify(status); if (k !== lastPairing) { lastPairing = k; log({ pairing: status }); } } });
  const pairingLoop = pairingWindow.done.catch(e => log({ pairing_error: e.message }));
  const jobs = [serveQueries(session, rpc, signal), ticks(session, rpc, signal), serveTelemetry(session, rpc, signal)];   // pairingLoop may end on a pause; serving goes on
  let busy = false, again = false;
  const kick = async () => {
    if (busy) { again = true; return; }
    busy = true;
    try { do { again = false; await reconcile(session, rpc, nativeId); } while (again); }
    catch (e) { log({ reconcile_error: e.message }); }
    finally { busy = false; }
  };
  await kick();
  jobs.push((async () => { for await (const change of session.changes({ signal })) { void change; await kick(); } })());
  try { await Promise.race(jobs.map(j => j.catch(e => { if (!signal.aborted) throw e; }))); }
  finally { await pairingLoop.catch(() => {}); await owned.close(); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const stopped = new AbortController();
  process.once('SIGINT', () => stopped.abort()); process.once('SIGTERM', () => stopped.abort());
  try { await runOpsPeer({ signal: stopped.signal }); }
  catch (e) { if (!stopped.signal.aborted) { console.error(`${e.code ?? e.name}: ${e.message}`); process.exitCode = 1; } }
}
