// The inbox worker (unit ops-inbox): the always-on side of the page's questions to the supervising agent.
// It attaches to this page's owner inside the running agent program (`tendril agent run`, never a second signer) and, from each
// coherent read of the agent-request feed (derived afresh on every change, never accumulated):
//   - gives every question a `seen` receipt (and `done` once the supervising agent has answered it);
//   - mirrors the questions with their answers to <dashboard>/inbox.jsonl, which `inbox.sh` reads for the supervising agent;
//   - announces its presence (announceAgentPresence, bound to this process) so the page can say whether questions are taken.
// A question is a chat message whose `data` is {kind, subject} (ask / request_stop / ack_alert; domain.mjs); the answer is the
// supervising agent's reply message (`inbox.sh answer`). This worker never answers: that is the supervising agent's work, and
// nothing here starts a conversation turn. It exits nonzero when the owner goes away (the agent program restarted); the unit's
// supervisor (units.sh) starts it again.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { announceAgentPresence } from '../../shared/helpers/agent-requests.mjs';
import { createCliLifetime } from '../../shared/cli-lifetime.mjs';
import { ASK_KINDS } from './domain.mjs';

export const CONSUMER_ID = 'ops-inbox';
const HERE = fileURLToPath(new URL('.', import.meta.url));
const digest = (...parts) => createHash('sha256').update(JSON.stringify(parts)).digest('hex').slice(0, 40);

/** The questions of one agent feed (`readAgentFeed`) with the receipts and answers around them. */
export function questionsOf(feed, self) {
  const replies = new Map();
  for (const m of feed.messages) if (m.author === self && m.inReplyTo) replies.set(`${m.inReplyTo.author}:${m.inReplyTo.operationId}`, [...(replies.get(`${m.inReplyTo.author}:${m.inReplyTo.operationId}`) || []), m]);
  const out = [];
  for (const m of feed.messages) {
    if (m.author === self || m.inReplyTo) continue;
    let d = {}; try { d = JSON.parse(m.data || '{}'); } catch { /* a plain chat message */ }
    if (!ASK_KINDS.includes(d.kind) || !d.subject?.kind || !d.subject?.id) continue;
    const key = `${m.author}:${m.operationId}`;
    out.push({ op: m.operationId, kind: d.kind, subject: d.subject, text: m.text, author: m.author, at_ms: m.atMs,
      answers: (replies.get(key) || []).map(r => ({ text: r.text, author: r.author, op: r.operationId, at_ms: r.atMs })),
      receipts: (feed.receipts?.[key] || []).filter(r => r.author === self).map(r => r.state) });
  }
  return out;
}

async function writeIfChanged(file, text) {
  try { if (await readFile(file, 'utf8') === text) return false; } catch { /* first write */ }
  await mkdir(dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  await writeFile(tmp, text); await rename(tmp, file);
  return true;
}

/** One pass: receipts for what lacks them, then the mirror file. Idempotent: it only reads derived state. */
export async function pass(session, { file, log = () => {} }) {
  const self = session.identity.publicKey;
  const feed = (await session.query('agentFeed', null)).value;
  const questions = questionsOf(feed, self);
  for (const q of questions) {
    const want = q.kind === 'ack_alert' || q.answers.length ? 'done' : 'seen';
    if (q.receipts.includes(want)) continue;
    const operationId = 'ops-inbox-' + want + '-' + digest(q.author, q.op);
    if (!await session.operations.get(operationId)) {
      const op = await session.command('agentReceipt', { messageAuthor: q.author, messageOperationId: q.op, runId: 'ops-inbox', state: want }, { operationId });
      const st = await op.materialized({ signal: AbortSignal.timeout(30000) }).catch(e => ({ outcome: e.message }));
      log({ receipt: want, ask: q.op.slice(-12), kind: q.kind, outcome: st.outcome });
    }
  }
  const text = questions.map(({ receipts, ...q }) => JSON.stringify(q) + '\n').join('');
  const changed = await writeIfChanged(file, text);
  if (changed) log({ inbox: questions.length, open: questions.filter(q => !q.answers.length && q.kind !== 'ack_alert').length });
  return questions;
}

/** <data dir>/ops/dashboard, from the dashboard's config.json (the same file data_service.py reads). */
export function dashboardDirectory(config = JSON.parse(readFileSync(new URL('../../../config.json', import.meta.url), 'utf8'))) {
  return resolve(config.data_dir.replace(/^~/, process.env.HOME), 'ops/dashboard');
}

export async function main({ signal, workspaceRoot = process.env.NATLANG_OPS_WORKSPACE || resolve(HERE, '../..'), slug = 'ops',
  file = resolve(process.env.NATLANG_OPS_DASH || dashboardDirectory(), 'inbox.jsonl'), log = o => console.log(JSON.stringify({ t: new Date().toISOString(), ...o })) } = {}) {
  const { openWorkspace, attachPage } = await import('@tendriller/tools/workspace');
  const workspace = await openWorkspace({ root: workspaceRoot });
  const attached = await attachPage(workspace, slug, { signal });
  const { session } = attached;
  const stop = new AbortController(), lifetime = AbortSignal.any([stop.signal, signal, session.signal].filter(Boolean));
  let presence;
  try {
    log({ running: true, page: slug, file });
    await pass(session, { file, log });
    presence = announceAgentPresence(session, { consumerId: CONSUMER_ID, role: 'inbox', topics: ['chat'], handle: workspace.config.agent?.handle ?? '', capabilities: ['chat'], signal: lifetime });
    let again = false, busy = false;
    const kick = async () => {
      if (busy) { again = true; return; }
      busy = true;
      try { do { again = false; await pass(session, { file, log }); } while (again); }
      catch (e) { log({ pass_error: String(e.message).slice(0, 300) }); }
      finally { busy = false; }
    };
    for await (const change of session.changes({ signal: lifetime })) { void change; await kick(); }
    if (!lifetime.aborted) throw Object.assign(Error('the agent program closed this page'), { code: 'OWNER_LOST' });
  } finally {
    stop.abort();
    try { await presence?.close(); } catch { /* best effort: the card ages to stale */ }
    await attached.detach().catch(() => {});
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const lifetime = createCliLifetime();
  try { lifetime.opened(); await main({ signal: lifetime.signal }); lifetime.closed(); }
  catch (error) { lifetime.closed(); lifetime.fail(error); }
  process.exit(await lifetime.finish());
}
