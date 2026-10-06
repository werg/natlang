#!/usr/bin/env node
// Answer clef-programs.mjs skeleton requests with Cloudflare Workers AI Clef-flash, resumably: answers already in OUT
// are skipped; stops at the daily input-token budget (free tier: about 1.22M Clef-flash input tokens a day).
//   node clef-label.mjs SKELETONS.jsonl ANSWERS.jsonl [--budget 1100000] [--concurrency 4]
// Credentials: the wrangler OAuth token (~/.config/.wrangler/config/default.toml) and CLOUDFLARE_ACCOUNT_ID or the
// first account the token sees.
import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';

const [skeletonPath, outPath, ...rest] = process.argv.slice(2);
if (!skeletonPath || !outPath) throw new Error('usage: clef-label.mjs SKELETONS ANSWERS [--budget N] [--concurrency N]');
const flag = (name, fallback) => { const i = rest.indexOf(name); return i < 0 ? fallback : Number(rest[i + 1]); };
const budget = flag('--budget', 1_100_000), concurrency = flag('--concurrency', 4);
// A wrangler OAuth token lasts an hour; on 401/403 `wrangler whoami` refreshes it (an API token does not expire).
const oauthToken = () => readFileSync(`${homedir()}/.config/.wrangler/config/default.toml`, 'utf8').match(/oauth_token = "([^"]+)"/)[1];
const api = 'https://api.cloudflare.com/client/v4';
const headers = { Authorization: `Bearer ${process.env.CLOUDFLARE_API_TOKEN ?? oauthToken()}`, 'Content-Type': 'application/json' };
function refreshToken() {
  if (process.env.CLOUDFLARE_API_TOKEN) return false;
  try { execFileSync('wrangler', ['whoami'], { stdio: 'ignore', timeout: 60_000 }); } catch { /* reported by the retry */ }
  headers.Authorization = `Bearer ${oauthToken()}`;
  return true;
}
const account = process.env.CLOUDFLARE_ACCOUNT_ID ?? (await (await fetch(`${api}/accounts`, { headers })).json()).result[0].id;

const done = new Set(existsSync(outPath) ? readFileSync(outPath, 'utf8').split('\n').filter(Boolean)
  .map(l => JSON.parse(l)).map(a => `${a.seed}:${a.index}:${a.item}`) : []);
const queue = [];
for (const line of readFileSync(skeletonPath, 'utf8').split('\n').filter(Boolean)) {
  const sk = JSON.parse(line);
  for (const r of sk.requests) if (!done.has(`${sk.seed}:${sk.index}:${r.item}`)) queue.push({ seed: sk.seed, index: sk.index, ...r });
}
let used = 0, answered = 0, failed = 0;
async function ask(request) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const response = await fetch(`${api}/accounts/${account}/ai/run/@cf/cloudflare/clef-flash`, { method: 'POST', headers,
      body: JSON.stringify({ model: 'clef-flash', state: request.state, questions: request.questions }) });
    const body = await response.json().catch(() => ({}));
    if (response.ok && body.result?.answers) return body.result;
    if ((response.status === 401 || response.status === 403) && attempt < 3 && refreshToken()) continue;
    if (response.status === 429 || response.status >= 500) { await new Promise(r => setTimeout(r, 2000 * 2 ** attempt)); continue; }
    throw new Error(`clef ${response.status}: ${JSON.stringify(body.errors ?? body).slice(0, 300)}`);
  }
  throw new Error('clef: retries exhausted');
}
async function worker() {
  while (queue.length && used < budget) {
    const request = queue.shift();
    try {
      const result = await ask(request);
      used += result.usage?.input_tokens ?? 0;
      appendFileSync(outPath, JSON.stringify({ seed: request.seed, index: request.index, item: request.item, answers: result.answers,
        usage: result.usage, model: '@cf/cloudflare/clef-flash' }) + '\n');
      answered++;
    } catch (error) { failed++; console.error(String(error)); if (failed > 20) throw error; }
  }
}
await Promise.all(Array.from({ length: concurrency }, worker));
console.log(JSON.stringify({ answered, failed, input_tokens: used, remaining: queue.length, budget }));
