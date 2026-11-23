/**
 * A scripted model for applications/migration's natural-language stages: each stage answers with the algorithm its
 * instructions spell out, for a rename of `from` to `to` (identifier text). Shared by tests that drive the migration loop
 * through another caller (codemods). The repository-migration test keeps its own copy with per-stage overrides.
 */
import { runInNewContext } from 'node:vm';
import { scriptedModel, withJudge } from './natlang.mjs';

const AsyncFunction = (async () => {}).constructor;

export const STAGES = [
  ['Restate request for the later stages', 'understand'],
  ['Find the sites of intent at revision', 'survey'], ['Locate the site of hit at revision', 'locate'],
  ['Classify the use of intent.old', 'classify'], ['Plan the migration of intent over classified', 'plan'],
  ['Edit the site of classified at revision', 'edit'], ['Write the patches for classified.site', 'patch'],
  ['Check patches, written for classified.site', 'exactjudge'], ['Find the causes of the failure', 'triage'],
  ['Repair the candidate at snapshot.revision', 'repair'], ['Run one round on state', 'round'],
  ['Decide whether the repair loop stops', 'settled'], ['Write the closing words of the migration', 'summarize'],
];

export function renameStages(from, to) {
  const F = JSON.stringify(from), T = JSON.stringify(to);
  return {
    understand: `return { summary: 'rename ' + ${F} + ' to ' + ${T}, old: ${F}, new: ${T}, queries: [...new Set([...seeds, ${F}])], invariants: [] };`,
    survey: `const seen = new Set(); const hits = [];
for (const q of intent.queries) for (const h of repository.search(q, revision).hits) { const k = h.path + ':' + h.offset; if (!seen.has(k)) { seen.add(k); hits.push(h); } }
hits.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : a.line - b.line);
const firsts = hits.filter((h, i) => i === 0 || hits[i - 1].path !== h.path || hits[i - 1].line !== h.line);
const sites = await Promise.all(firsts.map(h => locate(intent, h, revision)));
return sites.map((s, i) => ({ ...s, id: 's' + (i + 1) }));`,
    locate: `return { id: '', path: hit.path, from: hit.line, to: hit.line, text: repository.lines(hit.path, hit.line, hit.line, revision), hits: 1 };`,
    classify: `return { pattern: 'call', action: 'edit', reason: 'uses ' + intent.old + ' at ' + site.path + ':' + site.from };`,
    plan: `return { edits: classified.filter(c => c.usage.action === 'edit').map(c => c.site.id), leave: [], risks: [] };`,
    edit: `let patches = await patch(intent, classified, revision, problem);
let verdict = await exact(patches, classified, revision);
if (!verdict.exact) { patches = await patch(intent, classified, revision, verdict.problem); verdict = await exact(patches, classified, revision); }
return verdict.exact ? { site: classified.site.id, usage: classified.usage, patches, status: 'patched', note: classified.usage.reason }
  : { site: classified.site.id, usage: classified.usage, patches: [], status: 'failed', note: verdict.problem };`,
    patch: `const text = classified.site.text, path = classified.site.path;
return text.includes(intent.old) && repository.count(path, text, revision) === 1 ? [{ path, old: text, new: text.split(intent.old).join(intent.new) }] : [];`,
    exactjudge: `if (!patches.length) return { exact: false, problem: 'no patch for the site' };
for (const p of patches) { const found = repository.count(p.path, p.old, revision); if (found !== 1) return { exact: false, problem: 'old text occurs ' + found + ' times in ' + p.path }; }
return { exact: true, problem: '' };`,
    triage: `return (validation?.checks ?? []).filter(c => c.status === 'failed').map(c => ({ check: c.id, kind: 'missed-site', path: c.id, line: 1, evidence: c.output, repairable: false }));`,
    repair: `return [];`,
    round: `const findings = await triage(intent, state.snapshot, state.validation, state.rejected);
return { ...state, findings, remaining: state.remaining - 1 };`,
    settled: `return state.validation?.status === 'passed' || state.remaining === 0 || state.findings.length > 0;`,
    summarize: `return { summary: report.status + ': ' + report.changed.length + ' files changed', next: [] };`,
  };
}

/** A model driver answering the migration stages from `scripts` (and `other(opening)` for any other call). */
export function migrationModel(scripts, other = () => null) {
  const seen = [];
  const model = scriptedModel(opening => {
    const stage = STAGES.find(([phrase]) => opening.includes(phrase))?.[1];
    if (!stage) return other(opening);
    seen.push(stage);
    return scripts[stage];
  });
  const driver = model.driver;
  withJudge(driver);
  const judge = driver.decide;
  // `settled` is a `readout: decision`: score it by running the stage's script on the state the call shows.
  driver.decide = async request => {
    const text = request.messages.map(message => String(message.content ?? '')).join('\n');
    if (!text.includes('Decide whether the repair loop stops')) return judge(request);
    seen.push('settled');
    const shown = /^state: [^\n]*? = ([^\n]+)$/m.exec(text)?.[1];
    if (!shown) throw new Error(`settled: no state shown in ${text.slice(0, 400)}`);
    const state = runInNewContext('(' + shown + ')', Object.create(null), { timeout: 100 });
    const stop = Boolean(await new AsyncFunction('state', scripts.settled)(state));
    return { log_probs: request.options.map(option => Math.log(JSON.parse(option) === stop ? 0.97 : 0.03)) };
  };
  return { driver, seen, openings: model.openings };
}
