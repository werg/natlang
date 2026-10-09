#!/usr/bin/env node
/** Yes/no decision cases (natlang.decision-case/1, kind `noul`) as skill episodes for S7 rollouts.
 *
 * Each episode is one family's question as a natlang function `(state: string) => boolean` (no decision readout: the
 * reply is sampled, so a group of rollouts varies), scored by the graded `binary-brier` metric against the case's
 * label. Support cases come from the `train` role, query cases from `heldout`, from disjoint source groups (the
 * episode gate's rule). These give a weak policy rewards with spread, which RL needs; the harder episode families
 * (SQL, research, CSP) follow as the policy improves (S7 §2.3 curriculum).
 *
 * Usage: build-decision-episodes.mjs --cases decision-cases.jsonl --out EPISODES.jsonl [--families a,b]
 *          [--episodes-per-family 4] [--support 8] [--query 8]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { validateEpisode } from '../../dist/skills/episode.js';

const options = { 'episodes-per-family': '4', support: '8', query: '8' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  if (!['cases', 'out', 'families', 'episodes-per-family', 'support', 'query'].includes(key) || process.argv[i + 1] === undefined)
    throw Error('Usage: see the header of build-decision-episodes.mjs');
  options[key] = process.argv[i + 1];
}
const wanted = options.families ? new Set(options.families.split(',')) : null;
const byFamily = new Map();
for (const line of readFileSync(options.cases, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const c = JSON.parse(line);
  if (c.kind !== 'noul' || (wanted && !wanted.has(c.family))) continue;
  const answer = c.answer === true || c.answer === 'True' ? true : c.answer === false || c.answer === 'False' ? false : Number(c.answer);
  if (typeof answer === 'number' && !(answer >= 0 && answer <= 1)) continue;
  const family = byFamily.get(c.family) ?? byFamily.set(c.family, { train: [], heldout: [] }).get(c.family);
  (c.role === 'heldout' ? family.heldout : family.train).push({ ...c, answer });
}
const sha = value => createHash('sha256').update(value).digest('hex');
const perFamily = Number(options['episodes-per-family']), nSupport = Number(options.support), nQuery = Number(options.query);
const lines = [];
for (const [family, { train, heldout }] of [...byFamily].sort()) {
  const question = (train[0] ?? heldout[0]).question;
  const solve = `---\nargs: { state: string }\nreturns: boolean\n---\n${question} Answer true or false.\n`;
  const asCase = c => ({ id: c.id, group: c.group, args: [c.state], expected: { kind: 'binary', answer: c.answer } });
  for (let e = 0; e < perFamily; e++) {
    const support = train.slice(e * nSupport, (e + 1) * nSupport);
    const supportGroups = new Set(support.map(c => c.group));
    const query = heldout.filter(c => !supportGroups.has(c.group)).slice(e * nQuery, (e + 1) * nQuery);
    if (support.length < nSupport || query.length < nQuery || new Set(support.map(c => c.group)).size < 2) break;
    const episode = { version: 'natlang.skill-episode/1', id: `rl-${family.replace(/[^a-z0-9]+/gi, '-')}-${e}`, family, split: 'train',
      source_groups: [`group-commitment:sha256:${sha([...supportGroups].sort().join(','))}`],
      license: JSON.stringify([...new Set(support.map(c => c.license))]),
      target: { kind: 'improvement-case', entry: 'solve.nl', exportName: 'default',
        source: { schema: 'natlang.rl-decision-target/1', id: family }, files: { 'solve.nl': solve } },
      library: { kind: 'empty', skills: {} }, support: { cases: support.map(asCase) }, query: { cases: query.map(asCase) },
      operations: ['create', 'revise', 'select', 'test'], limits: { maxSteps: 6 },
      provenance: { generator: 'natlang.rl-decision-episodes/1', source: support[0].source,
        metric: { schema: 'natlang.skill-graded/1', kind: 'binary-brier' } } };
    const diagnostics = validateEpisode(episode);
    if (diagnostics.length) throw Error(`${episode.id}: ${JSON.stringify(diagnostics)}`);
    lines.push(JSON.stringify(episode));
  }
}
writeFileSync(options.out, lines.join('\n') + '\n');
console.log(JSON.stringify({ episodes: lines.length, families: byFamily.size }));
