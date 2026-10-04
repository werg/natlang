#!/usr/bin/env node
/** Convert existing improvement runs to `natlang.improvement-step/1` records (LEARNING_CONTINUUM.md §8).
 *
 * Usage: convert-improvement-steps.mjs --out STEPS.jsonl INPUT...
 *
 * Inputs, recognised by content:
 * - crisp authoring results (`result.json`, `natlang.skill-authoring-trajectory/1`): one step per result, operator
 *   `crisp-skill-search` (the S2 author with search), before/after the baseline and selected source digests, query and
 *   transfer gains from the sealed paired evaluations, compute from the author and executor exchanges;
 * - soft-skill decision results (`results.jsonl` of soft-skill-decision.mjs): per family, one step per trained arm
 *   (`generic`, `tuned`, `specific`) from the text-initialised skill. Runs before direct writing did not keep the
 *   tuned blocks' IDs, so those records are marked legacy with null IDs.
 *
 * Authoring runs that failed before a baseline existed are skipped (counted in the summary).
 * The output is created, never overwritten. Every record validates; records repeat across inputs only by content ID,
 * and duplicates are written once.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { authoringStep, improvementStep } from '../../dist/improvement/step-record.js';

const args = process.argv.slice(2), outIndex = args.indexOf('--out');
if (outIndex < 0 || !args[outIndex + 1]) throw Error('Usage: convert-improvement-steps.mjs --out STEPS.jsonl INPUT...');
const out = args[outIndex + 1], inputs = args.filter((_, i) => i !== outIndex && i !== outIndex + 1);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const records = new Map();
let skipped = 0;
const keep = record => records.set(record.id, record);

function fromAuthoring(result, file, hash) {
  const step = authoringStep(result, { converter: 'convert-improvement-steps/1', file, sha256: hash });
  if (step) keep(step); else skipped++;
}

function fromSoftSkillDecision(rows, file, hash) {
  const generic = rows.find(row => row.arm === 'generic');
  for (const row of rows.filter(item => item.family && item.scores)) {
    // Gains from each arm's own start: the specific arm starts from the generic skill, the others from text-init.
    const gain = (arm, where) => {
      const before = row.scores[arm === 'specific' ? 'generic' : 'text-init']?.[where]?.quality, after = row.scores[arm]?.[where]?.quality;
      return before === undefined || after === undefined ? null : { before, after, effect: after - before };
    };
    for (const arm of ['generic', 'tuned', 'specific']) {
      if (!row.scores[arm]) continue;
      // The text-initialised starting block was never recorded, so these conversions are legacy whatever the arm.
      const id = arm === 'generic' ? generic?.block ?? null : null;
      const legacy = true;
      keep(improvementStep({
        episode: { id: `soft-skill-decision:${row.family}`, family: row.family },
        facets: [`family:${row.family}`, `operator:soft-skill-${arm}`, 'artifact:soft-skill'],
        before: [{ kind: 'soft-skill', id: null, role: 'skill' }],
        operator: { kind: `soft-skill-${arm}`, version: 'soft-skill-decision/3', regime: arm === 'specific' ? 'conditioned-distillation' : 'supervised',
          hyper: { optimizer: 'adam', steps: (arm === 'generic' ? generic?.trace : row.traces?.[arm])?.length ?? null,
            ...(arm === 'specific' ? { hold: 'cross-entropy to the generic readout off-family' } : {}) }, context: null, model: null },
        view: { visibility: 'full', evidence: [] },
        proposal: { deltas: [{ artifact: { kind: 'soft-skill', id, role: 'skill' }, delta: null, scale: 1 }] },
        after: [{ kind: 'soft-skill', id, role: 'skill' }],
        outcome: { query: gain(arm, 'query'), transfer: gain(arm, 'transfer'),
          compute: { gradient_steps: (arm === 'generic' ? generic?.trace : row.traces?.[arm])?.length ?? undefined } },
        trajectory: { id: `soft-skill-decision:${row.family}:${arm}`, step: 0 },
        provenance: { converter: 'convert-improvement-steps/1', file, sha256: hash, legacy },
      }));
    }
  }
}

for (const file of inputs) {
  const bytes = await readFile(file), hash = sha(bytes), text = bytes.toString();
  if (file.endsWith('.json')) {
    const value = JSON.parse(text);
    if (value.version === 'natlang.skill-authoring-trajectory/1') fromAuthoring(value, file, hash);
    else throw Error(`${file}: not a recognised improvement result`);
  } else {
    const rows = text.split('\n').filter(line => line.trim()).map(JSON.parse);
    if (rows.some(row => row.arm === 'generic') && rows.some(row => row.family && row.scores)) fromSoftSkillDecision(rows, file, hash);
    else throw Error(`${file}: not a recognised improvement result`);
  }
}
await writeFile(out, [...records.values()].map(record => JSON.stringify(record)).join('\n') + (records.size ? '\n' : ''), { flag: 'wx' });
console.log(JSON.stringify({ steps: records.size, skipped, inputs: inputs.length, out }));
