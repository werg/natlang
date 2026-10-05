// Dependency-rich named-function graphs. Hidden observations are scoped to leaves;
// gold labels live only in the oracle/reference, never in parent instructions.
import { Random, curriculumCase, evalCall, returnCall, nlFile, literal } from './lib.mjs';

const DOMAINS = {
  triage: {
    rule: 'Classify the report as active if customers are experiencing the problem now, otherwise resolved. A historical incident or a hypothetical risk alone is resolved.',
    labels: ['active', 'resolved'],
    reports: [
      ['Customers still cannot complete checkout although the status page says the incident ended.', 'active'],
      ['The outage yesterday was fixed; customers can now complete checkout normally.', 'resolved'],
      ['We thought the fix worked, but customers say their payments are failing again.', 'active'],
      ['If the old fault returns, payments might fail; nobody currently reports failures.', 'resolved'],
    ],
  },
  evidence: {
    rule: 'Classify the claim as supported if the record establishes that the parcel reached the recipient, otherwise unsupported. Arrival at a depot or a proposed delivery does not establish receipt.',
    labels: ['supported', 'unsupported'],
    reports: [
      ['The recipient signed for the parcel at their front door, after the depot delay.', 'supported'],
      ['The depot signed the incoming manifest; the parcel has not left for the recipient.', 'unsupported'],
      ['The courier scan failed, but the recipient confirmed they took the parcel inside.', 'supported'],
      ['The courier plans to ask for a signature tomorrow; the recipient has received nothing.', 'unsupported'],
    ],
  },
  policy: {
    rule: 'Classify the request as eligible if the cancellation was before dispatch, otherwise ineligible. Payment time and delivery time do not decide eligibility.',
    labels: ['eligible', 'ineligible'],
    reports: [
      ['Payment cleared Monday. Cancellation arrived Tuesday. Dispatch was Wednesday.', 'eligible'],
      ['Payment cleared Monday. Dispatch was Tuesday. Cancellation arrived Wednesday.', 'ineligible'],
      ['The cancellation was accepted in the morning; dispatch happened that evening.', 'eligible'],
      ['Delivery has not happened, but dispatch preceded the cancellation request.', 'ineligible'],
    ],
  },
};

export function recurrenceGraphs(seed, index, _split, { inlineJudgments = false } = {}) {
  const rng = new Random(seed, `recurrence:${index}`);
  const domain = Object.keys(DOMAINS)[index % 3], task = DOMAINS[domain];
  const depth = 1 + Math.floor(index / 3) % 4;
  const width = 1 + Math.floor(index / 12) % 3;
  const padding = [0, 2048, 8192, 32768][Math.floor(index / 36) % 4];
  const observations = Array.from({ length: width }, (_, i) => ({ id: `case_${rng.int(10000,99999)}_${i}`, choice: rng.int(0, 3) }));
  // Siblings have one opening but different hidden observations and gold labels.
  return [0, 1].map(variant => {
    const files = {}, children = [], expected = [], edges = [];
    const directory = level => ['review', ...Array.from({ length: level }, (_, k) => `merge_${k + 1}`)].join('/');
    const leafNames = observations.map((_, i) => `assess_${i}`);
    for (let i = 0; i < width; i++) {
      const observation = observations[i];
      const [report, label] = task.reports[(observation.choice + (i === 0 ? variant : 0)) % 4];
      const value = [{ id: observation.id, label }];
      expected.push(...value);
      const name = leafNames[i];
      // Service implementation is outside the readable tree and callable only by this leaf.
      files[`${directory(depth - 1)}/${name}/record.ts`] = `/** Private observation for this assessor. */\nexport function read(): string { return ${literal(`${observation.id}: ${report}\n${'Background: archive maintenance is routine.\n'.repeat(Math.ceil(padding / 44))}`)}; }\n`;
      files[`${directory(depth - 1)}/${name}.nl`] = nlFile({ args: {}, returns: 'Finding[]', description: `Assess one private ${domain} observation.`,
        instructions: inlineJudgments ? `Read the private observation with record.read(). Use an inline nl<Finding> lambda to judge that observation by this criterion: ${task.rule} Pass the observation, including its case id. Return an array containing its Finding.` : `Read record.read(). ${task.rule} Return one Finding with the case id from the record and its label.` });
      const assessCode = inlineJudgments ? `const observation = record.read(); const findings = [await nl<Finding>\`${task.rule} Return a Finding containing the case id from observation and its label.\`(observation)]; JSON.stringify(findings)` : 'record.read()';
      children.push({ match: `You are inside this call: ${name}(`, calls: [evalCall(assessCode), returnCall(value)] });
      if (inlineJudgments) children.push({ match: ['You are inside this call: nl@eval:', observation.id], value: value[0] });
    }
    let names = leafNames;
    for (let level = depth - 1; level > 0; level--) {
      const name = `merge_${level}`;
      const code = `const findings = [${names.map(n => `await ${n}()`).join(', ')}].flat();\nJSON.stringify(findings)`;
      files[`${directory(level - 1)}/${name}.nl`] = nlFile({ args: {}, returns: 'Finding[]', description: 'Combine independently assessed findings without changing them.',
        instructions: `Call each listed function exactly once: ${names.join(', ')}. Merge their Finding arrays in that order, retaining every id and label exactly. Return the merged array.` });
      children.push({ match: `You are inside this call: ${name}(`, calls: [evalCall(code), returnCall(expected)] });
      edges.push(...names.map(child => [name, child]));
      names = [name];
    }
    const code = `const findings = [${names.map(n => `await ${n}()`).join(', ')}].flat();\nJSON.stringify(findings)`;
    edges.push(...names.map(child => ['review', child]));
    files['types.ts'] = 'export type Finding = { id: string; label: "active" | "resolved" | "supported" | "unsupported" | "eligible" | "ineligible" };\n';
    const shape = `graph${index}`;
    const record = curriculumCase({ family: inlineJudgments ? 'recurrence_inline_graphs' : 'recurrence_graphs', shape, variant: `world${variant}`, pairGroup: `recurrence:${shape}`,
      slice: 'nested_scoped', domain: 'logic', mode: 'followup', named: 'required',
      evidence: { world: observations.map(o => o.id), retrieved: [], background: [] },
      decisive: [{ marker: observations[0].id, source: 'child', note: 'The private case observation determines the finding.' }],
      plausibleActions: ['classify the report positively', 'classify the report negatively'],
      minimumSequence: ['obtain private observations through the named assessors', 'combine the child findings exactly'],
      reference: { root: [evalCall(code), returnCall(expected)], children },
      root: { name: 'review', args: {}, returns: 'Finding[]', instructions: `Call each listed function exactly once: ${names.join(', ')}. Return their Finding arrays merged in that order, without adding duplicate findings or calling unlisted functions. Preserve every returned id and label. Evidence is private to the assessors; do not invent their findings.` },
      files, expected });
    record.generation.recurrence = { version: 1, depth, width, inline_judgments: inlineJudgments, padding_chars_per_leaf: padding, edges,
      controls: ['correct', 'shuffled', 'zero', 'removed', 'counterfactual'],
      admission: 'requires-observed-named-call-edges-and-complete-producer-closure' };
    return record;
  });
}

export const recurrenceInlineGraphs = (seed, index, split) => recurrenceGraphs(seed, index, split, { inlineJudgments: true });
