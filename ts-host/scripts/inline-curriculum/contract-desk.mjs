// Contract desk: ContractNLI (train split) non-disclosure agreements checked against a checklist of hypotheses. The
// contract comes from the library; one inline nl call per checklist item judges it against the contract it captures
// (Entailment, Contradiction or NotMentioned); code combines the answers into the requested result: a record, the
// contradicted items, the required protections the contract does not grant, or how many items it never mentions.
// Gold choices are the dataset's annotations.
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const DATA = fileURLToPath(new URL('../../../vendor/datasets/contract-nli-20261004/data/train.json', import.meta.url));
let corpus = null;
function load() {
  if (corpus) return corpus;
  if (!existsSync(DATA)) throw new Error('ContractNLI is not in vendor/datasets/contract-nli-20261004');
  const data = JSON.parse(readFileSync(DATA, 'utf8'));
  corpus = { labels: data.labels, documents: data.documents.filter(d => d.text.length <= 12000)
    .map(d => ({ id: String(d.id), text: d.text.replace(/\s+\n/g, '\n').trim(), choices: Object.fromEntries(Object.entries(d.annotation_sets[0].annotations).map(([k, v]) => [k, v.choice])) })) };
  return corpus;
}

const OPS = {
  record: () => ({ text: 'Return a record from checklist item id to the contract\'s answer.', returns: 'Record<string, Answer>',
    expected: (items, choices) => Object.fromEntries(items.map(h => [h.id, choices[h.id]])),
    code: 'return Object.fromEntries(items.map((item, i) => [item.id, answers[i]]));' }),
  contradicted: () => ({ text: 'Return the short descriptions of the checklist items the contract contradicts, in checklist order.', returns: 'string[]',
    expected: (items, choices) => items.filter(h => choices[h.id] === 'Contradiction').map(h => h.short),
    code: 'return items.filter((_, i) => answers[i] === "Contradiction").map(item => item.short);' }),
  missing_required: () => ({ text: 'Our policy requires every item marked required. Return the ids of the required items the contract does not clearly grant (it contradicts them or never mentions them), in checklist order.', returns: 'string[]',
    expected: (items, choices) => items.filter(h => h.required && choices[h.id] !== 'Entailment').map(h => h.id),
    code: 'return items.filter((item, i) => item.required && answers[i] !== "Entailment").map(item => item.id);', requiredOnly: true }),
  not_mentioned: () => ({ text: 'How many checklist items does the contract not mention at all?', returns: 'number',
    expected: (items, choices) => items.filter(h => choices[h.id] === 'NotMentioned').length,
    code: 'return answers.filter(answer => answer === "NotMentioned").length;' }),
};

export function contractDesk(seed, index) {
  const rng = new Random(seed, `contract-desk:${index}`);
  const { labels, documents } = load();
  for (let attempt = 0; attempt < 60; attempt++) {
    const op = OPS[rng.pick(Object.keys(OPS))]();
    const ids = rng.sample(Object.keys(labels), rng.int(4, 6));
    const items = ids.map((id, i) => ({ id, short: labels[id].short_description, text: labels[id].hypothesis, required: rng.next() < 0.6 }));
    if (op.requiredOnly && !items.some(h => h.required)) continue;
    const [first, second] = rng.sample(documents, 2);
    const ea = op.expected(items, first.choices), eb = op.expected(items, second.choices);
    if (JSON.stringify(ea) === JSON.stringify(eb)) continue;
    const shape = `contract${index}`;
    const checklist = items.map(({ id, short, text, required }) => ({ id, short, text, required }));
    // Paired worlds: the same checklist against two contracts.
    return [['a', first], ['b', second]].map(([variant, doc]) => {
      const expected = op.expected(items, doc.choices);
      const code = `const contract = nda.contract();
const items = nda.checklist();
const judge = nl<(item: Item) => Promise<Answer>>\`Against contract, is item's statement an Entailment, a Contradiction, or NotMentioned?\`;
const answers = await Promise.all(items.map(item => judge(item)));
${op.code}`;
      const record = curriculumCase({ family: 'contract_desk', shape, variant, pairGroup: `contract:${shape}`, splitGroup: `contract:${doc.id}`,
        slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
        evidence: { world: items.map(h => `${h.id}: ${doc.choices[h.id]}`), retrieved: [], background: [] },
        minimumSequence: ['read the contract and the checklist', 'judge each checklist item against the contract in its own nl call',
          'combine the answers in code'],
        reference: { root: [evalCall(code), returnCall(expected)],
          children: items.map(h => ({ match: ['an Entailment, a Contradiction', JSON.stringify(h.id)], value: doc.choices[h.id] })) },
        root: { name: 'contract_desk', args: {}, returns: op.returns,
          instructions: `Review the agreement from nda.contract() against each item of nda.checklist(): the contract either entails the item's statement (Entailment), contradicts it (Contradiction), or does not address it (NotMentioned). ${op.text}` },
        files: { 'contract_desk/nda.ts': `const CONTRACT = ${JSON.stringify(doc.text)};\nconst CHECKLIST = ${literal(checklist)};\n` +
          '/** The agreement under review, as text. */\nexport function contract(): string { return CONTRACT; }\n' +
          '/** The review checklist: statements the agreement may grant, contradict or not address. */\nexport function checklist(): Item[] { return CHECKLIST; }\n',
          'types.ts': 'export type Item = { id: string, short: string, text: string, required: boolean };\nexport type Answer = "Entailment" | "Contradiction" | "NotMentioned";\n' },
        inputs: {}, expected });
      record.license = 'ContractNLI CC-BY-4.0';
      record.gold_sources = ['contractnli-annotations'];
      record.dataset = 'contractnli';
      record.dataset_records = [`contractnli:train:${doc.id}`];
      return record;
    });
  }
  throw new Error(`no discriminating contract case for ${seed}:${index}`);
}
