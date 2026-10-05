// Claims desk: SciFact claims (source-episode-pools-20261005 scifact-source-v6, train candidates) checked against their
// cited abstracts. The library holds the claims and an exact abstract lookup; each claim is judged by an inline nl call
// given the claim and its abstracts; code combines the verdicts (a verdict record, the contradicted ids, counts per
// verdict, the first claim the evidence does not back). Gold verdicts from SciFact's official oracle labels.
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const POOL = fileURLToPath(new URL('../../../data/neuralese/corpora/source-episode-pools-20261005/self-improvement-expansion-20261004/scifact-source-v6/candidates.jsonl', import.meta.url));
let claims = null;
function loadClaims() {
  if (claims) return claims;
  if (!existsSync(POOL)) throw new Error('SciFact pool missing; sync source-episode-pools-20261005');
  claims = readFileSync(POOL, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line))
    .filter(row => row.role === 'train')
    .map(row => ({ claim: row.task.claim, label: row.host_only_oracle.label, group: row.provenance.component_id,
      documents: row.task.documents.map(d => ({ id: String(d.doc_id), title: d.title, text: d.abstract_sentences.map(s => s.text).join(' ') })),
      sourceId: row.candidate_id }))
    .filter(c => c.documents.reduce((n, d) => n + d.text.length, 0) < 2400);
  return claims;
}

const VERDICTS = ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'];
const OPS = {
  record: () => ({ text: 'Return a record from claim id to its verdict.', returns: 'Record<string, Verdict>',
    expected: items => Object.fromEntries(items.map(c => [c.id, c.label])),
    code: 'return Object.fromEntries(all.map((claim, i) => [claim.id, verdicts[i]]));' }),
  contradicted: () => ({ text: 'Return the ids of the claims the cited abstracts contradict, in id order.', returns: 'string[]',
    expected: items => items.filter(c => c.label === 'CONTRADICT').map(c => c.id),
    code: 'return all.filter((_, i) => verdicts[i] === "CONTRADICT").map(claim => claim.id);' }),
  counts: () => ({ text: 'Count the claims per verdict, including every verdict with 0 when none has it.', returns: 'Record<string, number>',
    expected: items => Object.fromEntries(VERDICTS.map(v => [v, items.filter(c => c.label === v).length])),
    code: 'const out: Record<string, number> = { SUPPORT: 0, CONTRADICT: 0, NOT_ENOUGH_INFO: 0 };\nfor (const v of verdicts) out[v]++;\nreturn out;' }),
  unsupported: () => ({ text: 'Return the ids of the claims their cited abstracts do not support (contradicted, or not enough information), in id order.', returns: 'string[]',
    expected: items => items.filter(c => c.label !== 'SUPPORT').map(c => c.id),
    code: 'return all.filter((_, i) => verdicts[i] !== "SUPPORT").map(claim => claim.id);' }),
};

export function claimsDesk(seed, index) {
  const rng = new Random(seed, `claims-desk:${index}`);
  const pool = loadClaims();
  for (let attempt = 0; attempt < 40; attempt++) {
    const op = OPS[rng.pick(Object.keys(OPS))]();
    const n = rng.int(3, 5);
    const base = rng.sample(pool, n);
    const swapAt = rng.int(0, n - 1);
    const replacement = rng.pick(pool.filter(c => c.label !== base[swapAt].label && !base.includes(c)));
    const worlds = [base, base.map((c, i) => i === swapAt ? replacement : c)]
      .map(list => list.map((c, i) => ({ ...c, id: `C${i + 1}` })));
    if (JSON.stringify(op.expected(worlds[0])) === JSON.stringify(op.expected(worlds[1]))) continue;
    const shape = `claims${index}`;
    return worlds.map((items, w) => {
      const docs = {};
      for (const c of items) for (const d of c.documents) docs[d.id] = { title: d.title, text: d.text };
      const plain = items.map(c => ({ id: c.id, claim: c.claim, cited: c.documents.map(d => d.id) }));
      const expected = op.expected(items);
      const code = `const all = literature.claims();
const judge = nl<(claim: Claim, abstracts: Abstract[]) => Promise<Verdict>>\`Using only abstracts, does the evidence SUPPORT claim, CONTRADICT it, or give NOT_ENOUGH_INFO?\`;
const verdicts = await Promise.all(all.map(claim => judge(claim, claim.cited.map(id => literature.abstract(id)))));
${op.code}`;
      const record = curriculumCase({ family: 'claims_desk', shape, variant: w ? 'b' : 'a', pairGroup: `claims:${shape}`,
        splitGroup: `claims:${[...new Set(items.map(c => c.group))].sort()[0]}`,
        slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
        evidence: { world: items.map(c => `${c.id}: ${c.label}`), retrieved: [], background: [] },
        minimumSequence: ['read the claims and their cited abstracts', 'judge each claim against its own abstracts in its own nl call',
          'combine the verdicts in code'],
        reference: { root: [evalCall(code), returnCall(expected)],
          children: items.map(c => ({ match: ['SUPPORT claim, CONTRADICT it', JSON.stringify(c.id)], value: c.label })) },
        root: { name: 'claims_desk', args: {}, returns: op.returns,
          instructions: `Check each claim in literature.claims() against the abstracts it cites (literature.abstract(id)): SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO, using only those abstracts. ${op.text}` },
        files: { 'claims_desk/literature.ts': `const CLAIMS = ${literal(plain)};\nconst ABSTRACTS: Record<string, Abstract> = ${literal(Object.fromEntries(Object.entries(docs).map(([id, d]) => [id, { id, ...d }])))};\n` +
          '/** The claims to check, each with the ids of the abstracts it cites. */\nexport function claims(): Claim[] { return CLAIMS; }\n' +
          '/** One cited abstract by id. */\nexport function abstract(id: string): Abstract { return ABSTRACTS[id]; }\n',
          'types.ts': 'export type Claim = { id: string, claim: string, cited: string[] };\nexport type Abstract = { id: string, title: string, text: string };\nexport type Verdict = "SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO";\n' },
        inputs: {}, expected });
      record.license = 'SciFact claims CC-BY-4.0; abstracts ODC-By-1.0';
      record.gold_sources = ['scifact-official-oracle-labels'];
      record.dataset = 'scifact';
      record.dataset_records = items.map(c => c.sourceId);
      return record;
    });
  }
  throw new Error(`no discriminating claims case for ${seed}:${index}`);
}
