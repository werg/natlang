#!/usr/bin/env node
// Oracle-only adversarial probe of a task packet's edit_stream and contract_diagnosis rows (oracle-adversaries.mjs):
// every negative must be rejected and every positive accepted by the row's own oracle. Runs the oracle and the
// collector's execution path with scripted actions only (zero model calls); prints counts and failing probe names,
// never case content. Exit 1 on any accepted negative or rejected positive.
// node scripts/inline-curriculum/probe-packet-oracles.mjs CASES.jsonl
import { readFileSync } from 'node:fs';
import { checkConstraints } from '../../dist/evaluation/constraints.js';
import { replayReference } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { contractDiagnosisNegatives, contractDiagnosisPositives, editStreamNegatives, editStreamPositives } from './oracle-adversaries.mjs';

const records = readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
const summary = { edit_stream: { rows: 0, negatives: 0, negatives_rejected: 0, positives: 0, positives_accepted: 0 },
  contract_diagnosis: { rows: 0, negatives: 0, negatives_rejected: 0, positives: 0, positives_accepted: 0 } };
const failures = [];
for (const record of records) {
  const family = record.curriculum.family, s = summary[family];
  if (!s) continue;
  s.rows++;
  if (family === 'edit_stream') {
    for (const [name, text] of editStreamNegatives(record)) {
      s.negatives++;
      if (!checkConstraints(text, record.semantics.expected).passed) s.negatives_rejected++; else failures.push({ id: record.id, accepted_negative: name });
    }
    for (const [name, text] of editStreamPositives(record)) {
      s.positives++;
      if (checkConstraints(text, record.semantics.expected).passed) s.positives_accepted++; else failures.push({ id: record.id, rejected_positive: name });
    }
  } else {
    const run = async root => {
      const probe = structuredClone(record);
      probe.curriculum.reference.root = root;
      return (await replayReference(probe, TOOLS_PROMPT)).run.outcome;
    };
    for (const [name, root] of contractDiagnosisNegatives(record)) {
      s.negatives++;
      if (!(await run(root)).accepted) s.negatives_rejected++; else failures.push({ id: record.id, accepted_negative: name });
    }
    for (const [name, root] of contractDiagnosisPositives(record)) {
      s.positives++;
      if ((await run(root)).accepted) s.positives_accepted++; else failures.push({ id: record.id, rejected_positive: name });
    }
  }
}
console.log(JSON.stringify({ cases: records.length, by_family: summary, failures }, null, 1));
process.exit(failures.length ? 1 : 0);
