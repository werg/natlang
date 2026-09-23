#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const output = resolve(process.argv[2] ?? 'data/teacher/read-before-code-probe.ir.jsonl');
const rows = [];

function add(family, index, root, inputs, expected, extra = {}) {
  const id = `read-before-code:${family}-${String(index).padStart(2, '0')}`;
  rows.push({ version: 'natlang.program/1', id, kind: 'lambda_source',
    family: `read_before_code_${family}`, source: 'natlang-read-before-code-probe', split: 'test',
    source_ids: [id], source_groups: [id], source_revisions: ['natlang.read_before_code_probe/1'],
    license: 'project-generated', gold_sources: ['reviewed-policy-and-oracle'],
    semantics: { root: { $lambda: root }, inputs, expected, operation: 'read_before_code', ...extra } });
}

const numericPolicies = [
  { memo: 'Only the even measurements are eligible. Square each eligible value, retaining source order.',
    run: values => values.filter(n => n % 2 === 0).map(n => n * n) },
  { memo: 'Ignore negative measurements. Report the remaining values from largest to smallest, with duplicates retained.',
    run: values => values.filter(n => n >= 0).sort((a, b) => b - a) },
  { memo: 'Use each distinct positive measurement once, in the order of its first appearance, then double it.',
    run: values => [...new Set(values.filter(n => n > 0))].map(n => n * 2) },
  { memo: 'Keep measurements strictly above the median of this batch. Return their original zero-based positions.',
    run: values => { const sorted = [...values].sort((a, b) => a - b), median = sorted[Math.floor(sorted.length / 2)];
      return values.flatMap((n, i) => n > median ? [i] : []); } },
  { memo: 'A zero resets the running total. Report the running total after each nonzero entry; omit the reset entries.',
    run: values => { let sum = 0; return values.flatMap(n => { if (n === 0) { sum = 0; return []; } sum += n; return [sum]; }); } },
  { memo: 'Treat adjacent readings as a transition. Report the positive increases only, in encounter order.',
    run: values => values.slice(1).flatMap((n, i) => n > values[i] ? [n - values[i]] : []) },
];
const numericBatches = [
  [[-3, 0, 2, 5, 2, 8, -1], [4, -2, 4, 7, 0, 3, 6]],
  [[7, -1, 3, 3, 0, 9, -5], [2, 8, -4, 8, 1, 0, 5]],
  [[3, 1, 3, -2, 5, 1, 8], [0, 6, 6, -1, 2, 4, 2]],
  [[1, 9, 3, 7, 5, 2, 11], [8, 2, 6, 4, 10, 1, 7]],
  [[2, 3, 0, 4, -1, 2, 0, 5], [0, 7, -2, 1, 0, 3, 3, -1]],
  [[2, 7, 5, 9, 3, 3, 8], [10, 6, 6, 12, 4, 9, 11]],
];
numericPolicies.forEach((policy, p) => numericBatches[p].forEach((values, variant) => {
  add('input_numeric', p * 2 + variant, {
    type: '(brief: Record<string, string | number[]>) => number[]',
    instructions: 'Inspect brief/memo and brief/values before choosing the computation.\n' +
      'Write TypeScript in eval that implements the memo for these values.\nReturn the resulting number array.',
    function: 'apply_measurement_memo',
  }, { brief: { memo: policy.memo, values } }, policy.run(values),
  { lazy_inputs: ['brief'] });
}));

const textPolicies = [
  { memo: 'Keep only entries that start with an action verb (ship, call, or review), ignoring case. Return their ticket ids in source order.',
    run: lines => lines.flatMap(line => /^(ship|call|review)\b/i.test(line.split('|')[1].trim()) ? [line.split('|')[0]] : []) },
  { memo: 'Select entries marked urgent, but exclude ones explicitly marked resolved. Return their ticket ids in source order.',
    run: lines => lines.flatMap(line => /\burgent\b/i.test(line) && !/\bresolved\b/i.test(line) ? [line.split('|')[0]] : []) },
  { memo: 'Keep the first ticket for each distinct owner, ignoring owner case. Return those ticket ids in first-seen order.',
    run: lines => { const seen = new Set(); return lines.flatMap(line => { const [id, , owner] = line.split('|'), key = owner.trim().toLowerCase();
      if (seen.has(key)) return []; seen.add(key); return [id]; }); } },
  { memo: 'Keep entries whose action says to retry or rerun, but not entries saying not to retry or rerun. Return their ticket ids.',
    run: lines => lines.flatMap(line => /\b(retry|rerun)\b/i.test(line.split('|')[1]) &&
      !/\b(not|never|no)\s+(retry|rerun)\b/i.test(line.split('|')[1]) ? [line.split('|')[0]] : []) },
  { memo: 'Keep tickets with an owner of Ada or Bo, regardless of case. Return ids grouped by owner Ada first, Bo second; preserve source order within each group.',
    run: lines => ['ada', 'bo'].flatMap(owner => lines.flatMap(line => line.split('|')[2].trim().toLowerCase() === owner ? [line.split('|')[0]] : [])) },
  { memo: 'Keep entries whose action contains a question mark and that are not marked resolved. Return their ticket ids.',
    run: lines => lines.flatMap(line => line.split('|')[1].includes('?') && !/\bresolved\b/i.test(line) ? [line.split('|')[0]] : []) },
];
const textBatches = [
  [['T1|Ship replacement|Ada','T2|wait for reply|Bo','T3|CALL vendor|Cy','T4|review traces|Ada'],
   ['T5|Review evidence|Bo','T6|archive note|Ada','T7|ship label|Cy','T8|ask owner?|Bo']],
  [['T1|urgent: call owner|Ada','T2|urgent but resolved|Bo','T3|routine review|Cy','T4|URGENT follow-up|Ada'],
   ['T5|resolved urgent alert|Bo','T6|urgent investigation|Cy','T7|routine case|Ada','T8|urgent unresolved|Bo']],
  [['T1|review|Ada','T2|call|Bo','T3|ship|ADA','T4|wait|Cy'],
   ['T5|wait|Bo','T6|review|Cy','T7|call|ada','T8|ship|Dee']],
  [['T1|retry once|Ada','T2|do not retry|Bo','T3|rerun audit|Cy','T4|never rerun|Ada'],
   ['T5|no retry needed|Bo','T6|please retry|Cy','T7|rerun tomorrow|Ada','T8|archive|Dee']],
  [['T1|ship|Bo','T2|call|Cy','T3|review|Ada','T4|wait|bo'],
   ['T5|wait|ada','T6|review|Dee','T7|call|BO','T8|ship|Ada']],
  [['T1|why now?|Ada','T2|resolved: why now?|Bo','T3|review logs|Cy','T4|which host?|Ada'],
   ['T5|resolved question?|Bo','T6|what changed?|Cy','T7|archive|Ada','T8|who owns this?|Bo']],
];
textPolicies.forEach((policy, p) => textBatches[p].forEach((lines, variant) => {
  add('input_text', p * 2 + variant, {
    type: '(brief: Record<string, string | string[]>) => string[]',
    instructions: 'Inspect brief/memo and brief/entries before deciding how to process the entries.\n' +
      'Write the selected processing code in eval. Each entry is id|action|owner.\nReturn the selected ticket ids.',
    function: 'apply_ticket_memo',
  }, { brief: { memo: policy.memo, entries: lines } }, policy.run(lines),
  { lazy_inputs: ['brief'] });
}));

const tablePolicies = [
  { memo: 'Include active accounts with balance at least 50. Return their ids in file order.',
    run: records => records.filter(r => r.status === 'active' && r.balance >= 50).map(r => r.id) },
  { memo: 'Include accounts in the west region that are not suspended. Return ids in file order.',
    run: records => records.filter(r => r.region === 'west' && r.status !== 'suspended').map(r => r.id) },
  { memo: 'For each owner, keep only the account with the highest balance; break ties by earlier file order. Return selected ids in owner first-seen order.',
    run: records => { const owners = [...new Set(records.map(r => r.owner))]; return owners.map(owner =>
      records.filter(r => r.owner === owner).reduce((best, r) => r.balance > best.balance ? r : best).id); } },
  { memo: 'Include accounts whose balance is below zero and whose status is not closed. Return ids in file order.',
    run: records => records.filter(r => r.balance < 0 && r.status !== 'closed').map(r => r.id) },
  { memo: 'Include accounts opened in 2025 or later with active status. Sort by balance descending, with file order breaking ties. Return ids.',
    run: records => records.map((r, i) => ({ ...r, i })).filter(r => r.opened >= 2025 && r.status === 'active')
      .sort((a, b) => b.balance - a.balance || a.i - b.i).map(r => r.id) },
  { memo: 'Keep one account per region: the first active account in that region. Return ids in region alphabetical order.',
    run: records => [...new Set(records.map(r => r.region))].sort().flatMap(region =>
      records.find(r => r.region === region && r.status === 'active')?.id ?? []).flat() },
];
const tableDatasets = [
  [{id:'A1',owner:'Ada',region:'west',status:'active',balance:70,opened:2025},{id:'A2',owner:'Bo',region:'east',status:'closed',balance:-8,opened:2024},{id:'A3',owner:'Ada',region:'north',status:'active',balance:50,opened:2026},{id:'A4',owner:'Cy',region:'west',status:'suspended',balance:90,opened:2023}],
  [{id:'B1',owner:'Cy',region:'south',status:'active',balance:40,opened:2024},{id:'B2',owner:'Ada',region:'west',status:'active',balance:120,opened:2026},{id:'B3',owner:'Bo',region:'east',status:'active',balance:-5,opened:2025},{id:'B4',owner:'Cy',region:'south',status:'closed',balance:55,opened:2025}],
];
tablePolicies.forEach((policy, p) => tableDatasets.forEach((records, variant) => {
  add('file_table', p * 2 + variant, {
    type: '() => string[]', subtype: 'directory-reducer',
    instructions: 'Read policy.md and accounts.json in the folder before deciding what code to write.\n' +
      'Use eval to implement the policy against the file contents.\nReturn the selected account ids. Leave the files unchanged.',
    function: 'select_accounts',
  }, {}, policy.run(records), { folder_files: { 'policy.md': policy.memo + '\n',
    'accounts.json': JSON.stringify(records, null, 2) + '\n' } });
}));

const logPolicies = [
  { memo: 'Report ids of failed requests from the api service, excluding failures explicitly marked as expected.',
    run: events => events.filter(e => e.service === 'api' && e.level === 'fail' && e.note !== 'expected').map(e => e.id) },
  { memo: 'Report ids of security warnings that occurred after minute 10; minute 10 itself does not count.',
    run: events => events.filter(e => e.service === 'security' && e.level === 'warn' && e.minute > 10).map(e => e.id) },
  { memo: 'For each service with any failure, report the id of its first failure. Preserve the order in which services first fail.',
    run: events => { const seen = new Set(); return events.filter(e => { if (e.level !== 'fail' || seen.has(e.service)) return false;
      seen.add(e.service); return true; }).map(e => e.id); } },
  { memo: 'Report ids of failures that were not followed by a success from the same service later in this file.',
    run: events => events.filter((e, i) => e.level === 'fail' && !events.slice(i + 1).some(later =>
      later.service === e.service && later.level === 'ok')).map(e => e.id) },
  { memo: 'Report ids of warnings from a service that also has at least one failure anywhere in the file.',
    run: events => { const failing = new Set(events.filter(e => e.level === 'fail').map(e => e.service));
      return events.filter(e => e.level === 'warn' && failing.has(e.service)).map(e => e.id); } },
  { memo: 'Report ids of non-expected failures at minute 10 or later, ordered by minute descending and file order for ties.',
    run: events => events.map((e, i) => ({ ...e, i })).filter(e => e.level === 'fail' && e.note !== 'expected' && e.minute >= 10)
      .sort((a, b) => b.minute - a.minute || a.i - b.i).map(e => e.id) },
];
const logDatasets = [
  [{id:'L1',minute:3,service:'api',level:'fail',note:'unexpected'}, {id:'L2',minute:8,service:'security',level:'warn',note:'scan'},
   {id:'L3',minute:10,service:'api',level:'ok',note:'recovered'}, {id:'L4',minute:12,service:'security',level:'warn',note:'token'},
   {id:'L5',minute:13,service:'worker',level:'fail',note:'expected'}, {id:'L6',minute:15,service:'api',level:'fail',note:'unexpected'}],
  [{id:'M1',minute:2,service:'worker',level:'warn',note:'queue'}, {id:'M2',minute:10,service:'security',level:'warn',note:'login'},
   {id:'M3',minute:11,service:'worker',level:'fail',note:'unexpected'}, {id:'M4',minute:14,service:'api',level:'fail',note:'expected'},
   {id:'M5',minute:18,service:'api',level:'fail',note:'unexpected'}, {id:'M6',minute:19,service:'worker',level:'ok',note:'recovered'}],
];
logPolicies.forEach((policy, p) => logDatasets.forEach((events, variant) => {
  add('file_log', p * 2 + variant, {
    type: '() => string[]', subtype: 'directory-reducer',
    instructions: 'Read criteria.md and events.json in the folder before choosing the analysis code.\n' +
      'Use eval to implement the criteria against the events.\nReturn the selected event ids. Leave the files unchanged.',
    function: 'analyze_log',
  }, {}, policy.run(events), { folder_files: { 'criteria.md': policy.memo + '\n',
    'events.json': JSON.stringify(events, null, 2) + '\n' } });
}));

await mkdir(dirname(output), { recursive: true });
await writeFile(output, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
const examples = [0, 5, 12, 17, 24, 29, 36, 41].map(index => {
  const row = rows[index], s = row.semantics;
  const input = s.folder_files ? Object.entries(s.folder_files).map(([path, content]) =>
    `**${path}**\n\n\`\`\`text\n${content.trimEnd()}\n\`\`\``).join('\n\n') :
    `\`\`\`json\n${JSON.stringify(s.inputs, null, 2)}\n\`\`\``;
  return `## ${row.id}\n\n${s.root.$lambda.instructions}\n\n${input}\n\n` +
    `**Expected return**\n\n\`\`\`json\n${JSON.stringify(s.expected)}\n\`\`\`\n`;
});
const sheet = '# Read-before-code probe: sample problems\n\n' +
  'These eight cases illustrate the 48-case [IR corpus](../data/teacher/read-before-code-probe.ir.jsonl). ' +
  'The agent sees the function instructions and typed scope. Lazy input values must be inspected with `read_value` or `eval`; ' +
  'folder contents must be read through file tools or `fs` in `eval`. Expected returns are oracle data, not shown to the agent.\n\n' +
  examples.join('\n');
await writeFile(resolve('docs/read-before-code-examples.md'), sheet);
console.log(`${rows.length} cases across ${new Set(rows.map(row => row.family)).size} families -> ${output}`);
