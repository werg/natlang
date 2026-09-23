#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { PROGRAM_VERSION, definitionProject, lambdaSignature } from '../dist/teacher/program.js';

const output = resolve(process.argv[2] ?? 'data/teacher/read-before-code-probe.ir.jsonl');
const rows = [];

function add(family, index, root, inputs, expected, extra = {}) {
  const id = `read-before-code:${family}-${String(index).padStart(2, '0')}`;
  const project = definitionProject(root.function, { ...lambdaSignature(root.type), instructions: root.instructions,
    ...(root.types ? { types: root.types } : {}), ...(root.subtype ? { kind: root.subtype } : {}) });
  rows.push({ version: PROGRAM_VERSION, id, kind: 'lambda_source',
    family: `read_before_code_${family}`, source: 'natlang-read-before-code-probe', split: 'test',
    source_ids: [id], source_groups: [id], source_revisions: ['natlang.read_before_code_probe/1'],
    license: 'project-generated', gold_sources: ['reviewed-policy-and-oracle'],
    semantics: { ...project, inputs, expected, operation: 'read_before_code', ...extra } });
}

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

// Application-shaped batches are large because every account or event is a
// meaningful row. Their normal opening preview shows only the first few rows.
const accountExtras = [
  [{id:'A5',owner:'Bo',region:'west',status:'active',balance:35,opened:2025},
   {id:'A6',owner:'Cy',region:'north',status:'active',balance:-6,opened:2026},
   {id:'A7',owner:'Dee',region:'south',status:'active',balance:50,opened:2024},
   {id:'A8',owner:'Ada',region:'east',status:'suspended',balance:130,opened:2025},
   {id:'A9',owner:'Bo',region:'east',status:'active',balance:70,opened:2026},
   {id:'A10',owner:'Dee',region:'west',status:'closed',balance:-9,opened:2023},
   {id:'A11',owner:'Eli',region:'north',status:'active',balance:70,opened:2025},
   {id:'A12',owner:'Cy',region:'south',status:'active',balance:55,opened:2026}],
  [{id:'B5',owner:'Eli',region:'west',status:'active',balance:70,opened:2025},
   {id:'B6',owner:'Bo',region:'east',status:'suspended',balance:-11,opened:2026},
   {id:'B7',owner:'Ada',region:'north',status:'active',balance:120,opened:2024},
   {id:'B8',owner:'Dee',region:'south',status:'active',balance:50,opened:2025},
   {id:'B9',owner:'Cy',region:'west',status:'closed',balance:-3,opened:2026},
   {id:'B10',owner:'Dee',region:'north',status:'active',balance:80,opened:2026},
   {id:'B11',owner:'Bo',region:'west',status:'active',balance:90,opened:2025},
   {id:'B12',owner:'Eli',region:'east',status:'active',balance:25,opened:2024}],
];
const accountBatches = tableDatasets.map((base, index) => [...base, ...accountExtras[index]]);
tablePolicies.forEach((policy, p) => accountBatches.forEach((accounts, variant) => {
  add('input_accounts', p * 2 + variant, {
    type: '(request: string, accounts: { id: string, owner: string, region: string, status: string, balance: number, opened: number }[]) => string[]',
    instructions: 'Examine the complete accounts batch before deciding how to implement the request.\n' +
      'Write the selected account processing code in eval.\nReturn the requested account ids.',
    function: 'select_batch_accounts',
  }, { request: policy.memo, accounts }, policy.run(accounts));
}));

const eventExtras = [
  [{id:'L7',minute:17,service:'worker',level:'warn',note:'queue'},
   {id:'L8',minute:18,service:'security',level:'fail',note:'unexpected'},
   {id:'L9',minute:19,service:'api',level:'warn',note:'latency'},
   {id:'L10',minute:20,service:'worker',level:'ok',note:'recovered'},
   {id:'L11',minute:21,service:'security',level:'ok',note:'recovered'},
   {id:'L12',minute:22,service:'api',level:'fail',note:'expected'}],
  [{id:'M7',minute:21,service:'security',level:'fail',note:'unexpected'},
   {id:'M8',minute:22,service:'api',level:'warn',note:'latency'},
   {id:'M9',minute:23,service:'security',level:'warn',note:'token'},
   {id:'M10',minute:24,service:'worker',level:'fail',note:'expected'},
   {id:'M11',minute:25,service:'api',level:'ok',note:'recovered'},
   {id:'M12',minute:26,service:'security',level:'fail',note:'unexpected'}],
];
const eventBatches = logDatasets.map((base, index) => [...base, ...eventExtras[index]]);
logPolicies.forEach((policy, p) => eventBatches.forEach((events, variant) => {
  add('input_events', p * 2 + variant, {
    type: '(request: string, events: { id: string, minute: number, service: string, level: string, note: string }[]) => string[]',
    instructions: 'Examine the complete event batch before deciding how to implement the request.\n' +
      'Write the selected event analysis in eval.\nReturn the requested event ids.',
    function: 'analyze_event_batch',
  }, { request: policy.memo, events }, policy.run(events));
}));

// A full incident window exercises value paging: later recoveries can change
// whether an earlier failure is still unresolved.
const incidentWindow = [...eventBatches[0],
  {id:'L13',minute:23,service:'worker',level:'fail',note:'queue stalled'},
  {id:'L14',minute:24,service:'api',level:'warn',note:'latency'},
  {id:'L15',minute:25,service:'security',level:'fail',note:'token rejected'},
  {id:'L16',minute:26,service:'worker',level:'warn',note:'retrying'},
  {id:'L17',minute:27,service:'api',level:'fail',note:'upstream timeout'},
  {id:'L18',minute:28,service:'worker',level:'ok',note:'queue drained'},
  {id:'L19',minute:29,service:'security',level:'warn',note:'new token'},
  {id:'L20',minute:30,service:'api',level:'ok',note:'upstream recovered'},
  {id:'L21',minute:31,service:'security',level:'ok',note:'token rotated'},
  {id:'L22',minute:32,service:'worker',level:'fail',note:'worker restarted'},
  {id:'L23',minute:33,service:'api',level:'warn',note:'slow requests'},
  {id:'L24',minute:34,service:'worker',level:'warn',note:'backlog growing'},
  {id:'L25',minute:35,service:'security',level:'fail',note:'login rejected'},
  {id:'L26',minute:36,service:'api',level:'fail',note:'new timeout'},
  {id:'L27',minute:37,service:'worker',level:'ok',note:'backlog cleared'},
  {id:'L28',minute:38,service:'security',level:'warn',note:'rotation pending'},
  {id:'L29',minute:39,service:'api',level:'warn',note:'upstream degraded'},
  {id:'L30',minute:40,service:'security',level:'fail',note:'login rejected again'}];
const unresolvedPolicy = logPolicies[3];
add('input_events_paged', 0, {
  type: '(request: string, events: { id: string, minute: number, service: string, level: string, note: string }[]) => string[]',
  instructions: 'Inspect the incident window and decide how to implement the request.\n' +
    'Write the event analysis in eval.\nReturn the requested event ids.',
  function: 'analyze_incident_window',
}, { request: unresolvedPolicy.memo, events: incidentWindow }, unresolvedPolicy.run(incidentWindow));

// These incident notes require interpreting what happened, not just applying a
// field predicate. The preview contains only the first few notes.
const incidentNotes = [
  [
    {id:'A1',note:'Finance confirmed a second settled payment for the same order; the customer was charged twice.'},
    {id:'A2',note:'For a separate order, the second bank entry is only a pending authorization and has not settled.'},
    {id:'A3',note:'The reporter cleared the cache and can now open the dashboard again.'},
    {id:'A4',note:'The operator intends to roll back once traffic drains, but has not started.'},
    {id:'A5',note:'Rollback completed and the error rate has stayed at baseline for twenty minutes.'},
    {id:'A6',note:'The blank page reproduces in staging; production users are unaffected.'},
    {id:'A7',note:'Production checkout is rejecting cards, and monitoring confirms seventy failed attempts.'},
    {id:'A8',note:'Support thinks the incident may have recovered; no telemetry or user retest is available.'},
    {id:'A9',note:'The vendor reports that its webhook backlog cleared, but our queue is still growing.'},
    {id:'A10',note:'The patch is deployed and smoke checks pass, yet the customer still sees the failure.'},
    {id:'A11',note:'Key rotation restored service; monitors and the affected user both confirm recovery.'},
    {id:'A12',note:'The suspicious integration was disabled, while the other tenants continue normally.'},
  ],
  [
    {id:'B1',note:'For one renewal, a duplicate invoice was drafted but voided before capture; the ledger shows one payment.'},
    {id:'B2',note:'For a different renewal, two captures posted to the ledger, confirmed by finance.'},
    {id:'B3',note:'A fix is queued for tomorrow; no deployment has happened.'},
    {id:'B4',note:'The on-call engineer reverted the release and both health checks and customer retries pass.'},
    {id:'B5',note:'A test tenant saw failed logins in staging; live tenant traffic is healthy.'},
    {id:'B6',note:'Live tenants cannot upload files, confirmed in access logs and support reports.'},
    {id:'B7',note:'The provider marked the incident resolved, but our failed uploads are still increasing.'},
    {id:'B8',note:'A restart finished, though there has been no customer retest and metrics remain unavailable.'},
    {id:'B9',note:'The queue drained after the restart, and both telemetry and a user confirm uploads work.'},
    {id:'B10',note:'The patch passed tests but has not reached production.'},
    {id:'B11',note:'Synthetic checks look green, but a customer reproduced the error after deployment.'},
    {id:'B12',note:'The affected integration was switched off, preventing further bad events.'},
  ],
];
const incidentQuestions = [
  { request:'Which notes confirm that the reported problem has recovered? Require actual user or operational evidence, rather than a plan or an unverified claim.',
    expected:[['A3','A5','A11'],['B4','B9']] },
  { request:'Which notes establish real production or customer impact, rather than staging effects or a transaction that never settled?',
    expected:[['A1','A7','A10'],['B2','B6','B11']] },
  { request:'Which notes describe a remedy that is only planned or queued and has not been applied yet?',
    expected:[['A4'],['B3','B10']] },
  { request:'Which notes report an external claim of recovery that local evidence contradicts?',
    expected:[['A9'],['B7']] },
  { request:'Which notes describe a mitigation that was actually carried out, regardless of whether it fully resolved the issue?',
    expected:[['A3','A5','A10','A11','A12'],['B4','B8','B9','B11','B12']] },
];
incidentQuestions.forEach((question, q) => incidentNotes.forEach((notes, variant) => {
  add('input_incidents', q * 2 + variant, {
    type: '(request: string, notes: { id: string, note: string }[]) => string[]',
    instructions: 'Read the incident notes, then interpret the request using their meaning and evidence.\n' +
      'Write the resulting selection in eval.\nReturn the matching note ids in source order.',
    function: 'review_incident_notes',
  }, { request: question.request, notes }, question.expected[variant]);
}));

await mkdir(dirname(output), { recursive: true });
await writeFile(output, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
const examples = [0, 5, 12, 17, 24, 29, 36, 41, 49, 54].map(index => {
  const row = rows[index], s = row.semantics;
  const input = s.folder_files ? Object.entries(s.folder_files).map(([path, content]) =>
    `**${path}**\n\n\`\`\`text\n${content.trimEnd()}\n\`\`\``).join('\n\n') :
    `\`\`\`json\n${JSON.stringify(s.inputs, null, 2)}\n\`\`\``;
  return `## ${row.id}\n\n${s.files[s.root].replace(/^---\n[\s\S]*?\n---\n/, '').trim()}\n\n${input}\n\n` +
    `**Expected return**\n\n\`\`\`json\n${JSON.stringify(s.expected)}\n\`\`\`\n`;
});
const sheet = '# Read-before-code probe: sample problems\n\n' +
  'These ten cases illustrate the 59-case [IR corpus](../data/teacher/read-before-code-probe.ir.jsonl). ' +
  'The agent sees the function instructions and typed scope. Large ordinary input batches show only their first few rows in the opening preview. ' +
  'Folder contents must be read through the folder handle in `eval`. Expected returns are oracle data, not shown to the agent.\n\n' +
  examples.join('\n');
await writeFile(resolve('docs/read-before-code-examples.md'), sheet);
console.log(`${rows.length} cases across ${new Set(rows.map(row => row.family)).size} families -> ${output}`);
