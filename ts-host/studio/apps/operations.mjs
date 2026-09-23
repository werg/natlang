import { define, initial, COMMON, action as a, field as f, panel as p, form, table, find, text, number, requireValue as check, revised, uid, unsupported, parseRows, json } from '../shared/domain.mjs';
export const scheduling = define({ id: 'scheduling', project: 'P19', title: 'Daylight', subtitle: 'Make room for what matters', category: 'Organize', icon: '◷', color: '#e1c185', panelIds: ['day', 'tasks', 'plan'],
    stateType: `export type Task = { id: string; title: string; duration: number; start: string; done: boolean }; export type State = { ${COMMON} tasks: Task[]; date: string };`,
    initial: () => initial({ date: '2026-09-21', tasks: [{ id: 'write', title: 'A quiet hour of writing', duration: 60, start: '', done: false }, { id: 'walk', title: 'Walk outside', duration: 30, start: '', done: false }, { id: 'review', title: 'Review the experiment', duration: 45, start: '', done: false }] }),
    instructions: 'Plan tasks in explicit UTC. add creates task title text with duration amount minutes. schedule sets target task start from text ISO datetime; exact code rejects overlaps. complete toggles completion. unschedule clears start. date changes displayed UTC date. Plan one task at a time in response to priorities, respecting duration and existing commitments. Do not imply calendar provider synchronization.',
    apply(s, d) { if (d.action === 'add')
        s.tasks.push({ id: uid(s.tasks), title: text(d.text), duration: number(d.amount, 'Duration', 1, 1440), start: '', done: false });
    else if (d.action === 'date') {
        check(/^\d{4}-\d{2}-\d{2}$/.test(d.text), 'Use YYYY-MM-DD');
        s.date = d.text;
    }
    else {
        const task = find(s.tasks, d.target);
        if (d.action === 'schedule') {
            const start = Date.parse(d.text);
            check(Number.isFinite(start) && /Z$/.test(d.text), 'Use an ISO timestamp ending in Z (UTC)');
            const end = start + task.duration * 60000;
            check(!s.tasks.some(t => t.id !== task.id && t.start && start < Date.parse(t.start) + t.duration * 60000 && end > Date.parse(t.start)), 'This overlaps another task');
            task.start = new Date(start).toISOString();
        }
        else if (d.action === 'complete')
            task.done = !task.done;
        else if (d.action === 'unschedule')
            task.start = '';
        else
            unsupported(d.action);
    } return revised(s, 'Your day has a little more shape.'); },
    panels: s => [p('day', 'A little breathing room', 'timeline', { date: s.date, tasks: s.tasks }), form('tasks', 'Things worth doing', [f('text', 'Task', 'Read and reflect'), f('amount', 'Minutes', 30, 'number')], [a('Add task', 'add')], { rows: s.tasks, columns: ['title', 'duration', 'start', 'done'], rowActions: [a('Done / reopen', 'complete'), a('Unschedule', 'unschedule')] }), form('plan', 'Find its place', [f('target', 'Task', s.tasks[0]?.id ?? '', 'select', s.tasks.map(t => t.id)), f('text', 'Start · UTC', `${s.date}T09:00:00Z`)], [a('Reserve time', 'schedule', {}, 'primary')], { footer: 'Times are explicitly UTC. Collisions are checked before a reservation is saved.' })], smoke: { action: 'schedule', target: 'write', text: '2026-09-21T09:00:00Z' } });
export const logs = define({ id: 'logs', project: 'P12', title: 'Signal room', subtitle: 'Find the story in the noise', category: 'Observe', icon: '⌁', color: '#90bdb4', panelIds: ['stream', 'incidents', 'escalation'],
    stateType: `export type Log = { id: string; time: string; level: string; message: string }; export type Incident = { id: string; title: string; evidence: string[]; severity: string; status: string }; export type State = { ${COMMON} logs: Log[]; incidents: Incident[]; receipts: string[] };`,
    initial: () => initial({ logs: [{ id: 'l1', time: '10:02:01', level: 'info', message: 'Deploy v42 began.' }, { id: 'l2', time: '10:02:13', level: 'error', message: 'Database pool exhausted: 100/100 connections.' }, { id: 'l3', time: '10:02:14', level: 'error', message: 'GET /checkout timed out after 5000ms.' }, { id: 'l4', time: '10:02:18', level: 'warn', message: 'Retry queue depth reached 240.' }], incidents: [], receipts: [] }),
    instructions: 'Investigate logs with evidence. ingest parses text as JSON array of {id,time,level,message}; repeated identical IDs are ignored, conflicting duplicate IDs rejected. incident adds title text, severity target info/warning/critical, evidence IDs ids. resolve marks incident target resolved. escalate writes a LOCAL escalation receipt for target incident, idempotently. No message is sent to an external service. Distinguish correlation from a proven cause.',
    apply(s, d) { if (d.action === 'ingest') {
        for (const row of parseRows(d.text)) {
            for (const key of ['id', 'time', 'level', 'message'])
                text(row[key], key);
            const prior = s.logs.find(l => l.id === row.id);
            check(!prior || JSON.stringify(prior) === JSON.stringify(row), 'Conflicting duplicate log ID');
            if (!prior)
                s.logs.push({ id: row.id, time: row.time, level: row.level, message: row.message });
        }
    }
    else if (d.action === 'incident') {
        check(['info', 'warning', 'critical'].includes(d.target), 'Unknown severity');
        check(d.ids?.length, 'Cite log evidence');
        d.ids.forEach(id => find(s.logs, id));
        s.incidents.push({ id: uid(s.incidents), title: text(d.text), severity: d.target, evidence: [...new Set(d.ids)], status: 'open' });
    }
    else if (d.action === 'resolve')
        find(s.incidents, d.target).status = 'resolved';
    else if (d.action === 'escalate') {
        find(s.incidents, d.target);
        if (!s.receipts.includes(d.target))
            s.receipts.push(d.target);
    }
    else
        unsupported(d.action); return revised(s, 'Investigation updated.'); },
    panels: s => [table('stream', 'Log window', s.logs, ['time', 'level', 'message'], { extraForms: [{ title: 'Append a log window', fields: [f('text', 'Log records · JSON', '[{"id":"l5","time":"10:03:00","level":"info","message":"Pool recovered."}]', 'code')], actions: [a('Ingest records', 'ingest')] }] }), form('incidents', 'Evidence before alarm', [f('text', 'Incident summary', 'Checkout degraded after connection saturation.'), f('target', 'Severity', 'warning', 'select', ['info', 'warning', 'critical']), f('ids', 'Evidence IDs', 'l2,l3')], [a('Open incident', 'incident')], { rows: s.incidents, columns: ['id', 'title', 'severity', 'status'], rowActions: [a('Resolve', 'resolve'), a('Record escalation', 'escalate')] }), p('escalation', 'Escalation ledger', 'cards', { items: s.receipts.map(id => ({ title: id, body: 'Local receipt recorded. No external notification sent.' })), empty: 'No escalations recorded.' })], smoke: { action: 'incident', text: 'Pool saturation', target: 'warning', ids: ['l2', 'l3'] } });
export const data = define({ id: 'data', project: 'P13', title: 'Data kitchen', subtitle: 'Bring messy records into alignment', category: 'Explore', icon: '▦', color: '#a9c79c', panelIds: ['input', 'mapping', 'output'],
    stateType: `export type Mapping = { id: string; source: string; target: string }; export type State = { ${COMMON} input: string; mappings: Mapping[]; output: string; issues: string[] };`,
    initial: () => initial({ input: '[{"Full Name":"Ada Lovelace","Email Address":"ada@example.test"},{"Full Name":"Grace Hopper","Email Address":"grace@example.test"}]', mappings: [{ id: 'name', source: 'Full Name', target: 'name' }, { id: 'email', source: 'Email Address', target: 'email' }], output: '', issues: [] }),
    instructions: 'Reconcile schemas. input validates JSON records text. map sets source column text to target column secondary, rejecting duplicate targets. remove deletes mapping target ID. transform applies the complete mapping exactly and records missing fields rather than silently discarding them. Use semantic knowledge to propose mappings in response to a command; never silently merge distinct people or invent missing values.',
    apply(s, d) { if (d.action === 'input') {
        parseRows(d.text);
        s.input = d.text;
        s.output = '';
    }
    else if (d.action === 'map') {
        const source = text(d.text), target = text(d.secondary);
        check(!s.mappings.some(m => m.target === target && m.source !== source), 'Target column already mapped');
        s.mappings = s.mappings.filter(m => m.source !== source);
        s.mappings.push({ id: uid(s.mappings), source, target });
        s.output = '';
    }
    else if (d.action === 'remove') {
        s.mappings = s.mappings.filter(m => m.id !== d.target);
        s.output = '';
    }
    else if (d.action === 'transform') {
        check(s.mappings.length, 'Add a mapping');
        s.issues = [];
        const rows = parseRows(s.input).map((row, i) => Object.fromEntries(s.mappings.map(m => { if (!(m.source in row))
            s.issues.push(`Row ${i + 1}: missing ${m.source}`); return [m.target, row[m.source] ?? null]; })));
        s.output = json(rows);
    }
    else
        unsupported(d.action); return revised(s, 'Transformation workspace updated.'); },
    panels: s => [form('input', 'Ingredients', [f('text', 'Source records · JSON', s.input, 'code')], [a('Load records', 'input')], { rows: parseRows(s.input), columns: Object.keys(parseRows(s.input)[0] ?? {}) }), form('mapping', 'A shared vocabulary', [f('text', 'Source column', 'Full Name'), f('secondary', 'Target column', 'name')], [a('Set mapping', 'map'), a('Transform records', 'transform', {}, 'primary')], { rows: s.mappings, columns: ['source', 'target'], rowActions: [a('Remove', 'remove')] }), p('output', 'Ready to use', 'output', { output: s.output, rows: s.issues.map(issue => ({ issue })), columns: ['issue'], download: s.output ? { name: 'transformed.json', text: s.output } : null })], smoke: { action: 'transform' } });
export const workflows = define({ id: 'workflows', project: 'P15', title: 'Relay', subtitle: 'Make every step accountable', category: 'Organize', icon: '⇢', color: '#b0b6da', panelIds: ['flow', 'actions', 'receipts'],
    stateType: `export type WorkflowStep = { id: string; title: string; needs: string[]; status: string; key: string }; export type Receipt = { id: string; step: string; status: string }; export type State = { ${COMMON} steps: WorkflowStep[]; receipts: Receipt[] };`,
    initial: () => initial({ steps: [{ id: 'reserve', title: 'Reserve inventory', needs: [], status: 'pending', key: 'order-1-reserve' }, { id: 'charge', title: 'Charge payment', needs: ['reserve'], status: 'pending', key: 'order-1-charge' }, { id: 'ship', title: 'Arrange shipment', needs: ['charge'], status: 'pending', key: 'order-1-ship' }], receipts: [] }),
    instructions: 'Explore a workflow against an explicit simulated provider. execute advances target step after its dependencies succeed, writing an idempotent receipt keyed by step key. fail simulates a known failure. unknown marks an ambiguous provider outcome; reconcile supplies text succeeded or failed as a simulated provider lookup. compensate reverses a succeeded step only after its succeeded dependents have been compensated. Never retry unknown outcomes blindly. This app does not charge cards or contact real providers.',
    apply(s, d) { const step = find(s.steps, d.target); if (d.action === 'execute') {
        if (step.status === 'succeeded')
            return s;
        check(step.status === 'pending' || step.status === 'failed', 'Reconcile unknown outcomes before retrying');
        check(step.needs.every(id => find(s.steps, id).status === 'succeeded'), 'Dependencies are not complete');
        step.status = 'succeeded';
        if (!s.receipts.some(r => r.id === step.key))
            s.receipts.push({ id: step.key, step: step.id, status: 'succeeded' });
    }
    else if (d.action === 'fail' || d.action === 'unknown') {
        check(step.status === 'pending', 'Only pending steps can be fault-injected');
        step.status = d.action === 'fail' ? 'failed' : 'unknown';
    }
    else if (d.action === 'reconcile') {
        check(step.status === 'unknown', 'No unknown outcome');
        check(['succeeded', 'failed'].includes(d.text), 'Use succeeded or failed');
        check(d.text !== 'succeeded' || step.needs.every(id => find(s.steps, id).status === 'succeeded'), 'Dependencies were not complete');
        step.status = d.text;
        if (d.text === 'succeeded' && !s.receipts.some(r => r.id === step.key))
            s.receipts.push({ id: step.key, step: step.id, status: 'succeeded' });
    }
    else if (d.action === 'compensate') {
        check(step.status === 'succeeded', 'Only successful steps can be compensated');
        check(!s.steps.some(r => r.needs.includes(step.id) && ['succeeded', 'unknown'].includes(r.status)), 'Compensate or reconcile dependent steps first');
        step.status = 'compensated';
        s.receipts.push({ id: step.key + '-compensate', step: step.id, status: 'compensated' });
    }
    else
        unsupported(d.action); return revised(s, 'Simulated provider state recorded.'); },
    panels: s => [p('flow', 'One order, many promises', 'workflow', { steps: s.steps, badge: 'simulated provider' }), form('actions', 'Advance or investigate', [f('target', 'Step', 'reserve', 'select', s.steps.map(r => r.id)), f('text', 'Reconciled outcome', 'succeeded', 'select', ['succeeded', 'failed'])], [a('Execute step', 'execute', {}, 'primary'), a('Inject failure', 'fail'), a('Inject unknown', 'unknown'), a('Reconcile', 'reconcile'), a('Compensate', 'compensate')]), table('receipts', 'Receipts survive retries', s.receipts, ['id', 'step', 'status'])], smoke: { action: 'execute', target: 'reserve' } });
export const tests = define({ id: 'tests', project: 'P17', title: 'Counterexample', subtitle: 'The most useful answer might be a failure', category: 'Develop', icon: '◇', color: '#d4acb7', panelIds: ['contract', 'cases', 'results'],
    stateType: `export type Case = { id: string; input: string; expected: string; actual: string; status: string }; export type State = { ${COMMON} source: string; contract: string; cases: Case[] };`,
    initial: () => initial({ source: 'export function main(input: { values: number[] }): number {\n  return input.values.reduce((a, b) => a + b, 0);\n}\n', contract: 'Sum all numbers. An empty list sums to zero.', cases: [{ id: 'empty', input: '{"values":[]}', expected: '0', actual: '', status: 'pending' }, { id: 'positive', input: '{"values":[2,3]}', expected: '5', actual: '', status: 'pending' }] }),
    algorithm: 'For run, walk all case IDs and invoke run_case for each. Inspect actual results, keep failures as evidence, and continue through the collection. The host runs only one case at a time. For requests to generate cases, reason from the contract and call add for each useful example; do not claim they were executed until run_case returns.',
    instructions: 'Build a behavior collection and seek counterexamples. save stores TypeScript source text exporting main(input) and contract secondary, invalidating results. add supplies case ID target, JSON input text, JSON expected secondary. run_case evaluates the target case using actual child execution. Natlang should propose boundary cases from the contract, but exact host execution determines results. Remove redundant examples only with a justified request; do not label generated cases as reviewed training data.',
    async apply(s, d, host) { if (d.action === 'save') {
        s.source = text(d.text);
        s.contract = text(d.secondary);
        s.cases.forEach(c => { c.status = 'pending'; c.actual = ''; });
    }
    else if (d.action === 'add') {
        JSON.parse(d.text);
        JSON.parse(d.secondary);
        check(!s.cases.some(c => c.id === d.target), 'Case ID exists');
        s.cases.push({ id: text(d.target), input: d.text, expected: d.secondary, actual: '', status: 'pending' });
    }
    else if (d.action === 'run_case') {
        const c = find(s.cases, d.target);
        try {
            const result = await host.runSource([{ id: 'subject.ts', source: s.source }], 'subject.ts', JSON.parse(c.input));
            c.actual = json(result.value);
            c.status = JSON.stringify(result.value) === JSON.stringify(JSON.parse(c.expected)) ? 'passed' : 'failed';
        }
        catch (error) {
            c.status = 'error';
            c.actual = String(error);
        }
    }
    else
        unsupported(d.action); return revised(s, 'Behavior collection updated.'); },
    panels: s => [form('contract', 'What should be true?', [f('secondary', 'Behavioral contract', s.contract, 'textarea'), f('text', 'Program', s.source, 'code')], [a('Save subject', 'save')]), form('cases', 'One revealing example', [f('target', 'Case ID', 'negative'), f('text', 'Input · JSON', '{"values":[-2,3]}', 'code'), f('secondary', 'Expected · JSON', '1', 'code')], [a('Add case', 'add'), a('Run collection', 'run', {}, 'primary')]), table('results', 'Evidence from execution', s.cases, ['id', 'status', 'input', 'expected', 'actual'])], smoke: { action: 'add', target: 'negative', text: '{"values":[-2,3]}', secondary: '1' } });
export default [scheduling, logs, data, workflows, tests];
