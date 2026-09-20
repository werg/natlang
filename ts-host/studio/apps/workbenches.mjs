import { define, initial, COMMON, action as a, field as f, panel as p, form, table, text, requireValue as check, revised, uid, unsupported, json } from '../shared/domain.mjs';
const receiptType = 'export type Receipt = { id: Text; operation: Text; status: Text; output: Text };';
const receipt = (s, operation, result) => { s.receipts.push({ id: uid(s.receipts), operation, status: result.status ?? 'complete', output: json(result) }); };
const receipts = s => table('receipts', 'What actually happened', s.receipts, ['operation', 'status', 'output']);
export const media = define({ id: 'media', project: 'P01', title: 'Cut & light', subtitle: 'Shape a moment worth keeping', category: 'Create', icon: '◐', color: '#d5ae86', panelIds: ['preview', 'edit', 'receipts'],
    stateType: `${receiptType} export type State = { ${COMMON} asset: Text; output: Text; receipts: Receipt[] };`,
    initial: () => initial({ asset: '', output: '', receipts: [] }),
    instructions: 'Operate the CPU FFmpeg companion. sample creates a short synthetic video fixture. transform uses target trim, scale, crop or transcode and text JSON parameters {start,end,x,y,width,height,keep_audio}; native code validates against the asset. Uploaded assets arrive through the explicit import event with target host asset ID. Receipts contain actual hashes and inspected dimensions. Do not claim perceptual quality without visual evidence.',
    async apply(s, d, host) { if (d.action === 'sample') {
        const r = await host.service('media.sample', {});
        s.asset = r.asset;
        receipt(s, 'sample', r);
    }
    else if (d.action === 'import') {
        check(/^asset-[\w-]+\.mp4$/.test(d.target), 'Invalid uploaded asset');
        s.asset = d.target;
        s.output = '';
    }
    else if (d.action === 'transform') {
        check(s.asset, 'Create or import a video first');
        const r = await host.service('media.transform', { asset: s.asset, kind: d.target, parameters: JSON.parse(d.text) });
        s.output = r.asset;
        receipt(s, d.target, r);
    }
    else
        unsupported(d.action); return revised(s, 'Media operation completed; inspect the preview and receipt.'); },
    panels: s => [p('preview', 'A frame of possibility', 'media', { asset: s.output || s.asset, actions: [a('Create sample clip', 'sample')], upload: true }), form('edit', 'Give it a direction', [f('target', 'Transformation', 'trim', 'select', ['trim', 'scale', 'crop', 'transcode']), f('text', 'Exact parameters · JSON', '{"start":0,"end":2,"width":320,"height":180,"x":0,"y":0,"keep_audio":false}', 'code')], [a('Render & inspect', 'transform', {}, 'primary')], { footer: 'CPU FFmpeg runs in the local companion. Files remain in its studio workspace.' }), receipts(s)], smoke: { action: 'sample' } });
export const build = define({ id: 'build', project: 'P03', title: 'Foundry', subtitle: 'Small steps. Reproducible results.', category: 'Develop', icon: '⬡', color: '#b6bba0', panelIds: ['graph', 'source', 'receipts'],
    stateType: `${receiptType} export type State = { ${COMMON} source: Text; operation: Text; output: Text; receipts: Receipt[] };`,
    initial: () => initial({ source: 'A careful build begins with declared inputs.\n', operation: 'uppercase', output: '', receipts: [] }),
    instructions: 'Build a declared-input artifact through the real BuildWorkspace cache. save updates source text and transformation target copy or uppercase. build requests execution keyed by content, operation and output identity. Repeated identical builds reuse and verify cached bytes. If a build fails inspect its error before proposing a source repair; do not report successful checks without receipts.',
    async apply(s, d, host) { if (d.action === 'save') {
        check(['copy', 'uppercase'].includes(d.target), 'Unknown builtin');
        s.source = text(d.text);
        s.operation = d.target;
        s.output = '';
    }
    else if (d.action === 'build') {
        const r = await host.service('build.run', { source: s.source, operation: s.operation });
        s.output = r.output;
        receipt(s, 'build', r);
    }
    else
        unsupported(d.action); return revised(s, 'Build workspace updated.'); },
    panels: s => [p('graph', 'A transparent build', 'workflow', { steps: [{ id: 'source', title: 'Declared source', status: 'ready', needs: [] }, { id: 'transform', title: s.operation, status: s.output ? 'succeeded' : 'pending', needs: ['source'] }, { id: 'artifact', title: 'Verified artifact', status: s.output ? 'succeeded' : 'pending', needs: ['transform'] }] }), form('source', 'Inputs you can inspect', [f('text', 'Source', s.source, 'code'), f('target', 'Exact operation', s.operation, 'select', ['uppercase', 'copy'])], [a('Save input', 'save'), a('Build artifact', 'build', {}, 'primary')], { output: s.output }), receipts(s)], smoke: { action: 'build' } });
export const terminal = define({ id: 'terminal', project: 'P04', title: 'Waypoint', subtitle: 'A terminal with a sense of direction', category: 'Develop', icon: '›_', color: '#9ac3b0', panelIds: ['terminal', 'recipes', 'receipts'],
    stateType: `${receiptType} export type Line = { id: Text; command: Text; output: Text; status: Text }; export type State = { ${COMMON} lines: Line[]; receipts: Receipt[] };`,
    initial: () => initial({ lines: [], receipts: [] }),
    instructions: 'A semantic terminal in an explicitly trusted local companion. execute runs Bash text in the studio session directory and records actual stdout/stderr/exit code. recipe selects target list, system, or example. Translate user goals into commands judiciously; commands can access the host user environment. Do not claim a sandbox, successful command, or reversible external effect. The user can cancel the owned process group and inspect its job receipt.',
    async apply(s, d, host) { let command; if (d.action === 'execute')
        command = text(d.text);
    else if (d.action === 'recipe') {
        command = { list: 'ls -lah', system: 'uname -s; pwd', example: 'printf "Hello from the natlang studio\\n"' }[d.target];
        check(command, 'Unknown recipe');
    }
    else
        unsupported(d.action); const r = await host.service('terminal.run', { command }); s.lines.push({ id: uid(s.lines), command, output: r.output, status: r.status }); receipt(s, 'bash', r); return revised(s, `Command ${r.status}.`); },
    panels: s => [form('terminal', 'Your working session', [f('text', 'Bash command', 'pwd', 'code')], [a('Run command', 'execute', {}, 'primary')], { terminal: s.lines, footer: 'Trusted local Bash. The working directory is a studio session; the process retains your host user’s authority.' }), p('recipes', 'A few useful starting points', 'cards', { items: [{ title: 'Look around', body: 'List the working directory.', actions: [a('List files', 'recipe', { target: 'list' })] }, { title: 'Know your surroundings', body: 'Show system and directory.', actions: [a('Inspect environment', 'recipe', { target: 'system' })] }, { title: 'Say hello', body: 'A small, real process.', actions: [a('Run example', 'recipe', { target: 'example' })] }] }), receipts(s)], smoke: { action: 'recipe', target: 'example' } });
export const packages = define({ id: 'packages', project: 'P07', title: 'Parcel', subtitle: 'Good tools deserve a good home', category: 'Develop', icon: '⬢', color: '#aaaed3', panelIds: ['catalog', 'resolve', 'receipts'],
    stateType: `${receiptType} export type State = { ${COMMON} catalog: Text; lock: Text; receipts: Receipt[] };`,
    initial: () => initial({ catalog: '', lock: '', receipts: [] }),
    instructions: 'Use the real offline package registry. browse reads the companion catalog. resolve chooses package target and semver range text, engine typescript-host, and stores an exact content-addressed lock. install installs the current lock under target name through atomic pointer creation; installed names are immutable. Never invent compatibility or package contents. The initial registry ships small executable example bundles.',
    async apply(s, d, host) { if (d.action === 'browse') {
        const r = await host.service('packages.catalog', {});
        s.catalog = json(r.catalog);
    }
    else if (d.action === 'resolve') {
        const r = await host.service('packages.resolve', { name: d.target, range: d.text });
        check(r.locks.length, 'No compatible solution');
        s.lock = json(r.locks[0]);
        receipt(s, 'resolve', r);
    }
    else if (d.action === 'install') {
        check(s.lock, 'Resolve a lock first');
        const r = await host.service('packages.install', { lock: JSON.parse(s.lock), target: d.target });
        receipt(s, 'install', r);
    }
    else
        unsupported(d.action); return revised(s, 'Package workspace updated.'); },
    panels: s => [p('catalog', 'A small, useful collection', 'output', { output: s.catalog || 'Browse the bundled local registry to get started.', actions: [a('Browse packages', 'browse')] }), form('resolve', 'Choose an exact foundation', [f('target', 'Package', 'greetings'), f('text', 'Version range', '^1.0.0')], [a('Resolve dependencies', 'resolve', {}, 'primary')], { output: s.lock, extraForms: [{ title: 'Install verified bundle', fields: [f('target', 'Installation name', 'my_greetings')], actions: [a('Install lock', 'install')] }] }), receipts(s)], smoke: { action: 'resolve', target: 'greetings', text: '^1.0.0' } });
export const repositories = define({ id: 'repositories', project: 'P20', title: 'Patchwork', subtitle: 'Change with evidence', category: 'Develop', icon: '±', color: '#9fbdce', panelIds: ['diff', 'patch', 'receipts'],
    stateType: `${receiptType} export type State = { ${COMMON} before: Text; after: Text; receipts: Receipt[] };`,
    initial: () => initial({ before: 'export const greet = name => "Hello, " + name;\n', after: 'export const greet = name => "Hello, " + name;\n', receipts: [] }),
    instructions: 'Prepare a repository migration in an isolated candidate. propose supplies exact old text in text and replacement secondary; reject missing or ambiguous contexts. check writes the candidate into an isolated repository, executes syntax and greeting behavior checks, and returns actual evidence. reset restores the original. Download exports the proposed source; do not claim it was merged into the user repository.',
    async apply(s, d, host) { if (d.action === 'propose') {
        const old = text(d.text);
        check(s.after.includes(old) && s.after.indexOf(old) === s.after.lastIndexOf(old), 'Patch context missing or ambiguous');
        check(typeof d.secondary === 'string', 'Replacement required');
        s.after = s.after.replace(old, d.secondary);
    }
    else if (d.action === 'check') {
        const r = await host.service('repository.check', { before: s.before, after: s.after });
        receipt(s, 'checks', r);
    }
    else if (d.action === 'reset')
        s.after = s.before;
    else
        unsupported(d.action); return revised(s, 'Candidate revision updated.'); },
    panels: s => [p('diff', 'A change you can review', 'diff', { before: s.before, after: s.after, download: { name: 'greet.mjs', text: s.after } }), form('patch', 'One precise edit', [f('text', 'Exact existing text', '"Hello, "', 'code'), f('secondary', 'Replacement', '"Welcome, "', 'code')], [a('Apply to candidate', 'propose'), a('Run isolated checks', 'check', {}, 'primary'), a('Reset candidate', 'reset')], { footer: 'The example behavior gate expects greet("Ada") to return "Hello, Ada". A changed greeting should expose a failing test.' }), receipts(s)], smoke: { action: 'check' } });
export default [media, build, terminal, packages, repositories];
