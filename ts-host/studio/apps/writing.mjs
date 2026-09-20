import { define, initial, COMMON, action as a, field as f, panel as p, form, table, find, text, requireValue as check, revised, uid, unsupported, json } from '../shared/domain.mjs';
const source = '---\nargs:\n  name: Text\nreturns: Text\n---\nGreet the person by name, warmly and briefly.';
export const ide = define({ id: 'ide', project: 'P06', title: 'Atelier', subtitle: 'A home for living programs', category: 'Create', icon: '⌘', color: '#b6a2ee',
    panelIds: ['source', 'result', 'scenarios'],
    references: s=>s.trace_id?[{store:'child_runs',id:s.trace_id}]:[],
    stateType: `export type File = { id: Text; source: Text }; export type Scenario = { id: Text; input: Text; expected: Text; actual: Text; status: Text }; export type State = { ${COMMON} files: File[]; selected: Text; result: Text; trace_id: Text; trace_count: Num; trace_cursor: Num; trace_frame: Text; scenarios: Scenario[] };`,
    initial: () => initial({ files: [{ id: 'hello.nl', source }], selected: 'hello.nl', result: '', trace_id: '', trace_count: 0, trace_cursor: 0, trace_frame: '', scenarios: [] }),
    algorithm: 'For evaluate, iterate over all scenario IDs, calling apply with action evaluate_case and the scenario target. Inspect each actual result, preserve failures, and continue to cover the collection. Do not replace this with a single host evaluation loop.',
    instructions: 'Edit and run natlang source. save stores exact source text for target filename. add creates a new filename from target and source from text. run uses text as JSON inputs for selected source; exact child execution returns a trace reference. inspect reads frame amount from that trace; frames are host-owned, not copied wholesale into model context. scenario adds a frozen input (text) and expected JSON value (secondary). evaluate_case executes the target scenario against current source; the natlang caller iterates the collection. Never report tests passed without child results. Free-form requests may propose an edit using save. Preserve frontmatter and types.',
    async apply(s, d, host) {
        if (d.action === 'save')
            find(s.files, d.target).source = text(d.text, 'source');
        else if (d.action === 'select') {
            find(s.files, d.target);
            s.selected = d.target;
        }
        else if (d.action === 'add') {
            check(!s.files.some(row => row.id === d.target), 'Filename already exists');
            check(/^[\w/-]+\.(nl|ts)$/.test(d.target), 'Use a .nl or .ts filename');
            s.files.push({ id: d.target, source: text(d.text) });
            s.selected = d.target;
        }
        else if (d.action === 'run') {
            const result = await host.runSource(s.files, s.selected, JSON.parse(d.text || '{}'));
            s.result = json(result.value);
            s.trace_id = result.trace_id;
            s.trace_count = result.trace_count;
            s.trace_cursor = 0;
            s.trace_frame = result.trace_count ? await host.inspectTrace(result.trace_id, 0) : '';
        }
        else if (d.action === 'inspect') {
            s.trace_frame = await host.inspectTrace(s.trace_id, d.amount);
            s.trace_cursor = d.amount;
        }
        else if (d.action === 'scenario') {
            JSON.parse(d.text);
            JSON.parse(d.secondary);
            s.scenarios.push({ id: uid(s.scenarios), input: d.text, expected: d.secondary, actual: '', status: 'pending' });
        }
        else if (d.action === 'evaluate_case') {
            const row = find(s.scenarios, d.target);
            const out = await host.runSource(s.files, s.selected, JSON.parse(row.input));
            row.actual = json(out.value);
            row.status = JSON.stringify(out.value) === JSON.stringify(JSON.parse(row.expected)) ? 'passed' : 'failed';
        }
        else
            unsupported(d.action);
        return revised(s, d.action === 'run' ? 'Run completed. Inspect the reduction trace.' : 'Workspace updated.');
    },
    panels: s => [form('source', 'Source atelier', [f('target', 'File', s.selected, 'select', s.files.map(row => row.id)), f('text', 'Natlang source', find(s.files, s.selected).source, 'code')], [a('Save source', 'save'), a('Open selected file', 'select')], { badge: 'natlang', footer: 'Source edits are saved as revisions. Child programs receive their own host environment.' }),
        form('result', 'Run & inspect', [f('text', 'Inputs · JSON', '{"name":"Ada"}', 'code')], [a('Run program', 'run', {}, 'primary')], { output: s.result, extraForms: [{ title: `Reduction trace · ${s.trace_count} events`, fields: [f('amount', 'Trace cursor', s.trace_cursor, 'number')], actions: [a('Inspect frame', 'inspect')] }], trace: s.trace_frame ? [s.trace_frame] : [] }),
        form('scenarios', 'Behavior collection', [f('text', 'Input · JSON', '{"name":"Ada"}', 'code'), f('secondary', 'Expected · JSON', '"Hello, Ada!"', 'code')], [a('Add scenario', 'scenario'), a('Evaluate collection', 'evaluate')], { rows: s.scenarios, columns: ['id', 'status', 'expected', 'actual'] })],
    smoke: { action: 'save', target: 'hello.nl', text: source + '\nUse one sentence.' },
});
export const notebook = define({ references:s=>s.cells.filter(c=>c.result_id).map(c=>({store:'native_values',id:c.result_id})), id: 'notebook', project: 'P10', title: 'Fieldnotes', subtitle: 'Think in cells, follow the evidence', category: 'Create', icon: '▤', color: '#e9bd79', panelIds: ['cells', 'lineage'],
    stateType: `export type Cell = { id: Text; engine: Text; source: Text; needs: Text[]; result: Text; result_id: Text; status: Text }; export type State = { ${COMMON} cells: Cell[] };`,
    initial: () => initial({ cells: [{ id: 'numbers', engine: 'typescript-host', source: 'return [2, 3, 5, 7, 11];', needs: [], result: '', result_id: '', status: 'ready' }, { id: 'total', engine: 'typescript-host', source: 'return args.deps.numbers.reduce((sum, n) => sum + n, 0);', needs: ['numbers'], result: '', result_id: '', status: 'ready' }] }),
    algorithm: 'For run, inspect the requested cell and its needs. Walk the dependency graph yourself, detecting cycles. Execute dependencies before dependents by calling apply with action execute and the individual cell target. Inspect every Step; stop dependents on failure. Reuse fresh completed dependencies. The host executes only one cell per operation. For add/save/remove use the matching exact operation.',
    instructions: 'Manage a mixed natlang, TypeScript, SQLite notebook. add creates a cell with target ID, engine in secondary, source in text, dependencies in ids. save updates existing source and invalidates dependent outputs. execute evaluates one cell only; its dependencies must already have complete outputs. The natlang caller resolves ordering. remove rejects removal while another cell depends on it. Never invent an execution result. Natlang cells use complete .nl frontmatter and receive deps input; TS gets args.deps; SQLite is provided by the local companion.',
    async apply(s, d, host) {
        if (d.action === 'add') {
            check(/^[A-Za-z_]\w*$/.test(d.target) && !s.cells.some(c => c.id === d.target), 'Choose a new identifier');
            check(['typescript-host', 'sqlite', 'natlang'].includes(d.secondary), 'Unknown engine');
            (d.ids ?? []).forEach(id => find(s.cells, id));
            s.cells.push({ id: d.target, engine: d.secondary, source: text(d.text), needs: d.ids ?? [], result: '', result_id: '', status: 'ready' });
        }
        else if (d.action === 'save') {
            find(s.cells, d.target).source = text(d.text);
            const dirty = new Set([d.target]);
            for (let n = 0; n < s.cells.length; n++)
                for (const c of s.cells)
                    if (c.needs.some(id => dirty.has(id)))
                        dirty.add(c.id);
            for (const c of s.cells)
                if (dirty.has(c.id)) {
                    c.result = ''; c.result_id='';
                    c.status = 'stale';
                }
        }
        else if (d.action === 'remove') {
            check(!s.cells.some(c => c.needs.includes(d.target)), 'A dependent cell still uses this cell');
            find(s.cells, d.target);
            s.cells = s.cells.filter(c => c.id !== d.target);
        }
        else if (d.action === 'execute') {
            const c = find(s.cells, d.target);
            const deps = {};
            for (const id of c.needs) {
                const parent = find(s.cells, id);
                check(parent.status === 'complete', 'Dependency is missing or stale: ' + id);
                deps[id] = await host.readValue(parent.result_id);
            }
            const out = await host.cell(c, deps);
            c.result = out.preview; c.result_id=out.id;
            c.status = 'complete';
        }
        else
            unsupported(d.action);
        return revised(s, d.action === 'execute' ? 'Cell completed; its output is tied to this revision.' : 'Notebook updated.');
    },
    panels: s => [p('cells', 'Your notebook', 'cells', { cells: s.cells }), form('lineage', 'Add a cell', [f('target', 'Cell identifier', 'analysis'), f('secondary', 'Engine', 'typescript-host', 'select', ['typescript-host', 'natlang', 'sqlite']), f('ids', 'Dependencies (comma separated)', 'numbers'), f('text', 'Source', 'return args.deps.numbers.map(n => n * n);', 'code')], [a('Add cell', 'add')], { rows: s.cells.map(c => ({ cell: c.id, needs: c.needs.join(' → ') || 'source', status: c.status })), columns: ['cell', 'needs', 'status'] })],
    smoke: { action: 'add', target: 'extra', secondary: 'typescript-host', text: 'return 42;', ids: [] },
});
export const wiki = define({ id: 'wiki', project: 'P11', title: 'Commonplace', subtitle: 'Pages that think with you', category: 'Create', icon: '◈', color: '#8ac9b7', panelIds: ['pages', 'editor', 'read'],
    stateType: `export type Page = { id: Text; title: Text; body: Text; cell: Text; output: Text }; export type State = { ${COMMON} pages: Page[]; selected: Text; incoming: Text; conflicts: Text[] };`,
    initial: () => initial({ pages: [{ id: 'welcome', title: 'A garden of shared ideas', body: '# Welcome to Commonplace\n\nA page can hold an idea, a conversation, and a living program.\n\n## Our next experiment\nBuild something small enough to understand and interesting enough to surprise us.', cell: source, output: '' }], selected: 'welcome', incoming: '', conflicts: [] }),
    instructions: 'Manage a local wiki. save writes title in secondary and body in text to selected page. add creates page target/title secondary/body text. select changes page. incoming stores a collaborator draft. merge must semantically combine the current body and incoming draft, return merged text and unresolved issues as ids. There is no crisp convergence rule: use the pinned model, seed, source and ordered inputs. run executes selected page cell with JSON text inputs. save_cell updates cell. Do not pretend local pages are synchronized over a network.',
    async apply(s, d, host) { const page = find(s.pages, s.selected); if (d.action === 'save') {
        page.body = text(d.text);
        page.title = text(d.secondary);
    }
    else if (d.action === 'add') {
        check(!s.pages.some(p => p.id === d.target), 'Page exists');
        s.pages.push({ id: text(d.target), title: text(d.secondary), body: text(d.text), cell: source, output: '' });
        s.selected = d.target;
    }
    else if (d.action === 'select') {
        find(s.pages, d.target);
        s.selected = d.target;
    }
    else if (d.action === 'incoming')
        s.incoming = text(d.text);
    else if (d.action === 'merge') {
        check(s.incoming, 'Load an incoming draft first');
        page.body = text(d.text);
        s.conflicts = d.ids ?? [];
        s.incoming = '';
    }
    else if (d.action === 'save_cell') {
        page.cell = text(d.text);
        page.output = '';
    }
    else if (d.action === 'run') {
        const out = await host.runSource([{ id: 'page.nl', source: page.cell }], 'page.nl', JSON.parse(d.text || '{}'));
        page.output = json(out.value);
    }
    else
        unsupported(d.action); return revised(s, 'Page revision saved.'); },
    panels: s => { const page = find(s.pages, s.selected); return [form('pages', 'The collection', [f('target', 'Page', s.selected, 'select', s.pages.map(p => p.id))], [a('Open page', 'select')], { extraForms: [{ title: 'New page', fields: [f('target', 'Page ID', 'new-page'), f('secondary', 'Title', 'A new idea'), f('text', 'Body', 'Begin here.', 'textarea')], actions: [a('Create page', 'add')] }] }), form('editor', 'Edit this page', [f('secondary', 'Title', page.title), f('text', 'Markdown', page.body, 'textarea')], [a('Save page', 'save')], { extraForms: [{ title: 'Collaborator draft', fields: [f('text', 'Incoming text', s.incoming, 'textarea')], actions: [a('Stage incoming draft', 'incoming')] }], footer: 'Use the command bar to request a semantic merge after staging a draft.' }), p('read', page.title, 'prose', { body: page.body, rows: s.conflicts.map(issue => ({ issue })), columns: ['issue'], extraForms: [{ title: 'Living cell', fields: [f('text', 'Natlang source', page.cell, 'code')], actions: [a('Save cell', 'save_cell')] }, { title: 'Run cell', fields: [f('text', 'Inputs · JSON', '{"name":"reader"}', 'code')], actions: [a('Run cell', 'run')] }], output: page.output })]; }, smoke: { action: 'save', secondary: 'Shared garden', text: '# Shared garden\nA new revision.' },
});
export const merge = define({ id: 'merge', project: 'P02', title: 'Confluence', subtitle: 'Explore what it means to agree', category: 'Explore', icon: '⋈', color: '#d0a5dd', panelIds: ['replicas', 'proposal', 'history'],
    stateType: `export type Entry = { id: Text; replica: Text; text: Text }; export type State = { ${COMMON} family: Text; base: Text; left: Text; right: Text; merged: Text; issues: Text[]; history: Entry[] };`,
    initial: () => initial({ family: 'document', base: 'The garden opens at 9. Volunteers water the seedlings.', left: 'The garden opens at 8 on Saturdays. Volunteers water the seedlings.', right: 'The garden opens at 9. Water seedlings before visitors arrive.', merged: '', issues: [], history: [] }),
    instructions: 'This is a semantic, notional CRDT laboratory without crisp convergence rules. configure sets family from target, base from text, left from secondary. right sets right replica text. propose semantically merges base, left and right into text with unresolved issues in ids. commit adopts the proposal as the new common base, recording both replica inputs and result. Preserve facts, account for both edits, and flag ambiguity. Families: document, set, map, ordered-list, calendar, task-board, tree, graph, inventory, conversation. Do not claim convergence from one run.',
    apply(s, d) { if (d.action === 'configure') {
        check(['document', 'set', 'map', 'ordered-list', 'calendar', 'task-board', 'tree', 'graph', 'inventory', 'conversation'].includes(d.target), 'Unknown family');
        s.family = d.target;
        s.base = text(d.text);
        s.left = text(d.secondary);
        s.merged = '';
    }
    else if (d.action === 'right') {
        s.right = text(d.text);
        s.merged = '';
    }
    else if (d.action === 'propose') {
        s.merged = text(d.text);
        s.issues = d.ids ?? [];
    }
    else if (d.action === 'commit') {
        check(s.merged, 'Create a proposal first');
        for (const [replica, value] of [['left', s.left], ['right', s.right], ['merge', s.merged]])
            s.history.push({ id: uid(s.history), replica, text: value });
        s.base = s.merged;
        s.left = s.merged;
        s.right = s.merged;
        s.merged = '';
    }
    else
        unsupported(d.action); return revised(s, 'Merge laboratory updated.'); },
    panels: s => [form('replicas', 'Two perspectives', [f('target', 'Data family', s.family, 'select', ['document', 'set', 'map', 'ordered-list', 'calendar', 'task-board', 'tree', 'graph', 'inventory', 'conversation']), f('text', 'Common ancestor', s.base, 'textarea'), f('secondary', 'Replica A', s.left, 'textarea')], [a('Save ancestor & replica A', 'configure')], { extraForms: [{ title: 'Replica B', fields: [f('text', 'Replica B', s.right, 'textarea')], actions: [a('Save replica B', 'right')] }] }), p('proposal', 'A possible agreement', 'prose', { body: s.merged || 'Ask natlang to merge the two perspectives. Every proposal remains inspectable before you adopt it.', rows: s.issues.map(issue => ({ issue })), columns: ['issue'], actions: [a('Adopt proposal', 'commit')], extraForms: [{ title: 'Supply a proposal', fields: [f('text', 'Merged value', s.merged, 'textarea'), f('ids', 'Unresolved issues (comma separated)', s.issues.join(','))], actions: [a('Stage proposal', 'propose')] }] }), table('history', 'Operation history', s.history, ['id', 'replica', 'text'])], smoke: { action: 'right', text: 'The garden closes at dusk.' } });
export const evidence = define({ id: 'evidence', project: 'P14', title: 'Atlas', subtitle: 'Ideas with a paper trail', category: 'Explore', icon: '◎', color: '#8ebbdc', panelIds: ['library', 'search', 'answer'],
    stateType: `export type Passage = { id: Text; title: Text; text: Text }; export type Claim = { id: Text; text: Text; passage: Text; quote: Text }; export type State = { ${COMMON} passages: Passage[]; query: Text; hits: Text[]; claims: Claim[] };`,
    initial: () => initial({ passages: [{ id: 'garden', title: 'Garden observation · June', text: 'Seedlings watered in the morning retained more moisture at noon than seedlings watered the previous evening.' }, { id: 'study', title: 'Trial notes · July', text: 'There were twelve seedlings in each group. Shade and soil composition were not controlled.' }], query: '', hits: [], claims: [] }),
    instructions: 'Build evidence-backed notes. add stores passage target/title secondary/text. search performs literal ranked retrieval with text query. claim adds an interpretation text, source passage target and literal quote secondary; the host rejects fabricated quotes. Interpret evidence carefully and state unknowns. remove deletes a claim by target. A matching quotation establishes provenance, not semantic entailment.',
    apply(s, d) { if (d.action === 'add') {
        check(!s.passages.some(p => p.id === d.target), 'Source ID exists');
        s.passages.push({ id: text(d.target), title: text(d.secondary), text: text(d.text) });
    }
    else if (d.action === 'search') {
        s.query = text(d.text);
        const terms = s.query.toLowerCase().split(/\s+/);
        s.hits = s.passages.map(p => ({ id: p.id, score: terms.filter(t => (p.title + ' ' + p.text).toLowerCase().includes(t)).length })).filter(p => p.score).sort((a, b) => b.score - a.score).map(p => p.id);
    }
    else if (d.action === 'claim') {
        const p = find(s.passages, d.target);
        check(p.text.includes(text(d.secondary, 'quotation')), 'Quotation is not present in the selected passage');
        s.claims.push({ id: uid(s.claims), text: text(d.text), passage: p.id, quote: d.secondary });
    }
    else if (d.action === 'remove')
        s.claims = s.claims.filter(c => c.id !== d.target);
    else
        unsupported(d.action); return revised(s, 'Evidence notebook updated.'); },
    panels: s => [form('library', 'Source library', [f('target', 'Source ID', 'source-3'), f('secondary', 'Source title', 'Field observation'), f('text', 'Passage', '', 'textarea')], [a('Add source', 'add')], { rows: s.passages, columns: ['id', 'title', 'text'] }), form('search', 'Follow a thread', [f('text', 'Search passages', s.query || 'seedlings')], [a('Search sources', 'search', {}, 'primary')], { rows: s.hits.map(id => find(s.passages, id)), columns: ['title', 'text'] }), form('answer', 'Write with evidence', [f('text', 'Claim', '', 'textarea'), f('target', 'Source', s.passages[0]?.id ?? '', 'select', s.passages.map(p => p.id)), f('secondary', 'Exact quotation', '', 'textarea')], [a('Add cited claim', 'claim')], { rows: s.claims, columns: ['text', 'passage', 'quote'], footer: 'Literal citation checks are exact; whether a passage supports a claim is a semantic judgment.' })], smoke: { action: 'search', text: 'seedlings' } });
export const publisher = define({ id: 'publisher', project: 'P18', title: 'Folio', subtitle: 'From scattered thoughts to a finished piece', category: 'Create', icon: '¶', color: '#e2aa96', panelIds: ['outline', 'compose', 'preview'],
    stateType: `export type Section = { id: Text; heading: Text; body: Text }; export type State = { ${COMMON} title: Text; sections: Section[]; published: Bool };`,
    initial: () => initial({ title: 'Notes from a small garden', sections: [{ id: 'intro', heading: 'Small beginnings', body: 'Every experiment begins with a question worth tending.' }, { id: 'method', heading: 'What we tried', body: 'We watched, recorded, and changed one thing at a time.' }], published: false }),
    instructions: 'Author a structured document. save edits target section heading secondary and body text. title changes document title using text. add adds a section. reorder supplies every section ID exactly once in ids. publish marks a reviewed revision ready for local HTML and Markdown export; any subsequent edit invalidates it. Do not fabricate citations or claim external publication.',
    apply(s, d) { if (d.action === 'save') {
        const row = find(s.sections, d.target);
        row.heading = text(d.secondary);
        row.body = text(d.text);
    }
    else if (d.action === 'title')
        s.title = text(d.text);
    else if (d.action === 'add')
        s.sections.push({ id: uid(s.sections), heading: text(d.secondary), body: text(d.text) });
    else if (d.action === 'reorder') {
        check(d.ids?.length === s.sections.length && new Set(d.ids).size === s.sections.length, 'Include every section once');
        s.sections = d.ids.map(id => find(s.sections, id));
    }
    else if (d.action === 'publish') {
        check(s.sections.length, 'Add content first');
        s.published = true;
        return revised(s, 'This revision is ready to export.');
    }
    else
        unsupported(d.action); s.published = false; return revised(s, 'Draft saved.'); },
    panels: s => [form('outline', 'The shape of your story', [f('text', 'Document title', s.title)], [a('Rename', 'title')], { rows: s.sections.map((r, i) => ({ order: i + 1, id: r.id, heading: r.heading })), columns: ['order', 'id', 'heading'], extraForms: [{ title: 'Add section', fields: [f('secondary', 'Heading', 'A new thought'), f('text', 'Body', '', 'textarea')], actions: [a('Add section', 'add')] }] }), p('compose', 'Writing room', 'sections', { sections: s.sections }), p('preview', s.title, 'prose', { body: s.sections.map(r => `## ${r.heading}\n\n${r.body}`).join('\n\n'), actions: [a('Mark ready to export', 'publish', {}, 'primary')], badge: s.published ? 'ready' : 'draft', download: { name: 'document.md', text: `# ${s.title}\n\n` + s.sections.map(r => `## ${r.heading}\n\n${r.body}`).join('\n\n') }, htmlDownload: true })], smoke: { action: 'add', secondary: 'What comes next', text: 'One more experiment.' } });
export const types = define({ id: 'types', project: 'P05', title: 'Type garden', subtitle: 'Make implicit contracts visible', category: 'Develop', icon: 'λ', color: '#a7c6a0', panelIds: ['source', 'contracts', 'diagnostics'],
    stateType: `export type Contract = { id: Text; signature: Text; reason: Text }; export type Diagnostic = { id: Text; location: Text; detail: Text }; export type State = { ${COMMON} source: Text; contracts: Contract[]; diagnostics: Diagnostic[] };`,
    initial: () => initial({ source: 'function greet(person)\n  Say hello to person.name.\n\nfunction welcome(people)\n  Map greet over people and join with a newline.', contracts: [], diagnostics: [] }),
    instructions: 'Infer and review semantic types from the source. save stores source text and invalidates previous findings. infer adds target function ID, proposed TS-style signature text and evidence secondary. diagnostic adds location target and concrete mismatch text. clear discards findings. Do not claim exact static verification from a semantic judgment. Account for call sites and distinguish unknown host types using escape hatches.',
    apply(s, d) { if (d.action === 'save') {
        s.source = text(d.text);
        s.contracts = [];
        s.diagnostics = [];
    }
    else if (d.action === 'infer') {
        check(s.source.includes(text(d.target)), 'Function name is absent from source');
        const r = { id: d.target, signature: text(d.text), reason: text(d.secondary) };
        s.contracts = s.contracts.filter(r => r.id !== d.target);
        s.contracts.push(r);
    }
    else if (d.action === 'diagnostic')
        s.diagnostics.push({ id: uid(s.diagnostics), location: text(d.target), detail: text(d.text) });
    else if (d.action === 'clear') {
        s.contracts = [];
        s.diagnostics = [];
    }
    else
        unsupported(d.action); return revised(s, 'Source and semantic findings updated.'); },
    panels: s => [form('source', 'Context matters', [f('text', 'Source to analyze', s.source, 'code')], [a('Save source', 'save')], { footer: 'Ask natlang to infer a signature or explain a suspicious call using the command bar.' }), form('contracts', 'Proposed contracts', [f('target', 'Function', 'greet'), f('text', 'Signature', '(person: { name: string }) => string'), f('secondary', 'Evidence', 'Reads person.name and produces greeting text.')], [a('Record proposal', 'infer')], { rows: s.contracts, columns: ['id', 'signature', 'reason'] }), table('diagnostics', 'Semantic diagnostics', s.diagnostics, ['location', 'detail'], { actions: [a('Clear findings', 'clear')] })], smoke: { action: 'infer', target: 'greet', text: '(person: { name: string }) => string', secondary: 'Uses person.name.' } });
export default [ide, notebook, wiki, merge, evidence, publisher, types];
