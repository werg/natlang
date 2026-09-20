/** Exact DOM projection. All model/user text is inserted through textContent. */
export function el(tag, className = '', value) { const node = document.createElement(tag); if (className)
    node.className = className; if (value !== undefined)
    node.textContent = String(value); return node; }
const append = (root, ...nodes) => { for (const node of nodes.flat())
    if (node)
        root.append(node); return root; };
export function download(name, text, type = 'text/plain') { const url = URL.createObjectURL(new Blob([text], { type })); const link = el('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
function output(value) { return el('pre', '', value); }
function prose(text) { const root = el('div', 'prose'); for (const block of String(text).split(/\n\s*\n/)) {
    const heading = /^(#{1,3}) (.*)$/.exec(block);
    root.append(heading ? el('h' + heading[1].length, '', heading[2]) : el('p', '', block));
} return root; }
function metrics(rows) { const root = el('div', 'metrics'); for (const [label, value] of rows ?? [])
    root.append(append(el('div', 'metric'), el('span', '', label), el('strong', '', value))); return root; }
function button(def, ctx, getData = () => ({}), formId) { const b = el('button', def.tone ?? '', def.label); b.type = 'button'; b.dataset.action = def.kind; b.onclick = () => ctx.dispatch(def.kind, { ...getData(), ...def.data }, formId).catch(ctx.error); return b; }
function actions(rows, ctx, getData, formId) { return append(el('div', 'form-actions'), (rows ?? []).map(row => button(row, ctx, getData, formId))); }
function makeForm(spec, ctx, id) {
    const root = el('div', 'app-form');
    const controls = new Map();
    const fields = el('div', 'fields');
    for (const field of spec.fields ?? []) {
        const label = el('label', '', field.label), control = el(field.type === 'textarea' || field.type === 'code' ? 'textarea' : field.type === 'select' ? 'select' : 'input', field.type === 'code' ? 'code' : '');
        control.id = `${ctx.app}-${id}-${field.name}`;
        label.htmlFor = control.id;
        if (field.type === 'select')
            for (const option of field.options ?? [])
                control.add(new Option(option, option));
        else if (control.tagName === 'INPUT')
            control.type = field.type === 'number' ? 'number' : 'text';
        const key = `${id}/${field.name}`;
        control.value = ctx.drafts[key] ?? String(field.value ?? '');
        control.name = field.name;
        label.dataset.draft = String(control.value !== String(field.value ?? ''));
        control.oninput = () => { label.dataset.draft = String(control.value !== String(field.value ?? '')); ctx.draft(key, control.value); };
        if (field.type === 'code')
            control.onkeydown = event => { if (event.key === 'Tab') {
                event.preventDefault();
                control.setRangeText('  ', control.selectionStart, control.selectionEnd, 'end');
                control.oninput();
            } };
        controls.set(field.name, { control, field });
        label.append(control);
        fields.append(label);
    }
    const data = () => Object.fromEntries([...controls].map(([name, { control, field }]) => [name, name === 'ids' ? control.value.split(',').map(x => x.trim()).filter(Boolean) : field.type === 'number' ? Number(control.value) : control.value]));
    append(root, fields, actions(spec.actions, ctx, data, id));
    return root;
}
function dataTable(spec, ctx) { if (!spec.rows?.length)
    return el('div', 'empty', spec.empty ?? 'Nothing here yet. Your next step will leave a trace.'); const columns = spec.columns ?? Object.keys(spec.rows[0]); const scroll = el('div', 'table-scroll'), table = el('table'), head = el('tr'); for (const column of columns)
    head.append(el('th', '', column.replaceAll('_', ' '))); if (spec.rowActions)
    head.append(el('th', '', 'Actions')); table.append(append(el('thead'), head)); const body = el('tbody'); for (const row of spec.rows) {
    const tr = el('tr');
    for (const column of columns)
        tr.append(el('td', '', Array.isArray(row[column]) ? row[column].join(', ') : typeof row[column] === 'object' ? JSON.stringify(row[column]) : row[column] ?? ''));
    if (spec.rowActions)
        tr.append(append(el('td'), spec.rowActions.map(a => button(a, ctx, () => ({ target: row.id })))));
    body.append(tr);
} return append(scroll, append(table, body)); }
function bar(value, max, kind = '') { const node = el('div', 'bar ' + kind), fill = el('i'); fill.style.width = `${Math.max(0, Math.min(100, value / max * 100))}%`; node.append(fill); return node; }
export function renderPanel(spec, ctx) {
    const root = el('section', `panel panel-${spec.kind}`);
    root.dataset.panel = spec.id;
    const heading = el('div', 'panel-heading');
    append(heading, el('h2', '', spec.title), spec.badge ? el('span', 'badge', spec.badge) : null);
    root.append(heading);
    if (spec.chat) {
        const chat = el('div', 'chat');
        for (const row of spec.chat)
            chat.append(append(el('div', `bubble ${row.speaker}`), el('small', '', row.speaker), el('div', '', row.text), row.evidence.length ? el('small', '', `Memories: ${row.evidence.join(', ')}`) : null));
        root.append(chat);
    }
    if (spec.terminal) {
        const log = el('div', 'terminal-log');
        if (!spec.terminal.length)
            log.append(el('pre', '', 'Your session is ready.'));
        for (const row of spec.terminal)
            append(log, el('div', 'prompt', `› ${row.command} · ${row.status}`), output(row.output));
        root.append(log);
    }
    if (spec.kind === 'form')
        root.append(makeForm(spec, ctx, spec.id));
    if (spec.kind === 'prose')
        root.append(prose(spec.body));
    if (spec.kind === 'cards') {
        if (!spec.items.length)
            root.append(el('div', 'empty', spec.empty ?? 'Your collection will grow here.'));
        for (const item of spec.items)
            root.append(append(el('div', 'extra-form'), el('h3', '', item.title), el('p', '', item.body), actions(item.actions, ctx)));
    }
    if (spec.kind === 'cells')
        for (const cell of spec.cells) {
            const box = el('article', 'cell');
            append(box, append(el('div', 'cell-head'), el('strong', '', cell.id), el('span', 'badge', `${cell.engine} · ${cell.status}`)), makeForm({ fields: [{ name: 'text', label: 'Cell source', value: cell.source, type: 'code' }], actions: [{ label: 'Save cell', kind: 'save', data: { target: cell.id } }, { label: 'Run this cell', kind: 'execute', data: { target: cell.id } }, { label: 'Run with dependencies', kind: 'run', data: { target: cell.id }, tone: 'primary' }, { label: 'Remove', kind: 'remove', data: { target: cell.id } }] }, ctx, `cell-${cell.id}`), el('p', 'panel-footer', `Depends on: ${cell.needs.join(', ') || 'nothing'}`));
            if (cell.result)
                box.append(append(el('div', 'cell-output'), el('span', 'badge', 'Output preview'), output(cell.result)));
            if(cell.result_id){const save=el('button','','Download full result');save.onclick=()=>ctx.downloadValue(cell.result_id).catch(ctx.error);box.append(save);}
        root.append(box);
        }
    if (spec.kind === 'sections')
        for (const section of spec.sections)
            root.append(append(el('article', 'cell'), makeForm({ fields: [{ name: 'secondary', label: 'Heading', value: section.heading }, { name: 'text', label: 'Body', type: 'textarea', value: section.body }], actions: [{ label: 'Save section', kind: 'save', data: { target: section.id } }] }, ctx, section.id)));
    if (spec.kind === 'market') {
        const cards = el('div', 'market-cards');
        for (const m of spec.merchants)
            cards.append(append(el('article', 'merchant'), el('div', 'avatar', '◒'), el('h3', '', m.name), el('p', '', `${m.cash} coins · ${m.apples} apples`), el('p', '', `${m.price} coins per apple`)));
        root.append(cards);
    }
    if (spec.kind === 'arena') {
        const arena = el('div', 'arena');
        spec.fighters.forEach((f, i) => { if (i)
            arena.append(el('span', 'versus', 'vs.')); arena.append(append(el('div', 'fighter'), el('div', 'fighter-icon', i ? '♜' : '♞'), el('h3', '', f.id), bar(f.hp, 100), el('p', '', `${f.hp} health`), bar(f.energy, 10, 'energy'), el('p', '', `${f.energy} energy · ${f.stance}`))); });
        root.append(arena);
    }
    if (spec.kind === 'spell-arena') {
        root.append(append(el('div', 'spell-stage'), append(el('div', 'fighter'), el('div', 'spell-orb', '✧'), bar(spec.mana, 40, 'energy'), el('p', '', `${spec.mana} mana`)), append(el('div', 'fighter'), el('div', 'fighter-icon', '♜'), bar(spec.hp, 100), el('p', '', `${spec.hp} sentinel health`))));
        root.append(el('p', 'panel-footer', spec.last));
    }
    if (spec.kind === 'scene')
        root.append(append(el('div', 'scene'), el('h3', '', spec.body), el('p', '', spec.detail)));
    if (spec.kind === 'workflow') {
        const flow = el('div', 'workflow');
        for (const step of spec.steps)
            flow.append(append(el('div', `flow-step ${step.status}`), append(el('div'), el('strong', '', step.title), el('small', '', step.needs.length ? `after ${step.needs.join(', ')}` : 'entry point')), el('span', 'badge', step.status)));
        root.append(flow);
    }
    if (spec.kind === 'timeline') {
        root.append(el('p', '', `${spec.date} · UTC`));
        const timeline = el('div', 'timeline');
        for (let hour = 0; hour < 24; hour++) {
            const slot = el('div', 'time-slot');
            for (const task of spec.tasks.filter(t => t.start && t.start.startsWith(spec.date) && new Date(t.start).getUTCHours() === hour))
                slot.append(append(el('div', `appointment ${task.done ? 'done' : ''}`), el('strong', '', task.title), el('small', '', `${task.start.slice(11, 16)} · ${task.duration} min`)));
            append(timeline, el('div', 'time-label', `${String(hour).padStart(2, '0')}:00`), slot);
        }
        root.append(timeline);
        queueMicrotask(() => { timeline.scrollTop = 8 * 64; });
    }
    if (spec.kind === 'chart') {
        if (!spec.rows.length)
            root.append(el('div', 'empty', 'Run a matched comparison to see the shape of your results.'));
        const max = Math.max(1, ...spec.rows.map(r => Math.abs(r.value)));
        for (const row of spec.rows) {
            const track = el('div', 'chart-track'), fill = el('i', row.value < 0 ? 'negative' : '');
            fill.style.width = `${Math.abs(row.value) / max * 100}%`;
            track.append(fill);
            root.append(append(el('div', 'chart-row'), el('span', '', row.label), track, el('strong', '', row.value)));
        }
    }
    if (spec.kind === 'diff')
        root.append(append(el('div', 'diff'), output(spec.before), output(spec.after)));
    if (spec.kind === 'media') {
        if (spec.asset) {
            const video = el('video', 'video-preview');
            video.controls = true;
            video.preload = 'metadata';
            video.src = ctx.assetUrl(spec.asset);
            root.append(video);
        }
        else
            root.append(el('div', 'empty', 'Create a short sample clip or import an MP4 to begin.'));
        if (spec.upload) {
            const label = el('label', 'extra-form', 'Import an MP4'), input = el('input');
            input.type = 'file';
            input.accept = 'video/mp4';
            input.onchange = () => { if (input.files[0])
                ctx.upload(input.files[0]).catch(ctx.error); };
            append(root, append(label, input));
        }
    }
    if (spec.metrics)
        root.append(metrics(spec.metrics));
    if (spec.output)
        root.append(output(spec.output));
    if (spec.rows && spec.kind !== 'chart')
        root.append(dataTable(spec, ctx));
    if (spec.trace?.length) {
        const details = el('details');
        details.append(el('summary', '', `${spec.trace.length} reduction events`));
        spec.trace.forEach((row, i) => details.append(output(`${i + 1}. ${row}`)));
        root.append(details);
    }
    for (const [i, sub] of (spec.extraForms ?? []).entries())
        root.append(append(el('div', 'extra-form'), el('h3', '', sub.title), makeForm(sub, ctx, `${spec.id}-extra-${i}`)));
    if (spec.kind !== 'form' && spec.actions)
        root.append(actions(spec.actions, ctx));
    if (spec.download) {
        const b = el('button', '', 'Download ' + spec.download.name);
        b.onclick = () => download(spec.download.name, spec.download.text);
        root.append(append(el('div', 'form-actions'), b));
    }
    if (spec.htmlDownload) {
        const b = el('button', '', 'Download HTML');
        b.onclick = () => { const doc = document.implementation.createHTMLDocument(spec.title); doc.body.append(prose(spec.body)); download('document.html', '<!doctype html>' + doc.documentElement.outerHTML, 'text/html'); };
        root.append(b);
    }
    if (spec.footer)
        root.append(el('p', 'panel-footer', spec.footer));
    return root;
}
export function renderPanels(root, panels, view, ctx) { const ids = panels.map(p => p.id); if (view.focus.length !== ids.length || new Set(view.focus).size !== ids.length || view.focus.some(id => !ids.includes(id)))
    throw new Error('View must include every application panel exactly once'); const active = document.activeElement?.id; root.replaceChildren(...view.focus.map(id => renderPanel(panels.find(p => p.id === id), ctx))); if (active)
    document.getElementById(active)?.focus({ preventScroll: true }); }
