// Tests for the website dev server: node --test tests/test_website_dev.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotate, applyEdits, findEditableTargets, fragmentProblem } from '../scripts/website-dev/html-targets.mjs';
import { parseArgs, startServer } from '../scripts/website-dev/server.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const SAMPLE = `<!doctype html><html><head><title>t</title><style>p { color: red; }</style></head><body>
<h1>Title <em>here</em></h1>
<p id="filled-by-script"></p>
<p>First paragraph with <code>code</code>.</p>
<div class="facts"><div><b>Lead.</b> Detail <a href="#x">link</a></div></div>
<pre class="code"><p>not editable</p></pre>
<svg viewBox="0 0 1 1"><text>no</text><use href="#a"/></svg>
<p>Holds a <span id="live">value</span>.</p>
<ul><li>One <span>item</span></li><li>   </li></ul>
<script>const s = "<p>not a tag</p>";</script>
</body></html>`;

test('finds the outermost text elements and skips scripted, empty and excluded ones', () => {
  const t = findEditableTargets(SAMPLE);
  const inner = t.map(x => SAMPLE.slice(x.innerStart, x.innerEnd));
  assert.deepEqual(inner, [
    'Title <em>here</em>',
    'First paragraph with <code>code</code>.',
    '<b>Lead.</b> Detail <a href="#x">link</a>',
    'One <span>item</span>',
  ]);
  assert.deepEqual(t.map(x => x.id), [0, 1, 2, 3]);
});

test('annotate marks each target and changes nothing else', () => {
  const t = findEditableTargets(SAMPLE);
  const out = annotate(SAMPLE, t);
  assert.equal((out.match(/data-nl-edit="/g) || []).length, t.length);
  assert.equal(out.replace(/ data-nl-edit="\d+"/g, ''), SAMPLE);
  assert.match(out, /<h1 data-nl-edit="0">Title/);
});

test('applyEdits replaces only the chosen inner HTML', () => {
  const out = applyEdits(SAMPLE, [{ id: 1, html: 'Rewritten <b>text</b>.' }, { id: 3, html: 'Two' }]);
  assert.equal(out, SAMPLE.replace('First paragraph with <code>code</code>.', 'Rewritten <b>text</b>.').replace('One <span>item</span>', 'Two'));
  assert.throws(() => applyEdits(SAMPLE, [{ id: 99, html: 'x' }]), /no editable element 99/);
  assert.throws(() => applyEdits(SAMPLE, [{ id: 1, html: 'a <b>b' }]), /leaves <b> open/);
  assert.throws(() => applyEdits(SAMPLE, [{ id: 1, html: '<img src=x onerror="alert(1)">' }]), /event-handler/);
});

test('fragmentProblem accepts inline markup and rejects unsafe or unbalanced fragments', () => {
  assert.equal(fragmentProblem('a <code>b</code> <em>c</em><br> d'), null);
  assert.match(fragmentProblem('x</p>'), /unmatched <\/p>/);
  assert.match(fragmentProblem('<script>1</script>'), /script/);
});

test('the real page has editable prose and keeps its exhibits out of reach', async () => {
  const html = await readFile(join(REPO, 'website', 'index.html'), 'utf8');
  const t = findEditableTargets(html);
  assert.ok(t.length > 100, `only ${t.length} targets`);
  assert.equal(t[0].tag, 'h1');
  for (const x of t) {
    const inner = html.slice(x.innerStart, x.innerEnd);
    assert.doesNotMatch(inner, /\sid="/, 'a target contains a scripted element');
    assert.doesNotMatch(html.slice(x.openStart, x.openEnd), /\sid="/);
  }
});

test('parseArgs reads options and rejects unknown ones', () => {
  const o = parseArgs(['--port', '0', '--read-only', '--page', 'index.html']);
  assert.equal(o.port, 0);
  assert.equal(o.readOnly, true);
  assert.throws(() => parseArgs(['--nope']), /unknown option/);
  assert.throws(() => parseArgs(['--port', 'x']), /--port/);
});

test('the server serves an annotated page and saves edits back to disk', async () => {
  const root = await mkdtemp(join(tmpdir(), 'website-dev-'));
  await cp(join(REPO, 'website'), root, { recursive: true });
  const { url, close } = await startServer({ root, page: 'index.html', port: 0, host: '127.0.0.1', readOnly: false });
  try {
    const page = await (await fetch(url)).text();
    assert.match(page, /data-nl-edit="0"/);
    assert.match(page, /\/__dev\/editor\.js/);
    const version = page.match(/"version":"([0-9a-f]+)"/)[1];
    const css = await fetch(new URL('style.css', url));
    assert.equal(css.headers.get('content-type'), 'text/css; charset=utf-8');
    assert.equal((await fetch(new URL('../package.json', url))).status, 404);

    const save = body => fetch(new URL('/__dev/save', url), {
      method: 'POST', headers: { 'content-type': 'application/json', 'x-natlang-dev': '1' }, body: JSON.stringify(body),
    });
    const refused = await fetch(new URL('/__dev/save', url), { method: 'POST', body: JSON.stringify({ version, edits: [{ id: 0, html: 'x' }] }) });
    assert.equal(refused.status, 403);

    const before = await readFile(join(root, 'index.html'), 'utf8');
    const ok = await save({ version, edits: [{ id: 0, html: 'A new <em>headline.</em>' }] });
    assert.equal(ok.status, 200);
    const after = await readFile(join(root, 'index.html'), 'utf8');
    const target = findEditableTargets(before)[0];
    assert.equal(after, before.slice(0, target.innerStart) + 'A new <em>headline.</em>' + before.slice(target.innerEnd));

    const stale = await save({ version, edits: [{ id: 1, html: 'y' }] });
    assert.equal(stale.status, 409);
    assert.equal(await readFile(join(root, 'index.html'), 'utf8'), after);
  } finally {
    await close();
    await rm(root, { recursive: true, force: true });
  }
});

test('read-only mode serves the page without edit marks and refuses saves', async () => {
  const root = await mkdtemp(join(tmpdir(), 'website-dev-'));
  await writeFile(join(root, 'index.html'), SAMPLE);
  const { url, close } = await startServer({ root, page: 'index.html', port: 0, host: '127.0.0.1', readOnly: true });
  try {
    const page = await (await fetch(url)).text();
    assert.doesNotMatch(page, /data-nl-edit=/);
    assert.match(page, /"editable":false/);
    const res = await fetch(new URL('/__dev/save', url), { method: 'POST', headers: { 'x-natlang-dev': '1' }, body: '{}' });
    assert.equal(res.status, 403);
  } finally {
    await close();
    await rm(root, { recursive: true, force: true });
  }
});


test('concurrent saves from one version cannot overwrite each other', async () => {
  const root = await mkdtemp(join(tmpdir(), 'website-dev-'));
  await writeFile(join(root, 'index.html'), SAMPLE);
  const { url, close } = await startServer({ root, page: 'index.html', port: 0, host: '127.0.0.1', readOnly: false });
  try {
    const html = await (await fetch(url)).text();
    const version = html.match(/"version":"([0-9a-f]+)"/)[1];
    const save = html => fetch(new URL('/__dev/save', url), { method: 'POST',
      headers: { 'content-type': 'application/json', 'x-natlang-dev': '1' },
      body: JSON.stringify({ version, edits: [{ id: 0, html }] }) });
    const results = await Promise.all([save('first'), save('second')]);
    assert.deepEqual(results.map(r => r.status).sort(), [200, 409]);
    const winner = results[0].status === 200 ? 'first' : 'second';
    assert.match(await readFile(join(root, 'index.html'), 'utf8'), new RegExp('<h1>' + winner + '</h1>'));
    const malformed = await fetch(new URL('/__dev/save', url), { method: 'POST',
      headers: { 'x-natlang-dev': '1' }, body: 'null' });
    assert.equal(malformed.status, 400);
  } finally { await close(); await rm(root, { recursive: true, force: true }); }
});
