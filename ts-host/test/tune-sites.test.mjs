import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { assignSites } from '../dist/tune/sites.js';
import { scaffoldManifest } from '../dist/tune/collect.js';
import { invalidation, trainingScaffoldKey } from '../dist/tune/keys.js';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const header = "import { nl } from '@natlang/node';\n";
const site = (name, text, type = 'string') => `export function ${name}(value: string) { return nl<${type}>\`${text} \${value}.\`(); }\n`;

function programOf(source, extra = {}) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-sites-'));
  try {
    writeFileSync(join(root, 'package.json'), '{"type":"module"}');
    writeFileSync(join(root, 'main.ts'), source);
    for (const [path, text] of Object.entries(extra)) writeFileSync(join(root, path), text);
    const built = host.buildProject({ project: root, programId: 'sites', write: false, surfaceSpecifiers: ['@natlang/node'] });
    assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
    return built.manifest.adaptation;
  } finally { rmSync(root, { recursive: true, force: true }); }
}
const inline = program => program.components.filter(component => component.origin === 'authored-inline');

test('an edit elsewhere in the file keeps inline sites; training receipts survive, the release is resealed', () => {
  const before = programOf(header + site('a', 'Shout') + site('b', 'Whisper'));
  const first = assignSites(before);
  const after = programOf(header + '// a comment above everything\n' + site('a', 'Shout') + site('b', 'Whisper'));
  assert.notDeepEqual(inline(after).map(c => c.key).sort(), inline(before).map(c => c.key).sort(), 'offset-derived keys changed');
  const second = assignSites(after, first.registry);
  assert.deepEqual(second.migrations.map(m => m.how), ['moved', 'moved']);
  assert.deepEqual(Object.values(second.sites).sort(), Object.values(first.sites).sort(), 'both sites keep their IDs');
  const old = scaffoldManifest(before, base, { sites: first.sites }), now = scaffoldManifest(after, base, { sites: second.sites });
  assert.deepEqual(now.bindings.map(b => b.id).sort(), old.bindings.map(b => b.id).sort());
  assert.equal(trainingScaffoldKey(now), trainingScaffoldKey(old), 'training is reused');
  const change = invalidation(old, now);
  assert.equal(change.rekeyed.length, 2);
  assert.equal(change.fresh, false, 'the release still has to be resealed for the new component keys');
  assert.deepEqual(change.dirty, []);
});

test('an edited instruction continues its site only when the match is unique', () => {
  const before = programOf(header + site('a', 'Shout') + site('b', 'Count', 'number'));
  const first = assignSites(before);
  const edited = programOf(header + site('a', 'Shout loudly') + site('b', 'Count', 'number'));
  const second = assignSites(edited, first.registry);
  assert.deepEqual(second.migrations.map(m => m.how).sort(), ['edited', 'moved']);
  const old = scaffoldManifest(before, base, { sites: first.sites }), now = scaffoldManifest(edited, base, { sites: second.sites });
  const change = invalidation(old, now);
  assert.equal(change.changed.length, 1, 'the edited site is a semantic change of the same binding');
  assert.equal(change.added.length + change.removed.length, 0);
  // Sites in different enclosing functions differ in contract (visible bindings), so each edit above was unique.
  // Two edited sites with one contract cannot be told apart: both become new sites, nothing is inherited.
  const synthetic = (texts) => ({ components: texts.map((text, index) => ({ key: `p::f.ts::exact:${text}`, origin: 'authored-inline',
    source: { path: 'f.ts', start: index * 10, end: index * 10 + 5 }, baselineHash: text, contractHash: 'same' })) });
  const t1 = assignSites(synthetic(['shout', 'whisper']));
  const both = assignSites(synthetic(['yell', 'murmur']), t1.registry);
  assert.deepEqual(both.migrations, []);
  assert.equal(both.unmatched.length, 2);
  for (const id of Object.values(both.sites)) assert.ok(!Object.values(t1.sites).includes(id), 'no ambiguous inheritance');
  // Identical twins that only moved pair in source order.
  const dup = assignSites(synthetic(['same', 'same2']));
  const moved = assignSites({ components: synthetic(['same', 'same2']).components.map(c => ({ ...c, key: c.key + '@v2' })) }, dup.registry);
  assert.deepEqual(moved.migrations.map(m => m.how), ['moved', 'moved']);
});

test('explicit migrations carry renamed definitions and ambiguous edits; stale ones are errors', () => {
  const before = programOf(header + site('a', 'Shout') + site('b', 'Whisper'));
  const first = assignSites(before);
  const after = programOf(header + site('a', 'Yell') + site('b', 'Murmur'));
  const [yell] = inline(after).filter(c => c.baseline.template.segments.join('').includes('Yell'));
  const shoutSite = first.registry.entries.find(e => before.components.find(c => c.key === e.component).baseline.template.segments.join('').includes('Shout')).site;
  const migrated = assignSites(after, first.registry, { [shoutSite]: yell.key });
  assert.equal(migrated.sites[yell.key], shoutSite);
  assert.deepEqual(migrated.migrations.map(m => m.how), ['explicit', 'edited'], 'the remaining site then matches uniquely');
  assert.throws(() => assignSites(after, first.registry, { [shoutSite]: 'gone' }), /update or remove that entry/);
  assert.throws(() => assignSites(after, first.registry, { nosuch: yell.key }), /not in the site registry/);
});

test('a named definition moves only by an explicit migration', () => {
  const named = '---\nargs: { value: string }\nreturns: string\n---\nReturn value.\n';
  const before = programOf("import echo from './echo.nl'; export default echo;\n", { 'echo.nl': named });
  const first = assignSites(before);
  const after = programOf("import echo from './repeat.nl'; export default echo;\n", { 'repeat.nl': named });
  const second = assignSites(after, first.registry);
  assert.deepEqual(second.migrations, [], 'a renamed file is a new site unless migrated');
  const [old] = first.registry.entries, [now] = second.registry.entries;
  assert.equal(assignSites(after, first.registry, { [old.site]: now.component }).sites[now.component], old.site);
});
