import test from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
import { createLandscapeEpisodes, labels, templates, variants } from '../scripts/skills/build-research-landscape-episodes.mjs';
import { validateEpisode } from '../dist/skills/episode.js';
import { scoreResearchObjective } from '../src/skills/research-objective.ts';

const referenceAnswer = expected => ({ label: expected.label, unresolved: expected.unresolved,
  citations: expected.requiredEvidence.map(row => ({ sourceId: row.sourceId, evidence: row.text })) });

test('landscape has six families, three library conditions, three replicas, and global template holdouts', () => {
  const episodes = createLandscapeEpisodes();
  assert.equal(templates().length, 6);
  assert.equal(variants.length, 3);
  assert.equal(episodes.length, 54);
  assert.equal(episodes.reduce((n, e) => n + e.support.cases.length + e.query.cases.length, 0), 432);
  for (const episode of episodes) {
    assert.deepEqual(validateEpisode(episode), [], episode.id);
    assert.equal(episode.support.cases.length, 4);
    assert.equal(episode.query.cases.length, 4);
    assert.ok(episode.support.cases.every(row => !episode.query.cases.some(q => q.group === row.group)));
    assert.ok(episode.provenance.controlled_seed);
    assert.ok(episode.provenance.quality_note.includes('no empirical model-baseline'));
    const siblingEpisodes = episodes.filter(other => other.family === episode.family);
    assert.equal(siblingEpisodes.length, 9);
    for (const sibling of siblingEpisodes) {
      assert.deepEqual(sibling.support.cases.map(row => row.group), episode.support.cases.map(row => row.group));
      assert.deepEqual(sibling.query.cases.map(row => row.group), episode.query.cases.map(row => row.group));
    }
  }
});

test('every case is a substantial multi-document dossier with verified cross-source gold evidence', () => {
  const episodes = createLandscapeEpisodes();
  for (const episode of episodes) for (const row of [...episode.support.cases, ...episode.query.cases]) {
    const packet = JSON.parse(row.args[0]);
    const docs = row.expected.documents;
    assert.ok(packet.question.length > 80);
    assert.equal(packet.allowedLabels.join(','), labels[episode.family.split(':').at(-1)].join(','));
    assert.equal(docs.length, 10);
    const totalText = docs.reduce((n, document) => n + document.text.length, 0);
    assert.ok(totalText >= 3000 && totalText <= 10000, `${row.id} dossier text length ${totalText}`);
    assert.ok(docs.every(document => packet.catalog.some(hit => hit.id === document.id && !('text' in hit))));
    assert.equal(row.expected.requiredEvidence.length, 5);
    assert.ok(new Set(row.expected.requiredEvidence.map(evidence => evidence.sourceId)).size >= 3);
    assert.ok(row.expected.requiredEvidence.every(evidence => docs.find(d => d.id === evidence.sourceId)?.text.includes(evidence.text)));
    assert.equal(scoreResearchObjective(packet, referenceAnswer(row.expected), row.expected).quality, 1, row.id);
  }
});

test('typed search/read service returns catalog hits and privately retrieves full records', () => {
  const row = createLandscapeEpisodes()[0].support.cases[0];
  const packet = JSON.parse(row.args[0]);
  const service = row.services.research;
  const declarations = ts.transpileDeclaration(service, { fileName: 'research.ts', compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  assert.match(declarations, /search\(query: string\)/);
  assert.match(declarations, /read\(sourceId: string\)/);
  assert.ok(!declarations.includes('PRIVATE_DOSSIER'));
  assert.ok(row.expected.documents.every(document => !declarations.includes(document.text)));
  const js = ts.transpileModule(service, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const exports = {};
  new Function('exports', js)(exports);
  const hits = exports.search('governing threshold effective date incident record');
  assert.ok(hits.length > 0);
  assert.ok(hits.every(hit => packet.catalog.some(entry => entry.id === hit.id)));
  assert.equal(row.expected.documents.find(document => document.id === hits[0].id).text, exports.read(hits[0].id));
  assert.throws(() => exports.read('not-a-source'), /Unknown source ID/);
});

test('description/body variants are distinct and citationless label guesses get no evidence score', () => {
  const episodes = createLandscapeEpisodes();
  const group = episodes.filter(row => row.family.endsWith('incident-triage'));
  assert.equal(group.filter(row => row.provenance.library_variant === 'empty')[0].library.kind, 'empty');
  const helpful = group.find(row => row.provenance.library_variant === 'helpful-body-poor-description');
  const helpfulSkill = helpful.library.skills['dossier-research']['SKILL.md'];
  assert.match(helpfulSkill, /description: Miscellaneous notes\./);
  assert.match(helpfulSkill, /Search for the governing instrument/);
  const metadata = group.find(row => row.provenance.library_variant === 'description-only');
  const metadataSkill = metadata.library.skills['dossier-research']['SKILL.md'];
  assert.match(metadataSkill, /description: Search source catalogs/);
  assert.doesNotMatch(metadataSkill, /Search for the governing instrument/);
  const item = helpful.support.cases[0], packet = JSON.parse(item.args[0]);
  const guessed = scoreResearchObjective(packet, { label: item.expected.label, unresolved: item.expected.unresolved, citations: [] }, item.expected);
  assert.equal(guessed.quality, 0, 'correct classification without retrieved evidence receives no evidence credit');
});
