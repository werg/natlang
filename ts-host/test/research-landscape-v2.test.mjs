import test from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
import { buildLandscapeCase, buildResearchLandscapeEpisodes, auditLandscapeConstruction } from '../scripts/skills/build-research-landscape-v2.mjs';
import { scoreResearchObjective } from '../src/skills/research-objective.ts';
import { validateEpisode } from '../dist/skills/episode.js';

const answer = expected => ({ label: expected.label, unresolved: expected.unresolved,
  citations: expected.requiredEvidence.map(row => ({ sourceId: row.sourceId, evidence: row.text })) });

test('v2 derives labels and every cited pivot from explicit source facts', () => {
  const audit = auditLandscapeConstruction();
  assert.equal(audit.length, 24);
  for (const row of audit) {
    assert.ok(row.documents >= 4);
    assert.ok(row.pivotal >= 1, `${row.family}/${row.mode}`);
    assert.ok(row.pivotalOutcomes.every(outcome => outcome !== row.label), `${row.family}/${row.mode} pivot did not alter result`);
    const c = buildLandscapeCase(row.family, row.mode);
    assert.equal(c.expected.label, row.label);
    assert.equal(c.expected.unresolved, row.unresolved);
    assert.ok(c.expected.requiredEvidence.every(e => c.expected.documents.find(d => d.id === e.sourceId)?.text.includes(e.text)));
    assert.ok(c.expected.requiredEvidence.slice(1).every(e => c.pivotal.some(p => p.sourceId === e.sourceId && p.text === e.text)));
    assert.ok(c.pivotal.every(p => p.changedLabel !== c.expected.label), `${row.family}/${row.mode} has a noncausal evidence pivot`);
    assert.equal(scoreResearchObjective(c.packet, answer(c.expected), c.expected).quality, 1);
  }
});

test('generated episode packets hold back document bodies and do not expose mode names or labels in the brief', () => {
  const episodes = buildResearchLandscapeEpisodes();
  assert.equal(episodes.length, 54);
  assert.equal(episodes.reduce((n, e) => n + e.support.cases.length + e.query.cases.length, 0), 216);
  for (const episode of episodes) for (const row of [...episode.support.cases, ...episode.query.cases]) {
    assert.deepEqual(validateEpisode(episode), [], episode.id);
    const packet = JSON.parse(row.args[0]);
    assert.ok(!packet.question.includes(row.id.split('-').slice(1, -1).join('-')));
    assert.ok(packet.catalog.every(entry => !('text' in entry)));
    assert.ok(row.expected.documents.every(document => !packet.question.includes(document.text)));
    assert.ok(row.expected.documents.every(document => !row.args.join('\n').includes(document.text)));
    assert.ok(row.expected.documents.every(document => !episode.target.files['solve.nl'].includes(document.text)));
    assert.equal(scoreResearchObjective(packet, answer(row.expected), row.expected).quality, 1);
    const decl = ts.transpileDeclaration(row.services.research, { fileName: 'research.ts', compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
    assert.match(decl, /search\(query: string\)/);
    assert.match(decl, /read\(sourceId: string\)/);
    assert.ok(row.expected.documents.every(document => !decl.includes(document.text)));
  }
  for (const episode of episodes) {
    assert.equal(episode.support.cases.length, 2);
    assert.equal(episode.query.cases.length, 2);
    assert.ok(episode.support.cases.every(row => !episode.query.cases.some(query => query.group === row.group)));
    const familySiblings = episodes.filter(other => other.family === episode.family);
    assert.equal(familySiblings.length, 9);
    for (const sibling of familySiblings) {
      assert.deepEqual(sibling.support.cases.map(row => row.group), episode.support.cases.map(row => row.group));
      assert.deepEqual(sibling.query.cases.map(row => row.group), episode.query.cases.map(row => row.group));
    }
  }
});

test('wrong labels retain only partial evidence credit; missing evidence receives zero quality', () => {
  const row = buildResearchLandscapeEpisodes()[0].support.cases[0];
  const packet = JSON.parse(row.args[0]);
  const wrong = answer(row.expected); wrong.label = packet.allowedLabels.find(x => x !== wrong.label) ?? null;
  assert.equal(scoreResearchObjective(packet, wrong, row.expected).quality, 0.5);
  const noEvidence = answer(row.expected); noEvidence.citations = [];
  assert.equal(scoreResearchObjective(packet, noEvidence, row.expected).quality, 0);
});
