import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryResearchAdapter, ResearchWorkspace } from '../studio/shared/research-workspace.mjs';
import { auditResearchBundle } from '../studio/research/evaluation.mjs';

test('structural audit does not claim semantic success or admit training automatically', async () => {
    const adapter = new MemoryResearchAdapter(), workspace = new ResearchWorkspace(adapter);
    const manifest = await workspace.commitEdits('', {
        'evidence/rows.json': { kind: 'evidence', content: [{ requests: 100, failures: 9 }] },
        'claims/answer.json': { kind: 'claim', content: { text: 'The rate changed', supports: ['evidence/rows.json'] } },
    });
    const bundle = { format: 1, workspace: await workspace.export(), state: { head: manifest.id } };
    const report = await auditResearchBundle(bundle, 'cohort_reliability');
    assert.equal(report.structural.passed, false);
    assert.equal(report.observations.links, 1);
    assert.ok(report.semantic_review.every(row => row.result === 'unreviewed'));
    assert.match(report.training_admission, /independent semantic review/);
    const irrelevant = await auditResearchBundle(bundle, 'irrelevant_update');
    assert.equal(irrelevant.structural.passed, true);
    assert.ok(!irrelevant.structural.checks.some(row => row.name === 'actual execution receipts'));
});
