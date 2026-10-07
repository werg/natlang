import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = resolve(import.meta.dirname, '../..');
const builder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v15.mjs');
const v14Builder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-source-worlds-v14.mjs');

test('V15 builder emits 12 independent four-pass worlds with scoped evidence and justified revisions', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'semantic-iterate-v15-'));
  const out = join(temp, 'candidate');
  const v14Out = join(temp, 'v14');
  try {
    execFileSync(process.execPath, [v14Builder, '--out', v14Out], { cwd: repo, stdio: 'pipe' });
    execFileSync(process.execPath, [builder, '--out', out], { cwd: repo, stdio: 'pipe' });
    const [sourceText, reviewText, lineageText] = await Promise.all([
      readFile(join(out, 'source.cases.jsonl'), 'utf8'),
      readFile(join(out, 'source-quality-review.json'), 'utf8'),
      readFile(join(out, 'lineage.json'), 'utf8'),
    ]);
    const cases = sourceText.trimEnd().split('\n').map(JSON.parse);
    const review = JSON.parse(reviewText);
    const lineage = JSON.parse(lineageText);
    const oldRows = (await readFile(join(v14Out, 'source.cases.jsonl'), 'utf8')).trimEnd().split('\n').map(JSON.parse);
    const oldGroups = new Set(oldRows.flatMap(row => row.source_groups));

    assert.equal(cases.length, 12);
    assert.equal(new Set(cases.map(row => row.source_groups[0])).size, 12);
    assert.deepEqual(cases.map(row => row.split), ['train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test']);
    assert.equal(cases.every(row => !oldGroups.has(row.source_groups[0])), true);
    assert.equal(cases.every(row => row.curriculum.reference.children.length === 4), true);
    assert.equal(cases.every(row => row.curriculum.iterate === 'required'), true);
    assert.equal(cases.every(row => row.curriculum.reference.children.every(child => child.calls[0][0] === 'read_file')), true);
    assert.equal(review.counts.independent_factual_worlds, 12);
    assert.equal(review.counts.passes, 48);
    assert.equal(review.counts.justified_revisions, 12);
    assert.equal(lineage.world_count, 12);
    assert.equal(lineage.source_sha256, review.source_cases_sha256);

    for (const [index, row] of cases.entries()) {
      const task = JSON.parse(row.semantics.folder_files['task.json']);
      const outputFields = Object.keys(task.output_contract.fields);
      assert.ok(outputFields.length >= 4);
      assert.match(task.output_contract.format, new RegExp(`exactly these ${outputFields.length} string fields`));
      assert.deepEqual(Object.keys(row.semantics.expected).sort(), [...outputFields].sort());
      assert.deepEqual(Object.keys(row.semantics.expected_files), Object.keys(row.semantics.folder_files));
      assert.equal(task.passes.length, 4);
      assert.equal(row.curriculum.reference.children.length, 4);
      assert.equal(row.split, index % 2 === 0 ? 'train' : 'test');
      assert.ok(row.source_groups[0].startsWith('v15:'));
      assert.equal(row.curriculum.reference.children.every((child, pass) =>
        child.calls[0][1].path === task.passes[pass].evidence_path), true);

      let prior = task.initialDraft;
      for (const [passIndex, pass] of task.passes.entries()) {
        const next = row.curriculum.reference.children[passIndex].calls[1][1].value;
        const allowed = new Set(pass.allowed_fields);
        assert.deepEqual(Object.keys(next).sort(), [...outputFields].sort());
        for (const field of outputFields) {
          if (!allowed.has(field)) assert.equal(next[field], prior[field], `${row.id} pass ${passIndex + 1} preserves ${field}`);
          assert.equal(typeof next[field], 'string');
        }
        assert.ok(row.semantics.folder_files[pass.evidence_path]);
        assert.ok(row.semantics.expected_files[pass.evidence_path] === row.semantics.folder_files[pass.evidence_path]);
        prior = next;
      }
      for (const field of outputFields) {
        const description = task.output_contract.fields[field];
        const values = [task.initialDraft[field], ...row.curriculum.reference.children.map(child => child.calls[1][1].value[field])];
        if (description.includes('USD value with a dollar sign'))
          assert.equal(values.every(value => /^\$\d{1,3}(,\d{3})*\.\d{2}$/.test(value)), true, `${row.id}.${field} USD format`);
        if (description.includes('comma grouping every three digits'))
          assert.equal(values.every(value => /^\d{1,3}(,\d{3})*$/.test(value)), true, `${row.id}.${field} grouped integer format`);
        if (description.includes('one ASCII space, then m'))
          assert.equal(values.every(value => /^\d+\.\d m$/.test(value)), true, `${row.id}.${field} meter format`);
      }
      assert.deepEqual(prior, row.semantics.expected);

      const audit = review.worlds[index];
      const revision = audit.justified_revision;
      const before = revision.pass === 1 ? task.initialDraft : row.curriculum.reference.children[revision.pass - 2].calls[1][1].value;
      const revised = row.curriculum.reference.children[revision.pass - 1].calls[1][1].value;
      assert.notEqual(before[revision.field], revised[revision.field]);
      assert.ok(task.passes[revision.pass - 1].allowed_fields.includes(revision.field));
      assert.ok(revision.reason.length > 20);
      assert.ok(audit.calculation_or_decision_derivation.length > 40);
      assert.equal(task.output_contract.initial_state.includes('incomplete working placeholder'), true);
      assert.match(row.curriculum.reference.root[0][1].code, /iterateOn\(revise/);
      assert.match(row.curriculum.reference.root[0][1].code, /currentDraft/);
    }

    const bySlug = slug => cases.find(row => row.curriculum.shape.includes(slug));
    const notice = bySlug('public_notice_correction_deadline');
    assert.match(notice.semantics.folder_files['pass-02-publication-log.md'], /corrected notice replaced it on 15 May 2028/);
    assert.match(notice.semantics.folder_files['pass-03-notice-rule.md'], /excluding both event dates/);
    assert.equal(notice.semantics.expected.interveningDays, '6');
    assert.equal(notice.semantics.expected.waiverStatus, 'none');

    const grant = bySlug('community_grant_match_cap');
    assert.equal(grant.semantics.expected.allowableCosts, '$48,000.00');
    assert.equal(grant.semantics.expected.fundraisingAmount, '$4,000.00');
    assert.equal(grant.curriculum.reference.children[0].calls[1][1].value.fundraisingAmount, '$4,000.00');
    assert.equal(grant.semantics.expected.matchCap, '$24,000.00');
    assert.equal(grant.semantics.expected.awardCeiling, '$26,000.00');
    assert.equal(grant.semantics.expected.awardAmount, '$24,000.00');

    const demand = bySlug('demand_response_meter_settlement');
    assert.equal(demand.semantics.expected.reductionPercent, '15.00%');
    assert.equal(demand.curriculum.reference.children[1].calls[1][1].value.baselineKwh, '340/3');
    assert.equal(demand.semantics.expected.requiredReductionPercent, '15.00%');
    assert.notEqual(demand.curriculum.reference.children[1].calls[1][1].value.reductionPercent,
      demand.curriculum.reference.children[2].calls[1][1].value.reductionPercent);
    assert.equal(demand.curriculum.reference.children[2].calls[1][1].value.referenceReadsKwh, '120, 120, 120');
    assert.equal(demand.semantics.expected.decision, 'pay');

    const transit = bySlug('transit_mileage_reimbursement');
    assert.equal(transit.semantics.expected.eligibleMiles, '82');
    assert.equal(transit.semantics.expected.reimbursement, '$54.94');
    assert.equal(transit.semantics.expected.reimbursementCap, '$60.00');

    const service = bySlug('service_amendment_payment').semantics.expected;
    assert.equal(12 * 95, 1140);
    assert.equal(11 * 95, 1045);
    assert.equal(service.acceptedVisits, '11');
    assert.equal(service.payableAmount, '$1,045.00');
    assert.equal(service.decision, 'release');

    const archive = bySlug('archive_channel_consent').semantics.expected;
    assert.equal(archive.permittedChannels, 'reading room');
    assert.equal(archive.requestedChannel, 'public web');
    const archiveChannelFormat = JSON.parse(bySlug('archive_channel_consent').semantics.folder_files['task.json']).output_contract.fields;
    assert.match(archiveChannelFormat.permittedChannels, /exactly “public web” and “reading room”/);
    assert.match(archiveChannelFormat.permittedChannels, /normalize “reading-room” to “reading room”/);
    assert.doesNotMatch(archiveChannelFormat.permittedChannels, /values joined by values joined by/);
    assert.equal(archive.decision, 'withhold');

    const routeCase = bySlug('accessible_route_substitution');
    const route = routeCase.semantics.expected;
    assert.ok(7.5 <= 8.0 && 1.4 >= 1.2);
    assert.match(routeCase.semantics.folder_files['pass-02-inspection.md'], /step-free alternate Ramp B/);
    assert.equal(route.route, 'Ramp B');
    assert.equal(route.decision, 'confirm');

    const procurement = bySlug('procurement_eligible_bid').semantics.expected;
    assert.equal(procurement.vendor, 'Beacon Supply');
    assert.equal(procurement.decision, 'award');

    const course = bySlug('course_accommodation_version').semantics.expected;
    assert.equal(course.courseVersion, 'C8');
    assert.equal(course.coveredVersion, 'C8');
    assert.equal(course.addendumId, 'AC-73A');
    assert.equal(course.leadDays, '6');
    assert.equal(course.minimumLeadDays, '5');
    assert.match(JSON.parse(bySlug('course_accommodation_version').semantics.folder_files['task.json']).output_contract.fields.support, /one semicolon followed by one ASCII space/);
    assert.equal(course.decision, 'approve');

    const permit = bySlug('protected_species_field_permit').semantics.expected;
    assert.equal(permit.species, 'blue marsh orchid');
    assert.equal(permit.decision, 'deny');
    const permitRule = JSON.parse(bySlug('protected_species_field_permit').semantics.folder_files['task.json']).output_contract.decision_rule;
    assert.match(permitRule, /species is listed, the seasonal closure applies to the same site and field date, and no signed board exception/);

    const water = bySlug('drought_household_allocation').semantics.expected;
    assert.equal(4000 * 0.7 + 250, 3050);
    assert.equal(water.authorizedLiters, '3,050');
    assert.equal(water.districtCapLiters, '3,100');
    assert.equal(water.decision, 'issue');

    const vaccine = bySlug('vaccine_logger_correction').semantics.expected;
    assert.equal(15 + 23 + 12, 50);
    assert.equal(vaccine.correctedExposureMinutes, '50');
    assert.equal(vaccine.decision, 'quarantine');

    assert.equal(cases.every(row => {
      const fields = Object.keys(JSON.parse(row.semantics.folder_files['task.json']).output_contract.fields);
      return fields.every(field => !/^(calculation|finding|noticeFinding|comparison)$/.test(field));
    }), true);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
