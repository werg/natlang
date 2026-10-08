import { caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';
import { selectionContract } from './semantic-iterate-reducer-selection-contract.mjs';

// The scenarios are individually authored facts. Each group contains two
// different requests in one decision domain; both inherit the domain split.
const domains = [
  {
    slug: 'flood-pump-deployment', domain: 'flood response pump deployment', split: 'train',
    owner: 'Regional flood desk', action: 'dispatch', noAction: 'no_action', direction: 'desc', selectCount: 1,
    metricName: 'households protected', metricFormat: 'households as digits; none when no item qualifies',
    rule: 'A pump plan qualifies only when the water-level alert is verified, the pump passed its load test, and a rested certified crew is assigned. Rank qualifying plans by the measured number of households protected. Dispatch requires the incident commander release.',
    heading: ['Pump response register', 'Water alert, equipment, and crew review', 'Incident commander release'],
    exception: 'A high household estimate cannot replace a verified alert, passed load test, or rested certified crew.',
    scenarios: [
      { id: 'FLD-510', auth: true, authority: 'The incident commander signed release FLD-510 at 07:40.', rows: [
        ['PUMP-NORTH', 410, true, true, true, 'Water rose above the local alert threshold; the pump passed its load test; crew C-4 is certified and completed its rest period.'],
        ['PUMP-EAST', 620, true, false, true, 'Water rose above the local alert threshold; the pump load test failed; crew C-2 is certified and rested.'],
        ['PUMP-WEST', 385, true, true, true, 'Water rose above the local alert threshold; the pump passed its load test; crew C-8 is certified and rested.'],
      ] },
      { id: 'FLD-511', auth: false, authority: 'No incident commander release is recorded for FLD-511; the release field is blank.', rows: [
        ['PUMP-LOWER', 730, true, true, false, 'The alert threshold was exceeded; the pump passed its load test; assigned crew C-5 has not completed its rest period.'],
        ['PUMP-CANAL', 520, true, true, true, 'The alert threshold was exceeded; the pump passed its load test; crew C-7 is certified and rested.'],
        ['PUMP-HILL', 860, false, true, true, 'The alert threshold was not reached; the pump passed its load test; crew C-9 is certified and rested.'],
      ] },
    ],
    eligible: r => r.alert && r.tested && r.crewReady,
    request: id => `Select the response pump plan for flood event ${id}.`,
    metric: rows => rows.length ? String(rows[0].metric) : 'none',
    register: r => `${r.id}: measured households protected ${r.metric}.`,
  },
  {
    slug: 'accessible-transit-detour', domain: 'accessible transit detour planning', split: 'train',
    owner: 'City transit operations', action: 'publish', noAction: 'no_action', direction: 'desc', selectCount: 2,
    metricName: 'scheduled riders served', metricFormat: 'rider counts for selected route IDs in the same order, separated by "; "; none when empty',
    rule: 'A detour qualifies only when its full path is accessible, the road authority has cleared the route, and a trained driver is assigned. Select up to two by scheduled riders served, greatest first; if fewer than two qualify, select all that qualify. Publishing requires the transit duty manager approval.',
    heading: ['Detour service register', 'Path, clearance, and staffing check', 'Duty manager approval'],
    exception: 'A larger ridership estimate cannot waive an inaccessible path, missing road clearance, or untrained driver.',
    scenarios: [
      { id: 'TRN-520', auth: true, authority: 'Transit duty manager approved publication for TRN-520.', rows: [
        ['ROUTE-K2', 940, true, true, true, 'The whole detour path is wheelchair accessible; road clearance is recorded; driver D-12 completed detour training.'],
        ['ROUTE-K4', 1110, true, true, true, 'The whole detour path is wheelchair accessible; road clearance is recorded; driver D-18 completed detour training.'],
        ['ROUTE-K7', 1250, false, true, true, 'A curb ramp is missing on the detour; road clearance is recorded; driver D-19 completed detour training.'],
      ] },
      { id: 'TRN-521', auth: false, authority: 'Duty manager approval for TRN-521 is pending and has not been recorded.', rows: [
        ['ROUTE-M1', 760, true, true, true, 'The path is accessible end to end; road clearance is recorded; driver D-22 completed detour training.'],
        ['ROUTE-M3', 810, true, false, true, 'The path is accessible end to end; the road authority has not cleared the route; driver D-25 completed detour training.'],
        ['ROUTE-M5', 690, true, true, true, 'The path is accessible end to end; road clearance is recorded; driver D-27 completed detour training.'],
      ] },
    ],
    eligible: r => r.accessible && r.cleared && r.trained,
    request: id => `Choose accessible detours for service window ${id}.`,
    metric: rows => rows.length ? rows.map(r => String(r.metric)).join('; ') : 'none',
    register: r => `${r.id}: scheduled riders on the detour ${r.metric}.`,
  },
  {
    slug: 'archive-digitization-queue', domain: 'archival digitization queue', split: 'train',
    owner: 'Regional archives digitization desk', action: 'queue', noAction: 'no_action', direction: 'desc', selectCount: 1,
    metricName: 'preservation urgency score', metricFormat: 'score for the selected collection as digits; none when empty',
    rule: 'A collection qualifies only when digitization rights are cleared, a condition survey is complete, and a preservation master does not already exist. Select one qualifying collection with the highest measured preservation urgency score. Starting digitization requires the archivist of record authorization.',
    heading: ['Collection urgency register', 'Rights, survey, and master-copy review', 'Archivist authorization'],
    exception: 'Urgency does not override uncleared rights, an incomplete survey, or an existing preservation master.',
    scenarios: [
      { id: 'ARC-530', auth: true, authority: 'The archivist of record authorized queue ARC-530.', rows: [
        ['COLL-A12', 88, true, true, false, 'Rights are cleared for digitization; the condition survey is complete; no preservation master exists.'],
        ['COLL-A19', 97, false, true, false, 'Rights review remains open; the condition survey is complete; no preservation master exists.'],
        ['COLL-A24', 75, true, true, false, 'Rights are cleared for digitization; the condition survey is complete; no preservation master exists.'],
      ] },
      { id: 'ARC-531', auth: false, authority: 'The archivist of record has not authorized ARC-531; approval is pending.', rows: [
        ['COLL-B03', 91, true, true, true, 'Rights are cleared; the condition survey is complete; a preservation master already exists.'],
        ['COLL-B08', 83, false, true, false, 'Rights review remains open; the condition survey is complete; no preservation master exists.'],
        ['COLL-B11', 96, true, false, true, 'Rights are cleared; the condition survey is not yet complete; no preservation master exists.'],
      ] },
    ],
    eligible: r => r.rights && r.survey && !r.master,
    request: id => `Set the next archival digitization collection for request ${id}.`,
    metric: rows => rows.length ? String(rows[0].metric) : 'none',
    register: r => `${r.id}: measured preservation urgency ${r.metric} on the 1–100 scale.`,
  },
  {
    slug: 'broadband-build-site', domain: 'rural broadband construction site selection', split: 'train',
    owner: 'Rural connectivity program office', action: 'fund', noAction: 'no_action', direction: 'asc', selectCount: 1,
    metricName: 'cost per verified household passed', metricFormat: 'credits per household as digits; none when empty',
    rule: 'A site qualifies when the easement is signed, backhaul capacity is reserved, and the household count has been field verified. Select one qualifying site with the lowest documented cost per verified household passed. Funding requires a recorded program allocation.',
    heading: ['Construction site cost register', 'Easement, capacity, and household verification', 'Program allocation'],
    exception: 'A low modeled cost does not substitute for a signed easement, reserved capacity, or a field-verified household count.',
    scenarios: [
      { id: 'BB-540', auth: true, authority: 'A program allocation is recorded for BB-540.', rows: [
        ['SITE-PINE', 640, true, true, true, 'The land easement is signed; backhaul capacity is reserved; the listed household count was verified in the field.'],
        ['SITE-MILL', 510, true, false, true, 'The land easement is signed; backhaul capacity is not reserved; the household count was verified in the field.'],
        ['SITE-GLEN', 720, true, true, true, 'The land easement is signed; backhaul capacity is reserved; the listed household count was verified in the field.'],
      ] },
      { id: 'BB-541', auth: false, authority: 'No program allocation is recorded for BB-541.', rows: [
        ['SITE-ASH', 580, true, true, true, 'The land easement is signed; backhaul capacity is reserved; the household count was verified in the field.'],
        ['SITE-REEF', 430, false, true, true, 'The land easement is unsigned; backhaul capacity is reserved; the household count was verified in the field.'],
        ['SITE-ORCHARD', 670, true, true, false, 'The land easement is signed; backhaul capacity is reserved; household totals are estimates and have not been field verified.'],
      ] },
    ],
    eligible: r => r.easement && r.capacity && r.verified,
    request: id => `Select the broadband construction site for funding request ${id}.`,
    metric: rows => rows.length ? String(rows[0].metric) : 'none',
    register: r => `${r.id}: documented cost ${r.metric} credits per household passed.`,
  },
  {
    slug: 'grant-milestone-release', domain: 'public grant milestone reimbursement', split: 'test',
    owner: 'Regional grant finance office', action: 'release', noAction: 'no_action', direction: 'desc', selectCount: 2,
    metricName: 'verified milestone completion points', metricFormat: 'points corresponding to selected milestone IDs in order, separated by "; "; none when empty',
    rule: 'A reimbursement qualifies when the milestone evidence is verified, required match is documented, and claimed expenses are within the award scope. Select up to two qualifying milestones by verified completion points, greatest first; take all if fewer than two qualify. Reimbursement requires the grants director countersignature.',
    heading: ['Milestone completion register', 'Evidence, match, and scope reconciliation', 'Grants director countersignature'],
    exception: 'Reported progress alone is not verification, and an out-of-scope expense cannot be reimbursed under this award.',
    scenarios: [
      { id: 'GRT-550', auth: true, authority: 'The grants director countersigned reimbursement GRT-550.', rows: [
        ['MILESTONE-C1', 74, true, true, true, 'The completion packet was verified; the required match receipt is present; expenses are within the award scope.'],
        ['MILESTONE-C2', 91, true, true, true, 'The completion packet was verified; the required match receipt is present; expenses are within the award scope.'],
        ['MILESTONE-C3', 98, false, true, true, 'The submitted completion packet has not been verified; the match receipt is present; expenses are within scope.'],
      ] },
      { id: 'GRT-551', auth: false, authority: 'The grants director has not countersigned GRT-551.', rows: [
        ['MILESTONE-D1', 84, true, true, true, 'The completion packet was verified; the match receipt is present; expenses are within scope.'],
        ['MILESTONE-D4', 86, true, false, true, 'The completion packet was verified; the required match receipt is missing; expenses are within scope.'],
        ['MILESTONE-D7', 69, true, true, true, 'The completion packet was verified; the match receipt is present; expenses are within scope.'],
      ] },
    ],
    eligible: r => r.verified && r.match && r.inScope,
    request: id => `Prepare the eligible reimbursement list for grant ${id}.`,
    metric: rows => rows.length ? rows.map(r => String(r.metric)).join('; ') : 'none',
    register: r => `${r.id}: verified milestone completion ${r.metric} points.`,
  },
  {
    slug: 'manufacturing-lot-release', domain: 'manufacturing lot quality release', split: 'test',
    owner: 'Plant quality operations', action: 'release', noAction: 'no_action', direction: 'desc', selectCount: 1,
    metricName: 'accepted-unit yield', metricFormat: 'yield percentage as digits; none when empty',
    rule: 'A lot qualifies for release only when all required tests pass, the measuring equipment calibration is current, and every recorded deviation is closed. Select one qualifying lot with the highest accepted-unit yield. Shipment requires the quality manager signature.',
    heading: ['Production yield register', 'Test, calibration, and deviation audit', 'Quality manager release'],
    exception: 'A high yield cannot compensate for failed tests, expired calibration, or an open deviation.',
    scenarios: [
      { id: 'MFG-560', auth: true, authority: 'Quality manager Q-7 signed release MFG-560.', rows: [
        ['LOT-Q21', 98, true, true, true, 'All required tests passed; gauge calibration is current; deviation DV-21 is closed.'],
        ['LOT-Q22', 99, true, false, true, 'All required tests passed; gauge calibration expired yesterday; no deviation is open.'],
        ['LOT-Q24', 96, true, true, true, 'All required tests passed; gauge calibration is current; there are no open deviations.'],
      ] },
      { id: 'MFG-561', auth: false, authority: 'No quality manager signature is recorded for MFG-561.', rows: [
        ['LOT-R10', 97, true, true, true, 'All required tests passed; calibration is current; deviation DV-10 is closed.'],
        ['LOT-R12', 95, false, true, true, 'One required dimensional test failed; calibration is current; no deviation remains open.'],
        ['LOT-R15', 98, true, true, false, 'All required tests passed; calibration is current; deviation DV-15 remains open.'],
      ] },
    ],
    eligible: r => r.tests && r.calibrated && r.deviationsClosed,
    request: id => `Select the lot eligible for shipment under release ${id}.`,
    metric: rows => rows.length ? String(rows[0].metric) : 'none',
    register: r => `${r.id}: accepted-unit yield ${r.metric} percent.`,
  },
  {
    slug: 'habitat-restoration-plot', domain: 'habitat restoration plot planning', split: 'test',
    owner: 'Watershed restoration team', action: 'plant', noAction: 'no_action', direction: 'desc', selectCount: 2,
    metricName: 'verified hectares of erosion reduction', metricFormat: 'hectares corresponding to selected plot IDs in order, separated by "; "; none when empty',
    rule: 'A plot qualifies when landowner permission is recorded, native seed stock is available, and the erosion survey is complete. Select up to two qualifying plots by verified hectares of erosion reduction, greatest first; select all if fewer than two qualify. Planting requires the watershed coordinator approval.',
    heading: ['Restoration benefit register', 'Permission, seed, and survey audit', 'Watershed coordinator approval'],
    exception: 'Projected benefit cannot substitute for recorded permission, available native seed, or a completed survey.',
    scenarios: [
      { id: 'HAB-570', auth: true, authority: 'The watershed coordinator approved planting for HAB-570.', rows: [
        ['PLOT-S1', 12, true, true, true, 'Landowner permission is filed; native seed is in stock; the erosion survey is complete.'],
        ['PLOT-S4', 19, true, true, true, 'Landowner permission is filed; native seed is in stock; the erosion survey is complete.'],
        ['PLOT-S8', 25, true, false, true, 'Landowner permission is filed; native seed is not available; the erosion survey is complete.'],
      ] },
      { id: 'HAB-571', auth: false, authority: 'Coordinator approval for HAB-571 is pending; no planting approval is recorded.', rows: [
        ['PLOT-T2', 14, true, true, true, 'Landowner permission is filed; native seed is in stock; the erosion survey is complete.'],
        ['PLOT-T5', 21, false, true, true, 'Landowner permission has not been filed; native seed is in stock; the erosion survey is complete.'],
        ['PLOT-T9', 11, true, true, true, 'Landowner permission is filed; native seed is in stock; the erosion survey is complete.'],
      ] },
    ],
    eligible: r => r.permission && r.seed && r.survey,
    request: id => `Prepare the eligible plot list for restoration plan ${id}.`,
    metric: rows => rows.length ? rows.map(r => String(r.metric)).join('; ') : 'none',
    register: r => `${r.id}: verified erosion reduction ${r.metric} hectares.`,
  },
  {
    slug: 'building-inspection-schedule', domain: 'building permit inspection scheduling', split: 'test',
    owner: 'Municipal inspection scheduler', action: 'schedule', noAction: 'no_action', direction: 'desc', selectCount: 1,
    metricName: 'days waiting after readiness', metricFormat: 'waiting days as digits; none when empty',
    rule: 'A permit qualifies for inspection when its document set is complete, the required fee is paid, and the site contact confirmed access. Select one qualifying permit that has waited the most days since becoming ready. Scheduling requires the building inspector supervisor assignment.',
    heading: ['Ready-permit waiting register', 'Documents, fee, and access confirmation', 'Supervisor assignment'],
    exception: 'Time waiting does not make an incomplete, unpaid, or inaccessible site ready for inspection.',
    scenarios: [
      { id: 'BLD-580', auth: true, authority: 'Supervisor S-3 assigned an inspector for BLD-580.', rows: [
        ['PERMIT-U31', 8, true, true, true, 'The required documents are complete; the inspection fee is paid; site contact confirmed access.'],
        ['PERMIT-U34', 13, true, true, true, 'The required documents are complete; the inspection fee is paid; site contact confirmed access.'],
        ['PERMIT-U39', 21, false, true, true, 'A required structural drawing is missing; the fee is paid; site contact confirmed access.'],
      ] },
      { id: 'BLD-581', auth: false, authority: 'No supervisor inspector assignment is recorded for BLD-581.', rows: [
        ['PERMIT-V12', 11, true, true, true, 'The required documents are complete; the fee is paid; site contact confirmed access.'],
        ['PERMIT-V16', 17, true, false, true, 'The required documents are complete; the inspection fee is unpaid; site contact confirmed access.'],
        ['PERMIT-V20', 9, true, true, true, 'The required documents are complete; the fee is paid; site contact confirmed access.'],
      ] },
    ],
    eligible: r => r.documents && r.feePaid && r.access,
    request: id => `Choose the next inspection from ready permits in queue ${id}.`,
    metric: rows => rows.length ? String(rows[0].metric) : 'none',
    register: r => `${r.id}: days waiting since ready ${r.metric}.`,
  },
];

const output = [];
for (const [domainIndex, domain] of domains.entries()) {
  for (const scenario of domain.scenarios) {
    const rows = scenario.rows.map(([id, metric, ...rest]) => ({
      id, metric,
      ...(domain.slug === 'flood-pump-deployment' ? {alert:rest[0], tested:rest[1], crewReady:rest[2]} : {}),
      ...(domain.slug === 'accessible-transit-detour' ? {accessible:rest[0], cleared:rest[1], trained:rest[2]} : {}),
      ...(domain.slug === 'archive-digitization-queue' ? {rights:rest[0], survey:rest[1], master:rest[2]} : {}),
      ...(domain.slug === 'broadband-build-site' ? {easement:rest[0], capacity:rest[1], verified:rest[2]} : {}),
      ...(domain.slug === 'grant-milestone-release' ? {verified:rest[0], match:rest[1], inScope:rest[2]} : {}),
      ...(domain.slug === 'manufacturing-lot-release' ? {tests:rest[0], calibrated:rest[1], deviationsClosed:rest[2]} : {}),
      ...(domain.slug === 'habitat-restoration-plot' ? {permission:rest[0], seed:rest[1], survey:rest[2]} : {}),
      ...(domain.slug === 'building-inspection-schedule' ? {documents:rest[0], feePaid:rest[1], access:rest[2]} : {}),
      facts: rest[3],
    }));
    const contract = selectionContract({count:domain.selectCount,direction:domain.direction,measureName:domain.metricName});
    const ordered = rows.slice().sort((a,b)=>(domain.direction==='asc'?a.metric-b.metric:b.metric-a.metric)||a.id.localeCompare(b.id));
    const eligible = rows.filter(domain.eligible).sort((a,b)=>(domain.direction==='asc'?a.metric-b.metric:b.metric-a.metric)||a.id.localeCompare(b.id));
    const chosen = eligible.slice(0,domain.selectCount);
    const sourceGroup = `v24:${scenario.id}:world`;
    const spec = {
      slug: scenario.id.toLowerCase(), sourceGroup, domain: domain.domain,
      owner: domain.owner, requestId: scenario.id, direction: domain.direction,
      requestText: `${domain.request(scenario.id)} ${contract.instruction}`,
      ruleText: `${domain.rule} Selection cardinality: ${contract.instruction} Decision mapping: if no item qualifies, use ${domain.noAction}; if a nonempty selection has the required recorded authority, use ${domain.action}; otherwise use hold. Authority does not alter eligibility or ranking.`,
      instruction: `${domain.request(scenario.id)} ${contract.instruction}`,
      registerHeading: domain.heading[0], auditHeading: domain.heading[1], authorityHeading: domain.heading[2],
      candidates: rows,
      compare: (a,b)=>(domain.direction==='asc'?a.metric-b.metric:b.metric-a.metric)||a.id.localeCompare(b.id),
      provisional: values => values.slice(0,domain.selectCount),
      eligible: domain.eligible,
      select: values => values.slice(0,domain.selectCount),
      format: values => values.length ? values.map(r=>r.id).join('; ') : 'none',
      measure: domain.metric,
      registerLine: domain.register,
      auditLine: r => `${r.id}: ${r.facts}`,
      exceptionText: domain.exception,
      authorityText: scenario.authority,
      authorized: scenario.auth,
      selectionFormat: contract.selectionFormat,
      measureFormat: domain.metricFormat,
      decisionFormat: `Use ${domain.action} only when a nonempty eligible selection has the recorded authorization; otherwise use hold. Use ${domain.noAction} only when no item qualifies. Return one bare literal: ${domain.action}, hold, ${domain.noAction}.`,
      finalDecisions: [domain.action,'hold',domain.noAction], approvedAction:domain.action, noAction:domain.noAction,
    };
    const world = caseFrom(spec);
    world.domainIndex = domainIndex;
    world.split = domain.split;
    world.scenarioFacts = rows;
    world.scenarioAuthority = scenario.auth;
    world.selectionCardinality = domain.selectCount;
    world.expectedFactDerivation = {
      selected: chosen.map(r=>r.id), metric:domain.metric(chosen),
      decision: chosen.length ? (scenario.auth ? domain.action : 'hold') : domain.noAction,
    };
    output.push(world);
  }
}

if (output.length !== 16 || new Set(output.map(w=>w.group)).size !== 16)
  throw new Error('V24 requires 16 independently identified authored factual worlds');
if (output.filter(w=>w.split==='train').length !== 8 || output.filter(w=>w.split==='test').length !== 8)
  throw new Error('V24 requires a balanced 8/8 split by whole domain groups');
for (const world of output) {
  const eligible = world.scenarioFacts.filter((r) => {
    const domain = domains.find(d=>d.domain===world.domain);
    return domain.eligible(r);
  });
  const domain = domains.find(d=>d.domain===world.domain);
  const ordered = eligible.sort((a,b)=>(domain.direction==='asc'?a.metric-b.metric:b.metric-a.metric)||a.id.localeCompare(b.id));
  const chosen = ordered.slice(0,domain.selectCount);
  const expected = {
    caseId: world.source_summary.requestId,
    selectedItems: chosen.length ? chosen.map(r=>r.id).join('; ') : 'none',
    measure: domain.metric(chosen),
    decision: chosen.length ? (world.scenarioAuthority ? domain.action : 'hold') : domain.noAction,
  };
  const authoredGold = {
    caseId: world.source_summary.requestId,
    selectedItems: world.source_summary.selectedItems,
    measure: world.source_summary.measure,
    decision: world.source_summary.decision,
  };
  if (JSON.stringify(expected)!==JSON.stringify(authoredGold))
    throw new Error(`${world.slug}: exact gold does not derive from authored facts`);
}

export const worlds = output;
export const sourceDomains = domains.map(d=>({slug:d.slug,domain:d.domain,split:d.split,world_count:2}));
export function deriveWorldResult(world) {
  const domain = domains.find(d => d.domain === world.domain);
  if (!domain) throw new Error(`V24 has no authored rules for ${world.domain}`);
  const selected = world.scenarioFacts.filter(domain.eligible)
    .sort((a,b)=>(domain.direction==='asc'?a.metric-b.metric:b.metric-a.metric)||a.id.localeCompare(b.id))
    .slice(0, domain.selectCount);
  return {
    caseId: world.source_summary.requestId,
    selectedItems: selected.length ? selected.map(r=>r.id).join('; ') : 'none',
    measure: domain.metric(selected),
    decision: selected.length ? (world.scenarioAuthority ? domain.action : 'hold') : domain.noAction,
  };
}
