import { worldSpecs, caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';

const bySlug = new Map(worldSpecs.map(spec => [spec.slug, spec]));

function variant(parentSlug, suffix, requestId, candidates, { authorized, authorityText, exceptionText, ruleAddition = '' }) {
  const parent = bySlug.get(parentSlug);
  if (!parent) throw new Error(`unknown inherited V18 source group: ${parentSlug}`);
  const replaceId = value => {
    if (typeof value === 'string') return value.replaceAll(parent.requestId, requestId);
    if (Array.isArray(value)) return value.map(replaceId);
    if (value && typeof value === 'object')
      return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, replaceId(child)]));
    return value;
  };
  const spec = replaceId(parent);
  const decisionMap = `Decision mapping: if the eligible selection is empty, use ${parent.noAction}. If it is nonempty and every stated authority condition is met, use ${parent.approvedAction}; if nonempty and any required authority is missing, use hold. Authority never changes selectedItems or measure.`;
  spec.slug = `${parentSlug}-v21-${suffix}`;
  spec.sourceGroup = `v18:${parentSlug}:world`;
  spec.requestId = requestId;
  spec.candidates = candidates;
  spec.authorized = authorized;
  spec.authorityText = authorityText;
  spec.exceptionText = exceptionText;
  spec.ruleText += ` ${decisionMap}${ruleAddition ? ` ${ruleAddition}` : ''}`;
  const baseDecisionFormat = spec.decisionFormat ??
    `Final disposition: exactly ${spec.finalDecisions.join(', ')}. Selection and authorization are separate; do not change selectedItems merely because authorization is absent.`;
  spec.decisionFormat = `${baseDecisionFormat} ${decisionMap}`;
  spec.instruction = `${spec.instruction} Apply the complete rule and branch mapping: ${decisionMap} ${ruleAddition}`.trim();
  const world = caseFrom(spec);
  world.group = spec.sourceGroup;
  world.decision_rule = spec.ruleText;
  world.instruction = spec.instruction;
  world.source_summary.scenario = suffix;
  world.source_summary.parent_group = spec.sourceGroup;
  return world;
}

const bookingHours = 'Facility staffed hours begin at 09:00. A slot before 09:00 is after-hours and needs a safety permit naming its exact instrument and complete slot ID.';

export const worlds = [
  variant('software-release-gate', 'rel-tied-eligible-waiver', 'REL-653', [
    {id:'BUILD-52',versionRank:52,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-52'},
    {id:'BUILD-51',versionRank:52,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-51'},
    {id:'BUILD-53',versionRank:53,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-50'},
    {id:'BUILD-54',versionRank:54,tests:false,blockers:0,critical:false,waiverBuild:'none'},
  ], {authorized:true, authorityText:'Signed change order CO-925 covers REL-653 and the deployment window is open.',
    exceptionText:'The blocker waiver must name the exact complete build ID. Failed tests and mismatched waivers remain disqualifying.',
    ruleAddition:'Eligibility requires required tests to have passed and either zero blockers or exactly one noncritical blocker covered by a waiver naming that same complete build ID. Among equally ranked eligible builds, choose ascending complete build ID.'}),
  variant('software-release-gate', 'rel-mismatched-waiver-and-hold', 'REL-654', [
    {id:'BUILD-61',versionRank:61,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-60'},
    {id:'BUILD-60',versionRank:60,tests:true,blockers:0,critical:false,waiverBuild:'none'},
    {id:'BUILD-59',versionRank:59,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-59'},
    {id:'BUILD-62',versionRank:62,tests:true,blockers:1,critical:true,waiverBuild:'BUILD-62'},
  ], {authorized:false, authorityText:'No signed change order is recorded for REL-654, though the deployment window is open.',
    exceptionText:'The highest-ranked builds have critical blockers or a waiver naming another build. A valid waiver is limited to its exact build ID.'}),

  variant('dependency-upgrade-batch', 'dep-equal-score-cutoff', 'DEP-233', [
    {id:'PKG-AA',reduction:100,reviewed:false,tests:true,conflict:false},
    {id:'PKG-AB',reduction:95,reviewed:true,tests:true,conflict:false},
    {id:'PKG-AC',reduction:95,reviewed:true,tests:true,conflict:false},
    {id:'PKG-AD',reduction:95,reviewed:true,tests:true,conflict:false},
    {id:'PKG-AE',reduction:90,reviewed:true,tests:true,conflict:false},
  ], {authorized:true, authorityText:'Two distinct maintainer approvals are recorded for DEP-233.',
    exceptionText:'Incomplete review excludes PKG-AA before ranking. Equal scores at the two-item cutoff use ascending complete package ID.'}),
  variant('dependency-upgrade-batch', 'dep-single-qualified-unsigned', 'DEP-234', [
    {id:'PKG-BA',reduction:99,reviewed:true,tests:false,conflict:false},
    {id:'PKG-BB',reduction:91,reviewed:true,tests:true,conflict:false},
    {id:'PKG-BC',reduction:90,reviewed:false,tests:true,conflict:false},
    {id:'PKG-BD',reduction:88,reviewed:true,tests:true,conflict:true},
  ], {authorized:false, authorityText:'Only one maintainer approval is recorded for DEP-234; two distinct approvals are required.',
    exceptionText:'Failed compatibility tests, incomplete review, and unresolved conflicts each independently disqualify an update.'}),

  variant('research-instrument-booking', 'lab-earliest-permitted-time-tie', 'LAB-373', [
    {id:'SLOT-A',start:'08:30',end:'10:00',duration:90,instrument:'MIC-2',calibrated:true,training:true,conflict:false,afterHours:true,permit:'MIC-2/SLOT-A'},
    {id:'SLOT-B',start:'08:30',end:'10:00',duration:90,instrument:'MIC-1',calibrated:true,training:true,conflict:false,afterHours:true,permit:'MIC-9/SLOT-B'},
    {id:'SLOT-C',start:'09:00',end:'10:30',duration:90,instrument:'MIC-3',calibrated:true,training:true,conflict:false,afterHours:false,permit:'none'},
    {id:'SLOT-D',start:'08:15',end:'09:45',duration:90,instrument:'MIC-4',calibrated:false,training:true,conflict:false,afterHours:true,permit:'MIC-4/SLOT-D'},
  ], {authorized:true, authorityText:'The PI approved LAB-373 and selected equipment is available.',
    exceptionText:'An exact permit is required before 09:00; it cannot fix poor calibration. Equal eligible starts are ordered by complete slot ID.',
    ruleAddition:bookingHours}),
  variant('research-instrument-booking', 'lab-authorized-but-empty', 'LAB-374', [
    {id:'SLOT-E',start:'08:00',end:'09:30',duration:90,instrument:'MIC-5',calibrated:true,training:true,conflict:true,afterHours:true,permit:'MIC-5/SLOT-E'},
    {id:'SLOT-F',start:'09:15',end:'10:45',duration:90,instrument:'MIC-6',calibrated:true,training:false,conflict:false,afterHours:false,permit:'none'},
    {id:'SLOT-G',start:'08:45',end:'10:15',duration:90,instrument:'MIC-7',calibrated:false,training:true,conflict:false,afterHours:true,permit:'MIC-7/SLOT-G'},
  ], {authorized:true, authorityText:'The PI approved LAB-374 and equipment is available.',
    exceptionText:'Reservation conflicts and missing current training are independent blockers; permits cannot override them.',
    ruleAddition:bookingHours}),

  variant('privacy-export-minimization', 'px-consent-purpose-and-identifier', 'PX-123', [
    {id:'region',position:1,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
    {id:'age_band',position:2,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
    {id:'patient_name',position:3,requested:true,classification:'public',consentedPurposes:['regional-outcomes'],directIdentifier:true},
    {id:'rare_condition',position:4,requested:true,classification:'restricted',consentedPurposes:['regional-outcomes'],directIdentifier:false},
    {id:'outcome_code',position:5,requested:false,classification:'public',consentedPurposes:[],directIdentifier:false},
  ], {authorized:true, authorityText:'The data protection officer approved PX-123.',
    exceptionText:'The requested flag in the current schema register is authoritative; include a field only when that register marks it requested. Consent for care-quality does not authorize regional-outcomes. Direct identifiers remain excluded despite request and consent.'}),
  variant('privacy-export-minimization', 'px-three-field-unsigned', 'PX-124', [
    {id:'region',position:1,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
    {id:'age_band',position:2,requested:true,classification:'restricted',consentedPurposes:['regional-outcomes'],directIdentifier:false},
    {id:'patient_name',position:3,requested:true,classification:'public',consentedPurposes:[],directIdentifier:true},
    {id:'rare_condition',position:4,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
    {id:'outcome_code',position:5,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
  ], {authorized:false, authorityText:'The data protection officer has not approved PX-124.',
    exceptionText:'Public non-identifiers and restricted fields with exact-purpose consent qualify; exclude the direct identifier and preserve schema order.'}),

  variant('incident-mitigation-selection', 'inc-eligible-top-two-waiver-tie', 'INC-923', [
    {id:'ACT-A',reduction:103,prerequisites:true,blast:3,cap:3,tested:false,waiver:'ACT-A'},
    {id:'ACT-C',reduction:103,prerequisites:true,blast:3,cap:3,tested:true,waiver:'none'},
    {id:'ACT-B',reduction:104,prerequisites:true,blast:4,cap:3,tested:true,waiver:'none'},
    {id:'ACT-D',reduction:99,prerequisites:true,blast:2,cap:3,tested:true,waiver:'none'},
  ], {authorized:true, authorityText:'Incident commander authorization is recorded for INC-923.',
    exceptionText:'The exact-action waiver permits ACT-A only because its prerequisites are complete and blast radius fits the cap. The untested waiver cannot be applied to another action.'}),
  variant('incident-mitigation-selection', 'inc-one-action-held', 'INC-924', [
    {id:'ACT-E',reduction:110,prerequisites:false,blast:1,cap:3,tested:true,waiver:'none'},
    {id:'ACT-F',reduction:92,prerequisites:true,blast:2,cap:3,tested:true,waiver:'none'},
    {id:'ACT-G',reduction:91,prerequisites:true,blast:3,cap:3,tested:false,waiver:'ACT-F'},
    {id:'ACT-H',reduction:90,prerequisites:true,blast:4,cap:3,tested:true,waiver:'none'},
  ], {authorized:false, authorityText:'No incident commander authorization is recorded for INC-924.',
    exceptionText:'A waiver naming ACT-F does not cover ACT-G. Missing prerequisites and an over-cap blast radius disqualify actions before ranking.'}),

  variant('research-sample-release', 'sample-linked-retest-tie', 'SR-473', [
    {id:'SMP-A',collected:'07:10',custody:true,initialQc:'fail',retest:'pass:SMP-A'},
    {id:'SMP-B',collected:'07:10',custody:true,initialQc:'pass',retest:'none'},
    {id:'SMP-C',collected:'07:00',custody:true,initialQc:'fail',retest:'pass:SMP-B'},
    {id:'SMP-D',collected:'07:15',custody:false,initialQc:'pass',retest:'none'},
  ], {authorized:true, authorityText:'QA lead signature for SR-473 is recorded.',
    exceptionText:'A retest changes only the batch named by its complete ID. Equal collection times are ordered by complete batch ID.'}),
  variant('research-sample-release', 'sample-single-qualified-unsigned', 'SR-474', [
    {id:'SMP-E',collected:'06:45',custody:false,initialQc:'pass',retest:'none'},
    {id:'SMP-F',collected:'06:50',custody:true,initialQc:'pass',retest:'none'},
    {id:'SMP-G',collected:'06:55',custody:true,initialQc:'fail',retest:'pass:SMP-F'},
    {id:'SMP-H',collected:'07:00',custody:true,initialQc:'fail',retest:'none'},
  ], {authorized:false, authorityText:'QA lead signature is absent for SR-474.',
    exceptionText:'Custody remains mandatory after a passing initial QC. SMP-G’s retest entry says pass:SMP-F, which names a different complete batch ID; it does not supersede SMP-G’s failed initial QC.'}),

  variant('volunteer-shift-coverage', 'vol-same-person-cross-role-tie', 'VS-633', [
    {id:'VOL-A/INTERPRETER',person:'VOL-A',role:'interpreter',priority:1,fit:98,qualified:true,available:true,rest:12,waiver:'none'},
    {id:'VOL-B/INTERPRETER',person:'VOL-B',role:'interpreter',priority:1,fit:98,qualified:true,available:true,rest:12,waiver:'none'},
    {id:'VOL-A/RADIO',person:'VOL-A',role:'radio',priority:2,fit:100,qualified:true,available:true,rest:12,waiver:'none'},
    {id:'VOL-C/RADIO',person:'VOL-C',role:'radio',priority:2,fit:96,qualified:true,available:true,rest:12,waiver:'none'},
    {id:'VOL-D/RADIO',person:'VOL-D',role:'radio',priority:2,fit:101,qualified:false,available:true,rest:12,waiver:'none'},
  ], {authorized:true, authorityText:'Coordinator approval is recorded for VS-633.',
    exceptionText:'A person can fill at most one role. Equal-fit interpreter candidates are ordered by complete person ID; unqualified volunteers cannot be selected.'}),
  variant('volunteer-shift-coverage', 'vol-scoped-rest-exception-held', 'VS-634', [
    {id:'VOL-E/INTERPRETER',person:'VOL-E',role:'interpreter',priority:1,fit:99,qualified:true,available:true,rest:7,waiver:'VOL-E/interpreter'},
    {id:'VOL-E/RADIO',person:'VOL-E',role:'radio',priority:2,fit:100,qualified:true,available:true,rest:7,waiver:'VOL-E/interpreter'},
    {id:'VOL-F/RADIO',person:'VOL-F',role:'radio',priority:2,fit:89,qualified:true,available:true,rest:10,waiver:'none'},
    {id:'VOL-G/INTERPRETER',person:'VOL-G',role:'interpreter',priority:1,fit:102,qualified:true,available:false,rest:12,waiver:'none'},
  ], {authorized:false, authorityText:'Coordinator approval is absent for VS-634.',
    exceptionText:'The signed rest exception names VOL-E/interpreter only; it cannot cover VOL-E/radio or permit the same person to fill both roles.'}),

  variant('records-retention-disposition', 'records-cutoff-date-tie', 'RT-243', [
    {id:'REC-A',expiry:'2026-10-01',hold:false,notice:true},
    {id:'REC-B',expiry:'2026-10-01',hold:false,notice:true},
    {id:'REC-C',expiry:'2026-10-02',hold:false,notice:true},
    {id:'REC-D',expiry:'2019-03-01',hold:true,notice:true},
  ], {authorized:true, authorityText:'Counsel signed a destruction order for RT-243.',
    exceptionText:'The cutoff date is inclusive. Legal holds still disqualify records, and equal expiry dates sort by complete record ID.'}),
  variant('records-retention-disposition', 'records-single-qualified-unsigned', 'RT-244', [
    {id:'REC-E',expiry:'2018-06-01',hold:true,notice:true},
    {id:'REC-F',expiry:'2026-10-01',hold:false,notice:true},
    {id:'REC-G',expiry:'2026-10-01',hold:false,notice:false},
    {id:'REC-H',expiry:'2020-01-01',hold:false,notice:false},
  ], {authorized:false, authorityText:'Counsel has not signed a destruction order for RT-244.',
    exceptionText:'An exact-cutoff expiry qualifies only when no legal hold applies and custodian notice is complete. Signature affects disposition, not the proposed selection.'}),
];

if (worlds.length !== 16) throw new Error(`V21 requires 16 authored variants; found ${worlds.length}`);
