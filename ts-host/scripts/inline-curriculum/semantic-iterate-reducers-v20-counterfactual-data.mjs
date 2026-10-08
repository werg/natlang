import { worldSpecs, caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';

const bases = new Map(worldSpecs.map(spec => [spec.slug, spec]));
const mapping = spec => `Decision mapping: if the eligible selection is empty, use ${spec.noAction}. If it is nonempty and every stated authority condition is met, use ${spec.approvedAction}; if nonempty and any required authority is missing, use hold. Authority never changes selectedItems or measure.`;

function variant(parentSlug, suffix, requestId, candidates, authorized, authorityText, exceptionText) {
  const base = bases.get(parentSlug);
  if (!base) throw new Error(`unknown parent world: ${parentSlug}`);
  const decisionMap = mapping(base);
  const spec = {
    ...base,
    slug: `${parentSlug}-v20-${suffix}-v2`,
    sourceGroup: `v18:${parentSlug}:world`,
    requestId,
    candidates,
    authorized,
    authorityText,
    exceptionText,
    ruleText: `${base.ruleText} ${decisionMap}`,
    decisionFormat: `${base.decisionFormat ?? `Return exactly one bare literal: ${base.finalDecisions.join(', ')}.`} ${decisionMap}`,
    instruction: `${base.instruction} Apply the complete rule and branch mapping: ${decisionMap}`,
  };
  const world = caseFrom(spec);
  world.group = spec.sourceGroup;
  world.decision_rule = spec.ruleText;
  world.instruction = spec.instruction;
  world.source_summary.scenario = suffix;
  world.source_summary.parent_group = spec.sourceGroup;
  return world;
}

export const worlds = [
  // Release gate: scoped exact-build waiver, authorized; same policy, authority absent; empty branch.
  variant('software-release-gate','rel-authorized-exact-waiver','REL-650',[
    {id:'BUILD-22',versionRank:22,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-22'},
    {id:'BUILD-23',versionRank:23,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-22'},
    {id:'BUILD-21',versionRank:21,tests:true,blockers:0,critical:false,waiverBuild:'none'},
    {id:'BUILD-24',versionRank:24,tests:false,blockers:0,critical:false,waiverBuild:'none'},
  ],true,'Signed change order CO-920 covers REL-650 and the deployment window remains open.','The waiver names BUILD-22 only; it does not cover BUILD-23 or failed tests.'),
  variant('software-release-gate','rel-unauthorized-waiver-tie','REL-651',[
    {id:'BUILD-31',versionRank:31,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-31'},
    {id:'BUILD-30',versionRank:30,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-30'},
    {id:'BUILD-32',versionRank:32,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-30'},
    {id:'BUILD-29',versionRank:29,tests:true,blockers:0,critical:false,waiverBuild:'none'},
  ],false,'No signed change order is recorded for REL-651; the window is open.','Each one-blocker waiver applies only to its exact build ID; BUILD-32 is not covered.'),
  variant('software-release-gate','rel-empty-failed-conditions','REL-652',[
    {id:'BUILD-42',versionRank:42,tests:false,blockers:0,critical:false,waiverBuild:'none'},
    {id:'BUILD-41',versionRank:41,tests:true,blockers:1,critical:true,waiverBuild:'BUILD-41'},
    {id:'BUILD-40',versionRank:40,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-39'},
  ],true,'Signed change order CO-922 is present and the window is open.','A waiver cannot cure failed tests, a critical blocker, or a mismatched build ID.'),

  // Dependency batch: top-two risk reduction with complete review; unauthorized sibling; empty.
  variant('dependency-upgrade-batch','dep-authorized-ranked-tie','DEP-230',[
    {id:'PKG-M',reduction:96,reviewed:true,tests:true,conflict:false}, {id:'PKG-N',reduction:96,reviewed:true,tests:true,conflict:false},
    {id:'PKG-P',reduction:99,reviewed:false,tests:true,conflict:false}, {id:'PKG-Q',reduction:94,reviewed:true,tests:true,conflict:false},
  ],true,'Two distinct maintainer approvals are recorded for DEP-230.','The unreviewed highest-score update is ineligible; equal eligible scores are ordered by complete package ID.'),
  variant('dependency-upgrade-batch','dep-unauthorized-conflict-order','DEP-231',[
    {id:'PKG-R',reduction:88,reviewed:true,tests:true,conflict:false}, {id:'PKG-S',reduction:95,reviewed:true,tests:true,conflict:false},
    {id:'PKG-T',reduction:95,reviewed:true,tests:true,conflict:false}, {id:'PKG-U',reduction:98,reviewed:true,tests:true,conflict:true},
  ],false,'Only one maintainer approval is recorded for DEP-231; two are required.','Unresolved conflicts disqualify an update before ranking; ties use ascending full package ID.'),
  variant('dependency-upgrade-batch','dep-empty-no-reviewed-updates','DEP-232',[
    {id:'PKG-V',reduction:99,reviewed:false,tests:true,conflict:false}, {id:'PKG-W',reduction:97,reviewed:true,tests:false,conflict:false},
    {id:'PKG-X',reduction:95,reviewed:true,tests:true,conflict:true},
  ],true,'Two approvals are recorded for DEP-232.','Review, passing tests, and conflict-free status are all required; approvals cannot replace them.'),

  // Instrument booking: exact permit and time tie; unauthorized sibling; no eligible slot.
  variant('research-instrument-booking','lab-authorized-permit-tie','LAB-370',[
    {id:'SLOT-M',start:'08:20',end:'09:50',duration:90,instrument:'MIC-7',calibrated:true,training:true,conflict:false,afterHours:true,permit:'MIC-7/SLOT-M'},
    {id:'SLOT-N',start:'08:20',end:'09:50',duration:90,instrument:'MIC-8',calibrated:true,training:true,conflict:false,afterHours:true,permit:'MIC-8/SLOT-N'},
    {id:'SLOT-L',start:'08:10',end:'09:40',duration:90,instrument:'MIC-6',calibrated:false,training:true,conflict:false,afterHours:false,permit:'none'},
  ],true,'The PI approved LAB-370 and both listed instruments are available.','After-hours permits are scoped to the exact instrument and complete slot ID; equal starts break by slot ID.'),
  variant('research-instrument-booking','lab-unauthorized-earliest-eligible','LAB-371',[
    {id:'SLOT-P',start:'08:05',end:'09:35',duration:90,instrument:'MIC-2',calibrated:true,training:true,conflict:false,afterHours:false,permit:'none'},
    {id:'SLOT-Q',start:'08:00',end:'09:30',duration:90,instrument:'MIC-1',calibrated:true,training:true,conflict:true,afterHours:false,permit:'none'},
    {id:'SLOT-R',start:'08:05',end:'09:35',duration:90,instrument:'MIC-3',calibrated:true,training:true,conflict:false,afterHours:false,permit:'none'},
  ],false,'PI approval is absent for LAB-371, though the equipment availability record is present.','Choose the earliest eligible start; a conflict disqualifies a slot, and ties use full slot ID.'),
  variant('research-instrument-booking','lab-empty-calibration-training','LAB-372',[
    {id:'SLOT-S',start:'08:00',end:'09:30',duration:90,instrument:'MIC-4',calibrated:false,training:true,conflict:false,afterHours:false,permit:'none'},
    {id:'SLOT-T',start:'08:15',end:'09:45',duration:90,instrument:'MIC-5',calibrated:true,training:false,conflict:false,afterHours:false,permit:'none'},
    {id:'SLOT-U',start:'08:30',end:'10:00',duration:90,instrument:'MIC-6',calibrated:true,training:true,conflict:true,afterHours:false,permit:'none'},
  ],true,'PI approval and equipment availability are recorded for LAB-372.','Calibration, current training, and no-conflict are required; authority cannot make an ineligible slot eligible.'),

  // Privacy export: explicit request and purpose; authorized and unauthorized results; empty purpose/identifier branch.
  variant('privacy-export-minimization','px-authorized-direct-id-excluded','PX-120',[
    {id:'region',position:1,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
    {id:'age_band',position:2,requested:true,classification:'restricted',consentedPurposes:['regional-outcomes'],directIdentifier:false},
    {id:'patient_name',position:3,requested:true,classification:'public',consentedPurposes:['regional-outcomes'],directIdentifier:true},
    {id:'rare_condition',position:4,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
    {id:'outcome_code',position:5,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
  ],true,'The data protection officer approved export PX-120.','The direct-identifier rule excludes patient_name even though it is requested and public.'),
  variant('privacy-export-minimization','px-unauthorized-schema-order','PX-121',[
    {id:'region',position:1,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
    {id:'age_band',position:2,requested:true,classification:'restricted',consentedPurposes:['regional-outcomes'],directIdentifier:false},
    {id:'patient_name',position:3,requested:true,classification:'public',consentedPurposes:[],directIdentifier:true},
    {id:'rare_condition',position:4,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
    {id:'outcome_code',position:5,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
  ],false,'The data protection officer has not approved PX-121.','Include only requested non-identifiers that are public or consented for regional-outcomes; preserve schema order.'),
  variant('privacy-export-minimization','px-empty-purpose-mismatch','PX-122',[
    {id:'region',position:1,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
    {id:'age_band',position:2,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
    {id:'patient_name',position:3,requested:true,classification:'public',consentedPurposes:['regional-outcomes'],directIdentifier:true},
    {id:'rare_condition',position:4,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
    {id:'outcome_code',position:5,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
  ],true,'The data protection officer approved PX-122.','Public status does not override direct-identifier exclusion; consent for another purpose does not qualify.'),

  // Incident mitigation: scored top-two and scoped waiver; authority absent; empty due failed gates.
  variant('incident-mitigation-selection','inc-authorized-exact-waiver-tie','INC-920',[
    {id:'ACT-M',reduction:93,prerequisites:true,blast:4,cap:4,tested:false,waiver:'ACT-M'},
    {id:'ACT-N',reduction:93,prerequisites:true,blast:4,cap:4,tested:true,waiver:'none'},
    {id:'ACT-P',reduction:97,prerequisites:true,blast:5,cap:4,tested:true,waiver:'none'},
    {id:'ACT-Q',reduction:90,prerequisites:true,blast:3,cap:4,tested:true,waiver:'none'},
  ],true,'Incident commander authorization is recorded for INC-920.','ACT-M waiver covers only ACT-M; ACT-P exceeds the approved blast cap; ties use action ID.'),
  variant('incident-mitigation-selection','inc-unauthorized-ranked-actions','INC-921',[
    {id:'ACT-R',reduction:89,prerequisites:true,blast:2,cap:3,tested:true,waiver:'none'},
    {id:'ACT-S',reduction:94,prerequisites:true,blast:3,cap:3,tested:true,waiver:'none'},
    {id:'ACT-T',reduction:96,prerequisites:true,blast:4,cap:3,tested:true,waiver:'none'},
  ],false,'No incident commander execution authorization is recorded for INC-921.','Rank only actions meeting prerequisites and blast cap; execution authority does not change the selected actions.'),
  variant('incident-mitigation-selection','inc-empty-unmet-prerequisites','INC-922',[
    {id:'ACT-U',reduction:99,prerequisites:false,blast:1,cap:3,tested:true,waiver:'none'},
    {id:'ACT-V',reduction:98,prerequisites:true,blast:4,cap:3,tested:true,waiver:'none'},
    {id:'ACT-W',reduction:97,prerequisites:true,blast:2,cap:3,tested:false,waiver:'ACT-X'},
  ],true,'Incident commander authorization is recorded for INC-922.','Prerequisites, the blast cap, and a matching exact-action waiver are independently required.'),

  // Sample release: exact-batch retest; authority absent; empty due custody and mismatched retests.
  variant('research-sample-release','sample-authorized-linked-retests','SR-470',[
    {id:'SMP-M',collected:'06:40',custody:true,initialQc:'fail',retest:'pass:SMP-M'},
    {id:'SMP-N',collected:'06:30',custody:true,initialQc:'pass',retest:'none'},
    {id:'SMP-P',collected:'06:50',custody:true,initialQc:'fail',retest:'pass:SMP-M'},
    {id:'SMP-Q',collected:'07:00',custody:false,initialQc:'pass',retest:'none'},
  ],true,'QA lead signature for SR-470 is recorded.','A passing retest applies only to its named complete batch ID; SMP-M retest does not qualify SMP-P.'),
  variant('research-sample-release','sample-unauthorized-custody-order','SR-471',[
    {id:'SMP-R',collected:'07:20',custody:true,initialQc:'pass',retest:'none'},
    {id:'SMP-S',collected:'07:10',custody:true,initialQc:'fail',retest:'pass:SMP-S'},
    {id:'SMP-T',collected:'07:05',custody:false,initialQc:'pass',retest:'none'},
  ],false,'QA lead signature is absent for SR-471.','Sort qualified batches by collection time then ID; custody remains mandatory even when QC passed.'),
  variant('research-sample-release','sample-empty-no-valid-qc','SR-472',[
    {id:'SMP-U',collected:'07:05',custody:true,initialQc:'fail',retest:'pass:SMP-V'},
    {id:'SMP-V',collected:'07:10',custody:false,initialQc:'pass',retest:'none'},
    {id:'SMP-W',collected:'07:15',custody:true,initialQc:'fail',retest:'none'},
  ],true,'QA lead signature for SR-472 is recorded.','An exact linked passing retest or initial pass is required, and custody must be intact.'),

  // Volunteer staffing: complete role assignment with exact waiver; incomplete unauthorized; empty.
  variant('volunteer-shift-coverage','vol-authorized-scoped-rest-waiver','VS-630',[
    {id:'VOL-M/INTERPRETER',person:'VOL-M',role:'interpreter',priority:1,fit:96,qualified:true,available:true,rest:8,waiver:'VOL-M/interpreter'},
    {id:'VOL-N/INTERPRETER',person:'VOL-N',role:'interpreter',priority:1,fit:99,qualified:true,available:true,rest:7,waiver:'VOL-N/radio'},
    {id:'VOL-O/INTERPRETER',person:'VOL-O',role:'interpreter',priority:1,fit:90,qualified:true,available:true,rest:11,waiver:'none'},
    {id:'VOL-M/RADIO',person:'VOL-M',role:'radio',priority:2,fit:100,qualified:true,available:true,rest:8,waiver:'VOL-M/interpreter'},
    {id:'VOL-P/RADIO',person:'VOL-P',role:'radio',priority:2,fit:92,qualified:true,available:true,rest:12,waiver:'none'},
  ],true,'Coordinator approval is recorded for VS-630.','The rest waiver names VOL-M/interpreter only; VOL-M cannot fill both roles.'),
  variant('volunteer-shift-coverage','vol-unauthorized-one-role-only','VS-631',[
    {id:'VOL-Q/INTERPRETER',person:'VOL-Q',role:'interpreter',priority:1,fit:91,qualified:true,available:true,rest:12,waiver:'none'},
    {id:'VOL-R/INTERPRETER',person:'VOL-R',role:'interpreter',priority:1,fit:99,qualified:false,available:true,rest:14,waiver:'none'},
    {id:'VOL-Q/RADIO',person:'VOL-Q',role:'radio',priority:2,fit:100,qualified:true,available:true,rest:12,waiver:'none'},
    {id:'VOL-S/RADIO',person:'VOL-S',role:'radio',priority:2,fit:89,qualified:true,available:true,rest:10,waiver:'none'},
  ],false,'Coordinator approval is absent for VS-631.','A person can fill at most one role; missing qualification cannot be waived.'),
  variant('volunteer-shift-coverage','vol-empty-no-qualified-available-rest','VS-632',[
    {id:'VOL-T/INTERPRETER',person:'VOL-T',role:'interpreter',priority:1,fit:99,qualified:true,available:false,rest:12,waiver:'none'},
    {id:'VOL-U/INTERPRETER',person:'VOL-U',role:'interpreter',priority:1,fit:96,qualified:false,available:true,rest:12,waiver:'none'},
    {id:'VOL-V/RADIO',person:'VOL-V',role:'radio',priority:2,fit:98,qualified:true,available:true,rest:7,waiver:'VOL-V/interpreter'},
    {id:'VOL-W/RADIO',person:'VOL-W',role:'radio',priority:2,fit:90,qualified:true,available:false,rest:13,waiver:'none'},
  ],true,'Coordinator approval is recorded for VS-632.','Availability, role qualification, and minimum rest are required; a waiver must name the exact person and role.'),

  // Retention disposition: nonempty + signed; nonempty + unsigned; empty despite authority.
  variant('records-retention-disposition','records-authorized-expiry-tie','RT-240',[
    {id:'REC-M',expiry:'2020-03-01',hold:false,notice:true}, {id:'REC-N',expiry:'2020-03-01',hold:false,notice:true},
    {id:'REC-P',expiry:'2019-12-31',hold:true,notice:true}, {id:'REC-Q',expiry:'2027-01-01',hold:false,notice:true},
  ],true,'Counsel signed the destruction order for RT-240.','Active holds and future expiry disqualify records; selected items sort by expiry date then complete ID.'),
  variant('records-retention-disposition','records-unauthorized-notice-scope','RT-241',[
    {id:'REC-R',expiry:'2021-06-15',hold:false,notice:true}, {id:'REC-S',expiry:'2021-05-15',hold:false,notice:true},
    {id:'REC-T',expiry:'2020-01-01',hold:false,notice:false}, {id:'REC-U',expiry:'2018-01-01',hold:true,notice:true},
  ],false,'Counsel has not signed a destruction order for RT-241.','Missing notice or an active hold prevents selection; the absent signature affects only disposition.'),
  variant('records-retention-disposition','records-empty-active-holds','RT-242',[
    {id:'REC-V',expiry:'2010-01-01',hold:true,notice:true}, {id:'REC-W',expiry:'2020-01-01',hold:false,notice:false},
    {id:'REC-X',expiry:'2027-01-01',hold:false,notice:true},
  ],true,'Counsel signed the destruction order for RT-242.','An active hold, incomplete notice, or expiry after cutoff makes a record ineligible even when counsel has signed.'),
];
