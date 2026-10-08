import { worldSpecs, caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';

const bySlug = new Map(worldSpecs.map(spec => [spec.slug, spec]));
const decisionMap = spec => `Decision mapping: if the final eligible selection is empty, use ${spec.noAction}, even if authorization is absent. If the selection is nonempty and every stated authorization condition is met, use ${spec.approvedAction}; if the selection is nonempty but authorization is absent or incomplete, use hold. Authorization does not change selectedItems or measure.`;

function paired(baseSlug, suffix, changes = {}) {
  const base = bySlug.get(baseSlug);
  if (!base) throw new Error(`unknown V18 parent world: ${baseSlug}`);
  const spec = {
    ...base,
    ...changes,
    slug: `${baseSlug}-v19-${suffix}`,
    sourceGroup: `v18:${baseSlug}:world`,
    candidates: changes.candidates ?? base.candidates,
  };
  spec.ruleText = `${changes.ruleText ?? base.ruleText} ${decisionMap(spec)}`;
  spec.decisionFormat = `${changes.decisionFormat ?? base.decisionFormat ?? `Return one bare literal: ${spec.finalDecisions.join(', ')}.`} ${decisionMap(spec)}`;
  spec.instruction = `${changes.instruction ?? base.instruction} The decision mapping is part of the requested rule; preserve the eligible selection separately from authorization. ${decisionMap(spec)}`;
  spec.authorityText = changes.authorityText ?? 'The authority record states only whether the required approval is present; it does not prescribe or replace the evidence-derived selection.';
  return caseFrom(spec);
}

export const worlds = [
  paired('software-release-gate', 'build-counterfactual', {
    requestId:'REL-641', owner:'Platform release desk',
    requestText:'Choose the newest candidate build eligible for production deployment. The deployment window is open until 18:00 UTC.',
    instruction:'Resolve the production release candidate for REL-641. A blocker waiver is scoped to the exact build named in it.',
    candidates:[
      {id:'BUILD-13',versionRank:13,tests:false,blockers:0,critical:false,waiverBuild:'none'},
      {id:'BUILD-12',versionRank:12,tests:true,blockers:0,critical:false,waiverBuild:'none'},
      {id:'BUILD-11',versionRank:11,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-11'},
      {id:'BUILD-10',versionRank:10,tests:true,blockers:0,critical:false,waiverBuild:'none'},
    ],
    authorityText:'Signed change order CO-891 covers REL-641. The deployment window remains open until 18:00 UTC.',
    authorized:true,
  }),
  paired('dependency-upgrade-batch', 'review-tie-counterfactual', {
    requestId:'DEP-220', owner:'Open-source security maintainers',
    requestText:'Select at most two dependency updates for the next maintenance batch.',
    instruction:'Prepare the DEP-220 upgrade batch and report selection separately from merge authority.',
    candidates:[
      {id:'PKG-A',reduction:90,reviewed:true,tests:true,conflict:false},
      {id:'PKG-B',reduction:99,reviewed:false,tests:true,conflict:false},
      {id:'PKG-C',reduction:90,reviewed:true,tests:true,conflict:false},
      {id:'PKG-D',reduction:87,reviewed:true,tests:true,conflict:false},
    ],
    authorityText:'Two distinct maintainer approvals are recorded for DEP-220.',
    authorized:true,
  }),
  paired('research-instrument-booking', 'calibration-tie-counterfactual', {
    requestId:'LAB-363', owner:'Shared laboratory scheduler',
    requestText:'Book one 90-minute instrument slot that ends before 11:00.',
    instruction:'Resolve LAB-363. The earliest eligible start time wins; break equal start times by ascending complete slot ID.',
    candidates:[
      {id:'SLOT-A',start:'08:00',end:'09:30',duration:90,instrument:'MIC-4',calibrated:false,training:true,conflict:false,afterHours:false,permit:'none'},
      {id:'SLOT-B',start:'08:30',end:'10:00',duration:90,instrument:'MIC-2',calibrated:true,training:true,conflict:false,afterHours:false,permit:'none'},
      {id:'SLOT-C',start:'08:30',end:'10:00',duration:90,instrument:'MIC-3',calibrated:true,training:true,conflict:false,afterHours:false,permit:'none'},
      {id:'SLOT-D',start:'08:10',end:'09:40',duration:90,instrument:'MIC-1',calibrated:true,training:false,conflict:false,afterHours:false,permit:'none'},
    ],
    authorityText:'The PI approved LAB-363 and the selected equipment is available.',
    authorized:true,
  }),
  paired('privacy-export-minimization', 'identifier-override-counterfactual', {
    requestId:'PX-106', owner:'Regional outcomes analytics unit',
    requestText:'The regional outcome report requests all five schema fields: region, age_band, patient_name, rare_condition, and outcome_code. The approved purpose is regional-outcomes.',
    instruction:'Derive the permitted field list for PX-106 under the purpose and direct-identifier rules. Report export approval separately.',
    candidates:[
      {id:'region',position:1,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
      {id:'age_band',position:2,requested:true,classification:'restricted',consentedPurposes:['regional-outcomes'],directIdentifier:false},
      {id:'patient_name',position:3,requested:true,classification:'public',consentedPurposes:['regional-outcomes'],directIdentifier:true},
      {id:'rare_condition',position:4,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
      {id:'outcome_code',position:5,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
    ],
    authorityText:'The data protection officer has approved export PX-106.',
    authorized:true,
  }),
  paired('incident-mitigation-selection', 'risk-cap-counterfactual', {
    requestId:'INC-904', owner:'Service incident commander',
    requestText:'Select at most two mitigation actions for outage INC-904 to reduce customer impact.',
    instruction:'Compute the mitigation selection for INC-904; a test waiver applies only to the exact action named.',
    candidates:[
      {id:'ACT-A',reduction:99,prerequisites:true,blast:6,cap:5,tested:true,waiver:'none'},
      {id:'ACT-B',reduction:91,prerequisites:true,blast:4,cap:5,tested:false,waiver:'ACT-B'},
      {id:'ACT-C',reduction:91,prerequisites:true,blast:5,cap:5,tested:true,waiver:'none'},
      {id:'ACT-D',reduction:88,prerequisites:false,blast:2,cap:5,tested:true,waiver:'none'},
    ],
    authorityText:'The incident commander authorized execution of the INC-904 mitigation plan.',
    authorized:true,
  }),
  paired('research-sample-release', 'linked-retest-counterfactual', {
    requestId:'SR-449', owner:'Laboratory quality assurance',
    requestText:'Identify sample batches eligible for the next analysis run.',
    instruction:'Prepare the SR-449 eligible analysis batch list. A passing retest supersedes a failed initial QC only for the exact batch ID it names.',
    candidates:[
      {id:'SMP-Q',collected:'06:55',custody:true,initialQc:'fail',retest:'pass:SMP-Q'},
      {id:'SMP-R',collected:'07:05',custody:false,initialQc:'pass',retest:'none'},
      {id:'SMP-S',collected:'07:15',custody:true,initialQc:'pass',retest:'none'},
      {id:'SMP-T',collected:'07:20',custody:true,initialQc:'fail',retest:'pass:SMP-Q'},
    ],
    exceptionText:'A passing retest changes only the complete batch ID it names. A retest for SMP-Q cannot supersede SMP-T initial QC.',
    authorityText:'The QA lead signature for SR-449 is absent.',
    authorized:false,
  }),
  paired('volunteer-shift-coverage', 'role-waiver-counterfactual', {
    requestId:'VS-611', owner:'Community response coordinator',
    requestText:'Fill the two critical response roles: interpreter (priority 1) and radio operator (priority 2). A volunteer may fill at most one role.',
    instruction:'Build the maximum valid VS-611 assignments under qualification, availability, rest, exception-scope, and one-role-per-person rules.',
    candidates:[
      {id:'VOL-M/INTERPRETER',person:'VOL-M',role:'interpreter',priority:1,fit:99,qualified:true,available:true,rest:8,waiver:'VOL-M/radio'},
      {id:'VOL-N/INTERPRETER',person:'VOL-N',role:'interpreter',priority:1,fit:94,qualified:true,available:true,rest:11,waiver:'none'},
      {id:'VOL-N/RADIO',person:'VOL-N',role:'radio',priority:2,fit:100,qualified:true,available:true,rest:11,waiver:'none'},
      {id:'VOL-P/RADIO',person:'VOL-P',role:'radio',priority:2,fit:96,qualified:true,available:true,rest:12,waiver:'none'},
      {id:'VOL-Q/RADIO',person:'VOL-Q',role:'radio',priority:2,fit:98,qualified:false,available:true,rest:14,waiver:'none'},
    ],
    exceptionText:'A waiver names one exact person-role assignment; VOL-M/radio cannot waive VOL-M/interpreter rest. A volunteer may not be assigned twice, and a waiver cannot supply missing role qualification.',
    authorityText:'Coordinator approval is recorded for VS-611.',
    authorized:true,
  }),
  paired('records-retention-disposition', 'empty-selection-counterfactual', {
    requestId:'RT-228', owner:'Public records counsel',
    requestText:'Prepare a proposed records disposition list using cutoff date 2026-10-01. The list is not itself authority to destroy records.',
    instruction:'Compute the records that qualify for RT-228 and state separately whether destruction is authorized.',
    candidates:[
      {id:'REC-F',expiry:'2018-01-01',hold:true,notice:true},
      {id:'REC-G',expiry:'2019-01-01',hold:false,notice:false},
      {id:'REC-H',expiry:'2027-01-01',hold:false,notice:true},
    ],
    authorityText:'Counsel has not signed a destruction order for RT-228.',
    authorized:false,
  }),
];

// Same facts and gold as the V18 RT-227 row; this task variant makes the
// empty-selection/authority decision mapping explicit in the captured rule.
const rt227 = paired('records-retention-disposition', 'decision-map-contract', {
  requestId:'RT-227',
  instruction:'Compute the records that qualify for the RT-227 proposed destruction list and state separately whether destruction is authorized.',
  authorityText:'Counsel has not signed a destruction order for RT-227. Report the eligible proposed list, but do not represent destruction as authorized.',
  authorized:false,
});
worlds.push(rt227);
