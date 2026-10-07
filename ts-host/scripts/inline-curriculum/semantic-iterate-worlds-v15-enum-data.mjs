import { worlds as authoredWorlds } from './semantic-iterate-worlds-v15-data.mjs';

// These values are transcribed from the authored field contract strings. The
// intermediate set includes working placeholders; final is the contract's
// allowed terminal value set. Non-categorical fields remain ordinary strings.
const enumContracts = {
  service_amendment_payment: { decision: { intermediate: ['pending', 'release', 'hold'], final: ['release', 'hold'] } },
  community_grant_match_cap: { decision: { intermediate: ['pending', 'approve', 'hold'], final: ['approve', 'hold'] } },
  archive_channel_consent: {
    permittedChannels: {
      intermediate: ['unknown', 'public web', 'reading room', 'public web, reading room'],
      final: ['public web', 'reading room', 'public web, reading room'],
    },
    requestedChannel: { intermediate: ['unknown', 'public web', 'reading room'], final: ['public web', 'reading room'] },
    decision: { intermediate: ['pending', 'publish', 'withhold'], final: ['publish', 'withhold'] },
  },
  accessible_route_substitution: {
    routeStepFree: { intermediate: ['pending', 'yes', 'no'], final: ['yes', 'no'] },
    routeOperational: { intermediate: ['pending', 'yes', 'no'], final: ['yes', 'no'] },
    decision: { intermediate: ['pending', 'confirm', 'hold'], final: ['confirm', 'hold'] },
  },
  procurement_eligible_bid: {
    certificateStatus: { intermediate: ['pending', 'current', 'expired'], final: ['current', 'expired'] },
    conflictStatus: { intermediate: ['pending', 'cleared'], final: ['cleared'] },
    decision: { intermediate: ['pending', 'award', 'hold'], final: ['award', 'hold'] },
  },
  public_notice_correction_deadline: {
    waiverStatus: { intermediate: ['pending', 'none', 'recorded'], final: ['none', 'recorded'] },
    decision: { intermediate: ['pending', 'accept', 'reject'], final: ['accept', 'reject'] },
  },
  course_accommodation_version: { decision: { intermediate: ['pending', 'approve', 'hold'], final: ['approve', 'hold'] } },
  protected_species_field_permit: {
    protectedStatus: { intermediate: ['pending', 'listed', 'not-listed'], final: ['listed', 'not-listed'] },
    closureApplies: { intermediate: ['pending', 'yes', 'no'], final: ['yes', 'no'] },
    exceptionStatus: { intermediate: ['pending', 'none', 'recorded'], final: ['none', 'recorded'] },
    decision: { intermediate: ['pending', 'issue', 'deny'], final: ['issue', 'deny'] },
  },
  drought_household_allocation: { decision: { intermediate: ['pending', 'issue', 'hold'], final: ['issue', 'hold'] } },
  vaccine_logger_correction: { decision: { intermediate: ['pending', 'release', 'quarantine'], final: ['release', 'quarantine'] } },
  demand_response_meter_settlement: { decision: { intermediate: ['pending', 'pay', 'hold'], final: ['pay', 'hold'] } },
  transit_mileage_reimbursement: { decision: { intermediate: ['pending', 'reimburse', 'hold'], final: ['reimburse', 'hold'] } },
};

export const worlds = authoredWorlds.map(world => ({
  ...world,
  fields: world.slug === 'archive_channel_consent' ? {
    ...world.fields,
    permittedChannels: 'channel enum values are exactly “public web” and “reading room”; normalize “reading-room” to “reading room” and join multiple values with comma-space in canonical order “public web” then “reading room”',
  } : world.fields,
  evidence: world.slug === 'course_accommodation_version' ? {
    ...world.evidence,
    'pass-04-coordinator.md': world.evidence['pass-04-coordinator.md'].replace('for AC-73A and BIO-42 version C8', 'for AC-73 plan and AC-73A addendum covering BIO-42 version C8'),
  } : world.evidence,
  field_enums: enumContracts[world.slug],
}));
