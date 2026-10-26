export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type VolunteerMission = RecordMeta & WorkRecord & {
  missionId: string;
  objective: string;
  volunteerIds: string[];
  coordinatorId: string;
  missionWindow: string;
};
export type MissionRequirement = RecordMeta & WorkRecord & {
  caseId: string;
  task: string;
  requiredSkill: string;
  accessCondition: string;
  completionEvidence: string;
};
export type VolunteerCapability = RecordMeta & WorkRecord & {
  caseId: string;
  volunteerId: string;
  statedSkills: string[];
  verifiedQualifications: string[];
  availability: string;
  consentScope: string;
};
export type VolunteerTeam = RecordMeta & WorkRecord & {
  caseId: string;
  volunteerIds: string[];
  requirementIds: string[];
  coverageGaps: string[];
  supervisionPlan: string;
};
export type VolunteerBriefing = RecordMeta & WorkRecord & {
  caseId: string;
  teamId: string;
  roleInstructions: string[];
  escalationContactIds: string[];
  acknowledgmentRequired: boolean;
};
export type VolunteerRevision = RecordMeta & WorkRecord & {
  caseId: string;
  changedVolunteerIds: string[];
  affectedRoleIds: string[];
  replacementTeamId: string;
  acceptanceRefs: string[];
};
export type DraftMissionRequirement = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    task?: string;
    requiredSkill?: string;
    accessCondition?: string;
    completionEvidence?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type DeploymentCoordinator = { id: string };
export type MissionChange = { withdrawals: EvidenceItem[]; qualificationUpdates: EvidenceItem[]; modeChanges: EvidenceItem[]; accessInformation: EvidenceItem[]; completions: EvidenceItem[] };
export type MissionDossier = { mission: VolunteerMission; requirements: MissionRequirement[]; capabilities: VolunteerCapability[]; teams: VolunteerTeam[]; briefings: VolunteerBriefing[]; revisions: VolunteerRevision[]; evidence: EvidenceItem[]; checkpoint: MissionCheckpoint };
export type CapabilityCoverage = { requirementId: string; coveredBy: string[]; missingQualification: string[]; expiredQualification: string[]; availabilityConflicts: string[]; consentConflicts: string[]; };
export type SupervisionEdge = { traineeId: string; supervisorId: string; requirementId: string; window: string; verified: boolean; evidenceIds: string[] };
export type BriefingProjection = { volunteerId: string; role: string; disclosedAccessRefs: string[]; disclosureWindow: string; acknowledgments: string[]; };
export type MissionCheckpoint = { revision: string; mission: VolunteerMission; requirements: MissionRequirement[]; capabilities: VolunteerCapability[]; teams: VolunteerTeam[]; supervision: SupervisionEdge[]; briefings: BriefingProjection[]; withdrawals: string[]; completionReceipts: string[]; };
export type DeploymentProposal = { application: 'Volunteer Deployment Coordinator'; status: 'ready-for-review'|'blocked'; coverage: CapabilityCoverage[]; team: VolunteerTeam; supervision: SupervisionEdge[]; briefings: BriefingProjection[]; revision: VolunteerRevision; blockers: string[]; actionBoundary: string; resumeState: MissionCheckpoint; };
